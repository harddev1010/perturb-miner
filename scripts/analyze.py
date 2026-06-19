"""Analyze captured error-case challenges to profile what each image needs from the finder.

Loads every {timestamp}_{task_id}.json in --input-dir, de-duplicates by clean_image_b64, and for each
unique image computes the metrics that actually discriminate between the finder algorithms:

  * Margin scale       — CW margin m0, DLR-normalized margin (how far from the boundary, scale-free).
  * Logit competition  — top-K logits/probs, gap to runner-up, gap z_top1..z_top3, softmax entropy,
                          # of classes within a small band (many near-competitors => soft/boundary).
  * Gradient geometry  — q=1 feasibility ratio (predicted dense margin drop / m0), linear k* (channels
                          needed), concentration (participation ratio, top-1% mass), movable fraction.
  * Spatial structure  — top-tile mass fraction (coherent => tiles) and low-frequency energy fraction
                          (=> low-freq family); per-channel R/G/B gradient split (=> color family).

Then it prints a per-image report + an aggregate summary, and recommends a finder per image from the
metrics. All gradient metrics use the same primitives the miner uses (neurons.perturb.utils).

Usage:
  python scripts/analyze.py --input-dir /workspace/Perturb_error_cases [--limit N] [--topk 10]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_DIR = "/workspace/Perturb_error_cases"
_Q = 1.0 / 255.0


def load_challenges(input_dir: str):
    """Read every *.json case, de-duplicating on clean_image_b64. Returns (challenges, dups, skipped)."""
    root = Path(input_dir)
    if not root.is_dir():
        return [], 0, 0
    challenges, seen, dups, skipped = [], set(), 0, 0
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        b64 = data.get("clean_image_b64")
        if not b64:
            skipped += 1
            continue
        if b64 in seen:
            dups += 1
            continue
        seen.add(b64)
        challenges.append((path.name, data))
    return challenges, dups, skipped


def analyze_one(model, device, data, topk):
    """Compute the finder-relevant metrics for one challenge. Returns a dict (or {'error': ...})."""
    import torch

    from perturbnet.image_io import decode_image_b64
    from perturbnet.model import LABELS, logits_for_images, resolve_target_index
    from neurons.perturb.utils import cw_margin, loss_grad, movable

    true_label = data.get("true_label")
    t = resolve_target_index(true_label) if true_label else None
    if t is None:
        return {"error": f"unresolved true_label={true_label!r}"}

    clean = decode_image_b64(data["clean_image_b64"]).to(device).clamp(0.0, 1.0)
    c, h, w = clean.shape
    n = clean.numel()

    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=clean.unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=0)

    # --- margin scale ---
    m0 = cw_margin(logits, t)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    z1, z3 = float(sorted_logits[0]), float(sorted_logits[2])
    others = logits.clone(); others[t] = float("-inf")
    best_wrong = int(others.argmax())
    dlr = m0 / (z1 - z3 + 1e-12)

    # --- logit competition ---
    entropy = float(-(probs * (probs + 1e-12).log()).sum())
    gap_runner = float(logits[t] - logits[best_wrong])            # == m0
    gap_top1_top3 = z1 - z3
    # how many classes sit within 1.0 logit of the best wrong class (near-competitors)
    near_band = int(((others >= others.max() - 1.0)).sum())
    true_rank = int((logits > logits[t]).sum())                   # 0 = model already correct-top

    topk = min(int(topk), logits.numel())
    top_list = [(int(sorted_idx[i]), float(sorted_logits[i]), float(probs[sorted_idx[i]])) for i in range(topk)]

    # --- gradient geometry (hard CW margin gradient: the one that decides the flip) ---
    _, move_dir, score = loss_grad(model, clean, t, "hard")
    mv = movable(clean.view(-1), move_dir)
    valid = int(mv.sum())
    score_mv = (score * mv).float()
    total_mass = float(score_mv.sum())
    pred_dense_drop = _Q * total_mass                              # linear margin drop if ALL movable move ±1
    feas_ratio = pred_dense_drop / max(m0, 1e-9)                   # >1 => linearly flippable by dense q=1
    # linear k*: channels (by |g|) whose cumulative q*Σ reaches m0
    sorted_score, _ = torch.sort(score_mv, descending=True)
    cs = torch.cumsum(sorted_score, dim=0)
    reach = (cs * _Q >= max(m0, 1e-9)).nonzero()
    k_star = int(reach[0]) + 1 if reach.numel() > 0 else -1        # -1 => unreachable linearly
    # concentration: participation ratio (effective # active channels) + top-1% mass fraction
    part_ratio = (total_mass ** 2) / (float((score_mv ** 2).sum()) + 1e-12)
    k1pct = max(1, n // 100)
    top1pct_mass = float(sorted_score[:k1pct].sum()) / (total_mass + 1e-12)
    movable_frac = valid / n

    # --- spatial structure ---
    sal = score.view(c, h, w).sum(0)                               # [H,W] saliency map
    tile = 8
    tile_mass, tot = 0.0, float(sal.sum()) + 1e-12
    for r in range(0, h, tile):
        for cc in range(0, w, tile):
            tile_mass = max(tile_mass, float(sal[r:r + tile, cc:cc + tile].sum()))
    top_tile_frac = tile_mass / tot                               # coherent => high
    # low-frequency energy fraction (centered low-pass over the 2D spectrum of the saliency map)
    spec = torch.fft.fftshift(torch.abs(torch.fft.fft2(sal)) ** 2)
    cy, cx, rad = h // 2, w // 2, max(1, min(h, w) // 8)
    low_energy = float(spec[cy - rad:cy + rad + 1, cx - rad:cx + rad + 1].sum())
    lowfreq_frac = low_energy / (float(spec.sum()) + 1e-12)
    # per-channel gradient mass split
    ch_mass = score.view(c, h, w).sum((1, 2))
    ch_frac = (ch_mass / (ch_mass.sum() + 1e-12)).tolist()

    return {
        "true_label": true_label, "true_idx": t, "true_rank": true_rank,
        "already_wrong": m0 < 0, "best_wrong": best_wrong, "best_wrong_label": LABELS[best_wrong],
        "m0": m0, "dlr": dlr, "p_true": float(probs[t]), "p_best_wrong": float(probs[best_wrong]),
        "entropy": entropy, "gap_runner": gap_runner, "gap_top1_top3": gap_top1_top3, "near_band": near_band,
        "top": top_list,
        "valid": valid, "movable_frac": movable_frac, "feas_ratio": feas_ratio, "k_star": k_star,
        "part_ratio": part_ratio, "top1pct_mass": top1pct_mass,
        "top_tile_frac": top_tile_frac, "lowfreq_frac": lowfreq_frac, "ch_frac": ch_frac,
        "n": n,
    }


def recommend(m):
    """Heuristic finder pick from the metrics (advisory — a router would encode these rules)."""
    if m["already_wrong"]:
        return "trivial (already misclassified — 1-channel in-band edit)"
    if m["feas_ratio"] < 1.0:
        return "find_apgd_dlr (capture) — dense q=1 can't linearly clear m0; may be q=1 INFEASIBLE"
    # feasible at q=1 — pick by structure
    concentrated = m["k_star"] != -1 and m["k_star"] <= max(1, m["n"] // 200) and m["top1pct_mass"] > 0.5
    if concentrated and m["m0"] < 2.0:
        return "find_beam_byte_pgd / prefix — concentrated saliency, low margin"
    if m["near_band"] >= 3 or m["gap_top1_top3"] < 1.0:
        return "find_ensemble_byte_pgd — many near-competitors (soft/DLR/targeted)"
    if m["top_tile_frac"] > 0.15 or m["lowfreq_frac"] > 0.5:
        return "find_hydra — coherent/low-frequency gradient structure (tiles/low-freq)"
    return "find_apgd_dlr / find_population_pgd — high margin, diffuse"


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile error-case challenges for finder selection")
    ap.add_argument("--input-dir", default=DEFAULT_DIR, help="dir of {timestamp}_{task_id}.json cases")
    ap.add_argument("--limit", type=int, default=0, help="cap number of unique challenges (0 = all)")
    ap.add_argument("--topk", type=int, default=10, help="top-K logits to print per image")
    ap.add_argument("--show-processed", action=argparse.BooleanOptionalAction, default=True,
                    help="annotate each case with its processed flag (set by the replay smoke); default: show")
    args = ap.parse_args()

    challenges, dups, skipped = load_challenges(args.input_dir)
    if args.limit > 0:
        challenges = challenges[:args.limit]
    print(f"[load] {len(challenges)} unique challenge(s) from {args.input_dir} (deduped {dups}, skipped {skipped})")
    if not challenges:
        print("[load] nothing to analyze")
        return 0

    import torch

    from perturbnet.model import LABELS, load_efficientnet_v2_l

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device.type} cases={len(challenges)}")
    model = load_efficientnet_v2_l(device=device)

    rows, rec_counts = [], {}
    for i, (name, data) in enumerate(challenges):
        m = analyze_one(model, device, data, args.topk)
        if "error" in m:
            print(f"\n[{i}] {name}: SKIP ({m['error']})")
            continue
        rec = recommend(m)
        rec_key = rec.split(" ")[0]
        rec_counts[rec_key] = rec_counts.get(rec_key, 0) + 1
        m["processed"] = bool(data.get("processed"))
        rows.append(m)

        proc_str = ""
        if args.show_processed:
            if m["processed"]:
                proc_str = f"  processed=YES(t={data.get('time')})"
            else:
                proc_str = "  processed=no"
        print(f"\n[{i}] {name}  true={m['true_label']}(idx {m['true_idx']}, rank {m['true_rank']}){proc_str}")
        print(f"     margin   : m0={m['m0']:+.3f}  dlr={m['dlr']:+.3f}  p_true={m['p_true']:.3f} "
              f"p_bestwrong={m['p_best_wrong']:.3f}  entropy={m['entropy']:.2f}")
        print(f"     compete  : gap_runner={m['gap_runner']:+.3f}  gap_top1_3={m['gap_top1_top3']:.3f} "
              f"near_band(<=1.0)={m['near_band']}  best_wrong={m['best_wrong_label']}")
        print(f"     gradient : feas_ratio={m['feas_ratio']:.2f}  k*={m['k_star']}  "
              f"({(100.0 * m['k_star'] / m['n']) if m['k_star'] > 0 else -1:.3f}% of {m['n']})  "
              f"part_ratio={m['part_ratio']:.0f}  top1%mass={m['top1pct_mass']:.2f}  movable={m['movable_frac']:.2f}")
        print(f"     spatial  : top_tile_frac={m['top_tile_frac']:.3f}  lowfreq_frac={m['lowfreq_frac']:.3f}  "
              f"RGB_mass=[{m['ch_frac'][0]:.2f},{m['ch_frac'][1]:.2f},{m['ch_frac'][2]:.2f}]")
        print(f"     top{args.topk:<5}: " + "  ".join(
            f"{('*' if idx == m['true_idx'] else '')}{LABELS[idx][:14]}:{lg:.2f}/{p:.2f}" for idx, lg, p in m["top"]))
        print(f"     -> RECOMMEND: {rec}")

    if rows:
        def mean(key):
            return sum(r[key] for r in rows) / len(rows)
        feasible = sum(1 for r in rows if r["feas_ratio"] >= 1.0 and not r["already_wrong"])
        infeasible = sum(1 for r in rows if r["feas_ratio"] < 1.0 and not r["already_wrong"])
        already = sum(1 for r in rows if r["already_wrong"])
        print("\n========== AGGREGATE ==========")
        print(f"  analyzed         : {len(rows)}")
        if args.show_processed:
            n_proc = sum(1 for r in rows if r["processed"])
            print(f"  processed        : {n_proc}/{len(rows)} flipped on a prior replay  ({len(rows) - n_proc} unprocessed)")
        print(f"  m0               : mean {mean('m0'):+.3f}  (min {min(r['m0'] for r in rows):+.3f}, "
              f"max {max(r['m0'] for r in rows):+.3f})")
        print(f"  dlr              : mean {mean('dlr'):+.3f}")
        print(f"  feas_ratio       : mean {mean('feas_ratio'):.2f}  "
              f"(>=1 linearly-feasible: {feasible}, <1 maybe-infeasible: {infeasible}, already-wrong: {already})")
        print(f"  k* (channels)    : mean {mean('k_star'):.0f}")
        print(f"  top1%mass        : mean {mean('top1pct_mass'):.2f}   part_ratio mean {mean('part_ratio'):.0f}")
        print(f"  top_tile_frac    : mean {mean('top_tile_frac'):.3f}   lowfreq_frac mean {mean('lowfreq_frac'):.3f}")
        print(f"  near_band mean   : {mean('near_band'):.1f}")
        print(f"  recommendations  : {rec_counts}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"analyze failed: {exc}", file=sys.stderr)
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
