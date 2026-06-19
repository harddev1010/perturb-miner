"""Integration smoke for the current miner against captured error cases.

Replays the challenges that miner.py dumped when it found no flip (one JSON per case, see
neurons/miner._dump_error_case) through the CURRENT perturb engine — no dataset fetch, no label
inference; the target label and all params come straight from each JSON. Challenges are de-duplicated
by clean_image_b64 so repeated captures of the same image are tested once.

For each unique challenge it runs perturb() exactly as miner.forward() does (byte round-trip + flip
check) and independently verifies the perturbed image misclassifies under the model. Prints a per-case
line plus a summary so you can see whether the current miner now handles the previously-failing inputs.

Usage:
  python scripts/miner_ecase_smoke.py --input-dir /workspace/Perturb_error_cases [--limit N] [--timeout S] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_DIR = "/workspace/Perturb_error_cases"
_HALF_BYTE = 0.5 / 255.0


def load_challenges(input_dir: str):
    """Read every *.json case in input_dir, de-duplicating on clean_image_b64. Returns
    (challenges, n_duplicates, n_skipped)."""
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
        if data.get("processed") is not True:
            skipped += 1
            continue

        seen.add(b64)
        challenges.append((path, data))
    return challenges, dups, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay captured error cases through the current miner")
    ap.add_argument("--input-dir", default=DEFAULT_DIR, help="dir of {timestamp}_{task_id}.json cases")
    ap.add_argument("--limit", type=int, default=0, help="cap number of unique challenges (0 = all)")
    ap.add_argument("--timeout", type=float, default=0.0, help="override timeout_seconds (0 = per-case value)")
    ap.add_argument("--find-budget", type=float, default=10.0,
                    help="PERTURB_FIND_FLIP_BUDGET seconds for the search (default 20; raised from the 6.0 prod default)")
    ap.add_argument("--verbose", action="store_true", help="show perturb()'s own INFO logs")
    args = ap.parse_args()

    # Constants are read at import time, so set the budget BEFORE neurons.perturb is imported below.
    os.environ["PERTURB_FIND_FLIP_BUDGET"] = str(args.find_budget)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s | %(message)s")

    challenges, dups, skipped = load_challenges(args.input_dir)
    if args.limit > 0:
        challenges = challenges[:args.limit]
    print(f"[load] {len(challenges)} unique challenge(s) from {args.input_dir} "
          f"(deduped {dups}, skipped {skipped})")
    if not challenges:
        print("[load] nothing to test")
        return 0

    import torch

    from perturbnet.image_io import decode_image_b64, encode_image_b64
    from perturbnet.model import load_efficientnet_v2_l, logits_for_images, resolve_target_index
    from neurons.perturb import perturb
    from neurons.perturb.utils import estimate_validator_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device.type} cases={len(challenges)} find_budget={args.find_budget}s "
          f"(effective per case = min(timeout - reserve, find_budget))")
    model = load_efficientnet_v2_l(device=device)

    # Warm the kernels so the first case's timing isn't skewed by JIT/autotune.
    warm = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
    logits_for_images(model=model, image_bchw=warm).sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    flips = verified = returned_clean = errors = unresolved = 0
    rmses, scores, elapsed_all = [], [], []
    reason_counts: dict[str, int] = {}

    for i, (path, data) in enumerate(challenges):
        name = path.name
        reason = str(data.get("reason", "?"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        true_label = data.get("true_label")
        target_index = resolve_target_index(true_label) if true_label else None
        if target_index is None:
            unresolved += 1
            print(f"  [{i}] {name}: unresolved true_label={true_label!r} -> skip")
            continue

        try:
            clean = decode_image_b64(data["clean_image_b64"]).to(device).clamp(0.0, 1.0)
            epsilon = float(data.get("epsilon", 0.12))
            min_delta = float(data.get("min_delta", 0.002))
            timeout = args.timeout if args.timeout > 0 else float(data.get("timeout_seconds", 20))

            t0 = time.time()
            adv = perturb(model, clean, target_index, epsilon, min_delta, device,
                          timeout_seconds=timeout, start_time=t0)
            dt = time.time() - t0

            # Replicate miner.forward(): encode -> decode round-trip, then measure.
            seen = decode_image_b64(encode_image_b64(adv)).to(device)
            diff = seen - clean
            norm = float(diff.abs().max().item())
            rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
            nz = int((diff.abs() > _HALF_BYTE).sum().item())
            is_clean = norm < min_delta

            # Independent verdict: does the round-tripped image actually misclassify?
            with torch.no_grad():
                pred = int(logits_for_images(model=model, image_bchw=seen.unsqueeze(0))[0].argmax().item())
            flipped = pred != target_index
        except Exception as err:
            errors += 1
            print(f"  [{i}] {name}: ERROR {type(err).__name__}: {err}")
            continue

        elapsed_all.append(dt)
        if is_clean:
            returned_clean += 1
        else:
            flips += 1
            rmses.append(rmse)
            scores.append(estimate_validator_score(norm, rmse, epsilon))
        if flipped:
            verified += 1
            # Mark the successfully-flipped case in its source JSON so repeated runs can see
            # it's already handled (processed) and at what attack budget it flipped.
            data["processed"] = True
            data["time"] = args.find_budget
            try:
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception as err:
                print(f"  [{i}] {name}: WARN could not mark processed: {type(err).__name__}: {err}")
        print(f"  [{i}] {name}: produced_flip={not is_clean} model_misclassified={flipped} "
              f"nz={nz} linf={norm:.6f} (~{norm * 255:.2f}/255) rmse={rmse:.6f} t={dt:.2f}s")

    n = len(challenges) - unresolved - errors
    print("\n[summary]")
    print(f"  reasons          : {reason_counts}")
    print(f"  unique tested    : {n} (unresolved {unresolved}, errors {errors})")
    if n > 0:
        print(f"  produced a flip  : {flips}/{n} ({100.0 * flips / n:.1f}%)  still-clean {returned_clean}")
        print(f"  model-verified   : {verified}/{n} ({100.0 * verified / n:.1f}%)")
        if scores:
            print(f"  mean rmse        : {sum(rmses) / len(rmses):.6f}")
            print(f"  mean est_score   : {sum(scores) / len(scores):.4f}")
        if elapsed_all:
            print(f"  elapsed (s)      : mean {sum(elapsed_all) / len(elapsed_all):.2f} "
                  f"max {max(elapsed_all):.2f}")
        if flips != verified:
            print("  WARNING: produced-flip vs model-verified disagree — check the transfer/round-trip gate")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"Integration smoke failed: {exc}", file=sys.stderr)
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
