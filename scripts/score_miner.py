"""score_miner.py — offline miner⇄validator scoring harness.

Replays the real challenges the miner has already seen through the CURRENT perturb() engine and scores
each one exactly the way the validator does, so you can measure an algorithm change against a fixed set
of real inputs without touching the live network.

Workflow it mimics:
  * MINER side — for each challenge it loads the clean image and runs neurons/perturb.perturb() with the
    same budget the miner uses (see neurons/miner._ATTACK_TIMEOUT_SECONDS).
  * VALIDATOR side — it re-derives the true label, quantises to the uint8 grid, checks the L∞/SSIM/PSNR
    gates and the label flip, then computes the full validator objective (perturbation + margin +
    novelty), mirroring validator.PerturbValidator.verify_and_score.

Challenge source: the LATEST N records the miner uploaded to the cloud store (R2) under
challenge_store.CHALLENGE_PREFIX (perturb/attack-challenges/; miner.py writes them, this script only
reads). Pass --dir to read a local folder of the same JSON records instead. This script NEVER writes
challenges.

Usage:
  python scripts/score_miner.py --challenge-count 200
  python scripts/score_miner.py --challenge-count 50 --attack-timeout 10
  python scripts/score_miner.py --dir /path/to/records --challenge-count 200

Cloud mode uses the same storage env as the miner (PERTURB_STORAGE_BACKEND/BUCKET/keys/endpoint).
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

import torch

from neurons.perturb import perturb
from neurons.perturb.utils import (
    compute_psnr_db,
    compute_ssim,
    cw_margin,
    perturbation_score,
    validator_score,
)
from perturbnet import challenge_store
from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64, image_url_to_b64, quantize_image_uint8_grid
from perturbnet.model import (
    load_efficientnet_v2_l,
    logits_for_images,
    predict_label,
    resolve_target_index,
)

logger = logging.getLogger("score_miner")


# ==========================================================================================
# Validator-faithful scoring (mirrors validator.PerturbValidator.verify_and_score).
# ==========================================================================================
def score_response(
    *,
    model: torch.nn.Module,
    device: torch.device,
    clean_chw: torch.Tensor,
    adv_chw: torch.Tensor,
    true_index: int,
    epsilon: float,
    min_delta: float,
) -> dict:
    """Grade a perturbed image the way the validator does. Returns the full result row:
    reason, flipped, total_score, pert_score, margin (CW), linf, rmse, k, pixels, channels."""
    x_clean = quantize_image_uint8_grid(clean_chw.to(device))
    x_adv = quantize_image_uint8_grid(adv_chw.to(device))

    diff = x_adv - x_clean
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    k = int(round(linf * 255.0))
    changed = (x_clean != x_adv)
    pixels = int(changed.any(dim=0).sum().item())
    channels = int(changed.sum().item())
    effective_max = min(float(epsilon), float(C.MAX_LINF_DELTA))
    pert = perturbation_score(linf, rmse, effective_max)

    def row(reason: str, total: float, cw: float) -> dict:
        return {
            "reason": reason,
            "flipped": reason == "success",
            "total_score": float(total),
            "pert_score": float(pert),
            "margin": float(cw),
            "linf": float(linf),
            "rmse": float(rmse),
            "k": int(k),
            "pixels": int(pixels),
            "channels": int(channels),
        }

    if x_adv.shape != x_clean.shape:
        return row("shape_mismatch", 0.0, 0.0)
    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        return row("value_out_of_range", 0.0, 0.0)

    with torch.no_grad():
        logits = logits_for_images(model, x_adv.unsqueeze(0))[0]
    prediction_index = int(logits.argmax().item())
    cw = cw_margin(logits, true_index)  # logit_true - best_other; < 0 means flipped

    if linf < float(min_delta):
        return row("below_min_delta", 0.0, cw)
    if linf > effective_max:
        return row("above_max_delta", 0.0, cw)
    if prediction_index == int(true_index):
        return row("label_match_with_original", 0.0, cw)

    ssim = compute_ssim(x_clean, x_adv)
    if ssim < float(C.MIN_SSIM):
        return row("below_min_ssim", 0.0, cw)
    psnr = compute_psnr_db(x_clean, x_adv)
    if float(C.MIN_PSNR_DB) > 0.0 and psnr < float(C.MIN_PSNR_DB):
        return row("below_min_psnr_db", 0.0, cw)

    total = validator_score(linf, rmse, cw, pixels, effective_max)
    return row("success", total, cw)


# ==========================================================================================
# Challenge loading (latest-N records from the cloud store the miner uploads to, or a local folder).
# ==========================================================================================
def _load_cloud_records(count: int, netuid: int) -> list[dict]:
    from perturbnet.storage_uploader import ImageStorageUploader

    uploader = ImageStorageUploader(run_id="score-miner", netuid=int(netuid), uploader_hotkey="")
    latest = challenge_store.list_latest_challenges(uploader, count)
    logger.info(f"Found {len(latest)} latest challenge object(s) under {challenge_store.CHALLENGE_PREFIX}/")
    records: list[dict] = []
    for obj in latest:
        try:
            records.append(challenge_store.load_challenge(uploader, obj["key"]))
        except Exception as exc:
            logger.warning(f"Skipping unreadable record {obj['key']}: {exc}")
    return records


def _load_local_records(from_dir: str, count: int) -> list[dict]:
    root = Path(from_dir)
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    records: list[dict] = []
    for path in files[: max(0, int(count))]:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning(f"Skipping unreadable record {path}: {exc}")
    return records


def _clean_image_b64(record: dict, api_timeout: float) -> str:
    b64 = str(record.get("image_b64") or "")
    if b64:
        return b64
    image_url = str(record.get("image_url") or "")
    if not image_url:
        raise RuntimeError("record has neither image_b64 nor image_url")
    return image_url_to_b64(image_url, timeout_seconds=api_timeout)


# ==========================================================================================
# Report formatting.
# ==========================================================================================
# Extra, per-algorithm metric columns. Add/replace tuples here as the algorithm evolves — each is
# (header, width, value-fn(row)->str). `row` carries the challenge record + fresh score under "score".
EXTRA_COLUMNS: list[tuple[str, int, "callable"]] = [
    ("k", 3, lambda r: str(r["score"]["k"])),
    ("chan", 6, lambda r: str(r["score"]["channels"])),
    ("pix", 5, lambda r: str(r["score"]["pixels"])),
    ("time_s", 7, lambda r: f"{r['elapsed']:.2f}"),
    ("miner_sc", 8, lambda r: f"{r['miner_score']:.4f}" if r["miner_score"] is not None else "  -   "),
]


def _fmt_task_id(task_id: str, width: int) -> str:
    if len(task_id) <= width:
        return task_id.ljust(width)
    return ("…" + task_id[-(width - 1):]).ljust(width)


def print_report(rows: list[dict]) -> None:
    tid_w = 34
    headers = ["task_id".ljust(tid_w), "q1", "q2"]
    headers += [h.rjust(w) for h, w, _ in EXTRA_COLUMNS]
    headers += ["pert_score".rjust(10), "margin".rjust(9), "total_score".rjust(11)]
    header_line = "  ".join(headers)
    sep = "-" * len(header_line)

    print(sep)
    print(header_line)
    print(sep)
    for r in rows:
        s = r["score"]
        cells = [
            _fmt_task_id(str(r["task_id"]), tid_w),
            (" ✓" if r["q1_flip"] else " ·"),
            (" ✓" if r["q2_flip"] else " ·"),
        ]
        cells += [fn(r).rjust(w) for _, w, fn in EXTRA_COLUMNS]
        cells += [
            f"{s['pert_score']:.4f}".rjust(10),
            f"{s['margin']:+.3f}".rjust(9),
            f"{s['total_score']:.4f}".rjust(11),
        ]
        print("  ".join(cells))
    print(sep)

    total = len(rows)
    q1 = sum(1 for r in rows if r["q1_flip"])
    q2 = sum(1 for r in rows if r["q2_flip"])
    err = sum(1 for r in rows if not r["score"]["flipped"])
    avg = (sum(r["score"]["total_score"] for r in rows) / total) if total else 0.0
    flips = [r["score"]["total_score"] for r in rows if r["score"]["flipped"]]
    avg_flip = (sum(flips) / len(flips)) if flips else 0.0
    print(
        f"Total: {total}, Q1_Flip: {q1}, Q2_Flip: {q2}, "
        f"ERR: {err} (0.0000 score cases / no flip found)"
    )
    print(f"Average score: {avg:.4f}   (flips only: {avg_flip:.4f} over {len(flips)})")


# ==========================================================================================
# Main.
# ==========================================================================================
def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline miner/validator scoring harness")
    parser.add_argument("--challenge-count", type=int, default=200, help="Latest challenges to score (default 200)")
    parser.add_argument(
        "--attack-timeout",
        type=float,
        default=float(os.getenv("PERTURB_ATTACK_TIMEOUT_SECONDS") or "35.0"),
        help="Per-challenge perturb() budget in seconds (default matches the miner: 35).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="",
        help="Read challenge JSON records from a local folder instead of fetching from the cloud store (R2).",
    )
    parser.add_argument("--netuid", type=int, default=int(os.getenv("NETUID", "1")))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "WARNING"))
    return parser.parse_args()


def main() -> int:
    args = build_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    device = _resolve_device(args.device)
    api_timeout = float(C.PERTURB_API_TIMEOUT_SECONDS)

    print(f"[score_miner] loading EfficientNetV2-L on {device.type} …", file=sys.stderr)
    model = load_efficientnet_v2_l(device)

    if args.dir:
        print(f"[score_miner] loading up to {args.challenge_count} local records from {args.dir}", file=sys.stderr)
        records = _load_local_records(args.dir, args.challenge_count)
    else:
        print(
            f"[score_miner] fetching latest {args.challenge_count} challenges from cloud "
            f"({challenge_store.CHALLENGE_PREFIX}/) …",
            file=sys.stderr,
        )
        records = _load_cloud_records(args.challenge_count, args.netuid)

    if not records:
        print("No challenge records found. Run miner.py to populate them, or pass --dir.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for i, record in enumerate(records, start=1):
        task_id = str(record.get("task_id") or f"record-{i}")
        try:
            clean_b64 = _clean_image_b64(record, api_timeout)
            clean = decode_image_b64(clean_b64).to(device).clamp(0.0, 1.0)
        except Exception as exc:
            print(f"[{i}/{len(records)}] task={task_id} SKIP (image load failed: {exc})", file=sys.stderr)
            continue

        # Mimic the miner: derive the true class from the clean image (fall back to the stored index).
        true_index = resolve_target_index(predict_label(model, clean))
        if true_index is None:
            true_index = int(record.get("target_index", -1))
        if true_index is None or true_index < 0:
            print(f"[{i}/{len(records)}] task={task_id} SKIP (unresolved true label)", file=sys.stderr)
            continue

        epsilon = float(record.get("epsilon", C.MAX_LINF_DELTA))
        min_delta = float(record.get("min_delta", C.MIN_LINF_DELTA))

        t0 = time.time()
        adv = perturb(
            model, clean, int(true_index), epsilon, min_delta, device,
            timeout_seconds=float(args.attack_timeout), start_time=t0,
        )
        elapsed = time.time() - t0

        score = score_response(
            model=model, device=device, clean_chw=clean, adv_chw=adv,
            true_index=int(true_index), epsilon=epsilon, min_delta=min_delta,
        )
        flipped = score["flipped"]
        k = score["k"]
        miner_meta = record.get("miner") or {}
        miner_score = float(miner_meta["score"]) if isinstance(miner_meta, dict) and "score" in miner_meta else None

        rows.append({
            "task_id": task_id,
            # Fixed-q invariant: a q=1 flip lands at k=1, the q=2 fallback at k>=2.
            "q1_flip": bool(flipped and k <= 1),
            "q2_flip": bool(flipped and k >= 2),
            "elapsed": elapsed,
            "miner_score": miner_score,
            "score": score,
        })
        print(
            f"[{i}/{len(records)}] task={task_id} k={k} flip={flipped} "
            f"total={score['total_score']:.4f} margin={score['margin']:+.3f} elapsed={elapsed:.2f}s",
            file=sys.stderr,
        )

    if not rows:
        print("No challenges scored.", file=sys.stderr)
        return 1

    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
