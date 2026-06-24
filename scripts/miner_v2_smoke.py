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


def load_challenges(input_dir: str, status: str = "all"):
    """Read every *.json case in input_dir, de-duplicating on clean_image_b64. Returns
    (challenges, n_duplicates, n_skipped).

    status filters by the case's "processed" flag:
      - "processed": only cases that already produced a flip (processed is True)
      - "errored":   only cases that did not produce a flip (processed is not True)
      - "all":       every case (default)
    """
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
        is_processed = data.get("processed") is True
        if status == "processed" and not is_processed:
            skipped += 1
            continue
        if status == "errored" and is_processed:
            skipped += 1
            continue
        if b64 in seen:
            dups += 1
            continue
        seen.add(b64)
        challenges.append((path, data))
    return challenges, dups, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay captured error cases through the current miner")
    ap.add_argument("--input-dir", default=DEFAULT_DIR, help="dir of {timestamp}_{task_id}.json cases")
    ap.add_argument("--limit", type=int, default=0, help="cap number of unique challenges (0 = all)")
    ap.add_argument("--status", choices=("processed", "errored", "all"), default="all",
                    help="which cases to pick: processed (already flipped), errored (no flip), all (default)")
    ap.add_argument("--verbose", action="store_true", help="show perturb()'s own INFO logs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s | %(message)s")

    challenges, dups, skipped = load_challenges(args.input_dir, args.status)
    if args.limit > 0:
        challenges = challenges[:args.limit]
    print(f"[load] {len(challenges)} unique challenge(s) from {args.input_dir} "
          f"(status={args.status}, deduped {dups}, skipped {skipped})")
    if not challenges:
        print("[load] nothing to test")
        return 0

    import torch

    from perturbnet.image_io import decode_image_b64, encode_image_b64
    from perturbnet.model import load_efficientnet_v2_l, logits_for_images, resolve_target_index
    from neurons.perturb import perturb
    from neurons.perturb.utils import estimate_validator_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device.type} cases={len(challenges)}")
    model = load_efficientnet_v2_l(device=device)

    # Warm the kernels so the first case's timing isn't skewed by JIT/autotune.
    warm = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
    logits_for_images(model=model, image_bchw=warm).sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    verified = errors = unresolved = 0
    scores = []

    for i, (path, data) in enumerate(challenges):
        name = path.name
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
            timeout = float(data.get("timeout_seconds", 20))

            t0 = time.time()
            adv = perturb(model, clean, target_index, epsilon, min_delta, device,
                          timeout_seconds=timeout, start_time=t0)

            # Replicate miner.forward(): encode -> decode round-trip, then measure.
            seen = decode_image_b64(encode_image_b64(adv)).to(device)
            diff = seen - clean
            norm = float(diff.abs().max().item())
            rmse = float(torch.sqrt(torch.mean(diff * diff)).item())

            # Independent verdict: does the round-tripped image actually misclassify?
            with torch.no_grad():
                pred = int(logits_for_images(model=model, image_bchw=seen.unsqueeze(0))[0].argmax().item())
            flipped = pred != target_index
            score = estimate_validator_score(norm, rmse, epsilon)
        except Exception as err:
            errors += 1
            print(f"  [{i}] {name}: ERROR {type(err).__name__}: {err}")
            continue

        if flipped:
            verified += 1
            # Mark the successfully-flipped case in its source JSON so repeated runs can see
            # it's already handled (processed).
            data["processed"] = True
            try:
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception as err:
                print(f"  [{i}] {name}: WARN could not mark processed: {type(err).__name__}: {err}")

        scores.append(score)
        print(f"  [{i}] {name}: score={score:.4f} rmse={rmse:.6f} norm={norm:.6f} success={flipped}")

    n = len(challenges) - unresolved - errors
    print("\n[summary]")
    print(f"  total cases   : {n} (unresolved {unresolved}, errors {errors})")
    print(f"  success cases : {verified}/{n}" + (f" ({100.0 * verified / n:.1f}%)" if n > 0 else ""))
    if scores:
        print(f"  avg score     : {sum(scores) / len(scores):.4f}")
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
