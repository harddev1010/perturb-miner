from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbnet.imagenet100_bootstrap import (
    DEFAULT_REPO_ID,
    DEFAULT_SPLIT,
    load_imagenet100,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download the full ImageNet-100 split for validator challenges")
    parser.add_argument("--repo-id", default=os.getenv("PERTURB_IMAGENET100_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--split", default=os.getenv("PERTURB_IMAGENET100_SPLIT", DEFAULT_SPLIT))
    args = parser.parse_args()

    dataset = load_imagenet100(repo_id=str(args.repo_id), split=str(args.split))
    print(f"ImageNet-100 ready: repo={args.repo_id} split={args.split} images={int(dataset.num_rows)}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"ImageNet-100 bootstrap failed: {exc}", file=sys.stderr)
        exit_code = 1
    # Hard-exit: pyarrow/datasets can leave non-joinable IO threads that
    # deadlock normal interpreter shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
