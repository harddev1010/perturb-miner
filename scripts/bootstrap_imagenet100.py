from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbnet.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT
from perturbnet.imagenet100_bootstrap import load_imagenet100


def main() -> int:
    dataset = load_imagenet100(repo_id=IMAGENET100_REPO_ID, split=IMAGENET100_SPLIT)
    print(
        f"ImageNet-100 ready: repo={IMAGENET100_REPO_ID} split={IMAGENET100_SPLIT} "
        f"images={int(dataset.num_rows)}"
    )
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
