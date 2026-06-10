from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbnet.imagenet100_bootstrap import (
    DEFAULT_MAX_IMAGES,
    DEFAULT_MIN_IMAGES,
    DEFAULT_REPO_ID,
    DEFAULT_SPLIT,
    bootstrap_imagenet100,
    env_bool,
    env_int,
    resolve_repo_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap local ImageNet-100 challenge images")
    parser.add_argument("--root", default=os.getenv("PERTURB_IMAGENET100_ROOT", "assets/imagenet-100"))
    parser.add_argument("--repo-id", default=os.getenv("PERTURB_IMAGENET100_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--split", default=os.getenv("PERTURB_IMAGENET100_SPLIT", DEFAULT_SPLIT))
    parser.add_argument(
        "--max-images",
        type=int,
        default=env_int("PERTURB_IMAGENET100_MAX_IMAGES", DEFAULT_MAX_IMAGES),
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=env_int("PERTURB_IMAGENET100_MIN_IMAGES", DEFAULT_MIN_IMAGES),
    )
    parser.add_argument("--force", action="store_true", default=env_bool("PERTURB_IMAGENET100_FORCE", False))
    args = parser.parse_args()

    bootstrap_imagenet100(
        root=resolve_repo_path(args.root),
        repo_id=str(args.repo_id),
        split=str(args.split),
        max_images=max(1, int(args.max_images)),
        min_images=max(1, int(args.min_images)),
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"ImageNet-100 bootstrap failed: {exc}", file=sys.stderr)
        exit_code = 1
    # Hard-exit: pyarrow/datasets streaming can leave non-joinable IO threads
    # that deadlock normal interpreter shutdown after a partial stream read.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
