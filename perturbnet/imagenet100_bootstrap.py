from __future__ import annotations

import hashlib
from typing import Any

from perturbnet.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT


def load_imagenet100(repo_id: str = IMAGENET100_REPO_ID, split: str = IMAGENET100_SPLIT) -> Any:
    """Download (once) and open the full ImageNet-100 split with random access.

    Uses the non-streaming `datasets` path: parquet shards are cached locally
    by Hugging Face and memory-mapped, so `dataset[i]` is cheap and the whole
    split (~126k train images) is available without exporting image files.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "`datasets` is required for ImageNet-100 challenges. "
            "Run `python -m pip install -r requirements.txt` first."
        ) from exc

    split_candidates = [split]
    for candidate in ("train", "validation", "val", "test"):
        if candidate not in split_candidates:
            split_candidates.append(candidate)

    errors: list[str] = []
    for split_name in split_candidates:
        try:
            print(
                f"Loading ImageNet-100 repo={repo_id} split={split_name} "
                "(first run downloads the full split to the local Hugging Face cache)"
            )
            return load_dataset(repo_id, split=split_name)
        except Exception as exc:
            errors.append(f"{split_name}: {exc}")

    raise RuntimeError("Unable to load ImageNet-100 dataset from Hugging Face: " + " | ".join(errors))


def imagenet100_dataset_version(dataset: Any, repo_id: str, split: str) -> str:
    """Stable short identifier for a downloaded dataset snapshot.

    Intentionally based only on repo/split/row-count: it stays identical
    across restarts, `datasets` library upgrades, and cache rebuilds, so the
    validator's persisted traversal is never spuriously reset (which would
    allow duplicate image selections).
    """
    base = f"{repo_id}:{split}:{int(dataset.num_rows)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
