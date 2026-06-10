from __future__ import annotations

import gc
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_REPO_ID = "clane9/imagenet-100"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_IMAGES = 5000
DEFAULT_MIN_IMAGES = 1000
SHUFFLE_BUFFER_SIZE = 500


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def existing_image_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _mix_streaming_dataset(dataset: Any) -> Any:
    """Interleave parquet shards and shuffle with a small buffer.

    The source split is stored in class order, so a capped sequential export
    would only contain the first few classes. Round-robin over shards plus a
    buffered shuffle spreads the export across all classes.
    """
    import random

    seed = random.SystemRandom().randrange(2**31)
    try:
        num_shards = int(getattr(dataset, "num_shards", getattr(dataset, "n_shards", 1)) or 1)
        if num_shards > 1:
            from datasets import interleave_datasets

            dataset = interleave_datasets(
                [dataset.shard(num_shards=num_shards, index=i) for i in range(num_shards)]
            )
    except Exception as exc:
        print(f"ImageNet-100 shard interleave unavailable, falling back to sequential stream: {exc}", file=sys.stderr)
    try:
        dataset = dataset.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE)
    except Exception as exc:
        print(f"ImageNet-100 stream shuffle unavailable: {exc}", file=sys.stderr)
    return dataset


def _load_dataset(repo_id: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "`datasets` is required to bootstrap ImageNet-100. "
            "Run `python -m pip install -r requirements.txt` first."
        ) from exc

    split_candidates = [split]
    for candidate in ("train", "validation", "val", "test"):
        if candidate not in split_candidates:
            split_candidates.append(candidate)

    errors: list[str] = []
    for split_name in split_candidates:
        try:
            print(f"Loading ImageNet-100 source repo={repo_id} split={split_name} streaming=true")
            dataset = load_dataset(repo_id, split=split_name, streaming=True)
        except Exception as exc:
            errors.append(f"{split_name}: {exc}")
            continue
        return _mix_streaming_dataset(dataset)

    raise RuntimeError("Unable to load ImageNet-100 dataset from Hugging Face: " + " | ".join(errors))


def _coerce_image(value: Any) -> Image.Image:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return Image.open(BytesIO(raw)).convert("RGB")
        path = value.get("path")
        if isinstance(path, str) and path.strip():
            return Image.open(path).convert("RGB")
    raise ValueError(f"Unsupported image payload type: {type(value).__name__}")


def _safe_label(value: Any) -> str:
    text = str(value if value is not None else "unknown").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def bootstrap_imagenet100(
    *,
    root: Path,
    repo_id: str,
    split: str,
    max_images: int,
    min_images: int,
    force: bool,
) -> int:
    existing = existing_image_count(root)
    if existing >= min_images and not force:
        print(f"ImageNet-100 cache ready: root={root} images={existing}")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(repo_id=repo_id, split=split)
    manifest_lines: list[str] = []
    written = 0

    iterator = iter(dataset)
    idx = -1
    while written < max_images:
        idx += 1
        try:
            example = next(iterator)
        except StopIteration:
            break
        try:
            image = _coerce_image(example.get("image"))
            label = _safe_label(example.get("label"))
            label_dir = root / label
            label_dir.mkdir(parents=True, exist_ok=True)
            output_path = label_dir / f"{written:06d}.jpg"
            image.save(output_path, format="JPEG", quality=95)
            manifest_lines.append(output_path.relative_to(root).as_posix())
            written += 1
            if written % 500 == 0:
                print(f"Prepared ImageNet-100 images: {written}/{max_images}")
        except Exception as exc:
            print(f"Skipping invalid ImageNet-100 sample idx={idx}: {exc}", file=sys.stderr)

    # Release the abandoned streaming iterator promptly so pyarrow IO threads
    # are not left mid-task (which can deadlock interpreter shutdown).
    del iterator
    del dataset
    gc.collect()

    if written < min_images:
        raise RuntimeError(
            f"ImageNet-100 bootstrap wrote only {written} images; expected at least {min_images}."
        )

    (root / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (root / ".source.json").write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "split": split,
                "max_images": max_images,
                "written": written,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ImageNet-100 bootstrap complete: root={root} images={written}")
    return written
