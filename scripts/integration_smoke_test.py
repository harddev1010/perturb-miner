from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _load_manifest_images(manifest_path: Path, root: Path) -> list[Path]:
    paths: list[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        value = stripped.split(",", 1)[0].split()[0]
        candidate = Path(value)
        paths.append(candidate if candidate.is_absolute() else root / candidate)
    return paths


def _first_local_image(root: Path, manifest: Path | None) -> Path:
    if manifest is not None:
        image_paths = _load_manifest_images(manifest_path=manifest, root=root)
    else:
        image_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
        )
    for path in image_paths:
        if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS:
            return path
    raise RuntimeError(f"No ImageNet-100 images found under {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Perturb subnet local integration smoke test")
    parser.add_argument("--imagenet100-root", default=os.getenv("PERTURB_IMAGENET100_ROOT", ""))
    parser.add_argument("--imagenet100-manifest", default=os.getenv("PERTURB_IMAGENET100_MANIFEST", ""))
    parser.add_argument("--imagenet100-repo-id", default=os.getenv("PERTURB_IMAGENET100_REPO_ID", "clane9/imagenet-100"))
    parser.add_argument("--imagenet100-split", default=os.getenv("PERTURB_IMAGENET100_SPLIT", "train"))
    args = parser.parse_args()

    print("[1/3] Load ImageNet-100 challenge candidate")
    manifest = _resolve_path(args.imagenet100_manifest) if args.imagenet100_manifest.strip() else None
    if args.imagenet100_root.strip() or manifest is not None:
        root = _resolve_path(args.imagenet100_root or "assets/imagenet-100")
        image_path = _first_local_image(root=root, manifest=manifest)
        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise RuntimeError(f"ImageNet-100 image is empty: {image_path}")
        print(f"  image={image_path}")
    else:
        from perturbnet.imagenet100_bootstrap import load_imagenet100

        dataset = load_imagenet100(repo_id=args.imagenet100_repo_id, split=args.imagenet100_split)
        example = dataset[0]
        buffer = io.BytesIO()
        example["image"].convert("RGB").save(buffer, format="JPEG", quality=95)
        image_bytes = buffer.getvalue()
        print(f"  dataset rows={int(dataset.num_rows)} sample=row 0")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    print("[2/3] Run EfficientNetV2-L inference")
    import torch

    from perturbnet.image_io import decode_image_b64
    from perturbnet.model import load_efficientnet_v2_l, predict_label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device=device)
    image = decode_image_b64(image_b64).to(device)
    prediction = predict_label(model=model, image_chw=image)
    print(f"  model_prediction={prediction}")

    print("[3/3] Challenge target label will use model prediction directly")
    if not prediction:
        raise RuntimeError("Model prediction is empty")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        exit_code = 1
    # Hard-exit: pyarrow/datasets can leave non-joinable IO threads that
    # deadlock normal interpreter shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
