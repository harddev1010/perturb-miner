from __future__ import annotations

import argparse
import base64
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


def _first_imagenet100_image(root: Path, manifest: Path | None) -> Path:
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
    parser.add_argument("--imagenet100-root", default=os.getenv("PERTURB_IMAGENET100_ROOT", "assets/imagenet-100"))
    parser.add_argument("--imagenet100-manifest", default=os.getenv("PERTURB_IMAGENET100_MANIFEST", ""))
    parser.add_argument("--imagenet100-repo-id", default=os.getenv("PERTURB_IMAGENET100_REPO_ID", "clane9/imagenet-100"))
    parser.add_argument("--imagenet100-split", default=os.getenv("PERTURB_IMAGENET100_SPLIT", "train"))
    parser.add_argument("--imagenet100-max-images", type=int, default=int(os.getenv("PERTURB_IMAGENET100_MAX_IMAGES", "5000")))
    parser.add_argument("--imagenet100-min-images", type=int, default=int(os.getenv("PERTURB_IMAGENET100_MIN_IMAGES", "1000")))
    parser.add_argument("--skip-imagenet100-bootstrap", action="store_true")
    args = parser.parse_args()

    print("[1/3] Load ImageNet-100 challenge candidate")
    root = _resolve_path(args.imagenet100_root)
    manifest = _resolve_path(args.imagenet100_manifest) if args.imagenet100_manifest.strip() else None
    if manifest is None and not args.skip_imagenet100_bootstrap:
        from perturbnet.imagenet100_bootstrap import bootstrap_imagenet100

        bootstrap_imagenet100(
            root=root,
            repo_id=args.imagenet100_repo_id,
            split=args.imagenet100_split,
            max_images=max(1, args.imagenet100_max_images),
            min_images=max(1, args.imagenet100_min_images),
            force=False,
        )
    image_path = _first_imagenet100_image(root=root, manifest=manifest)
    image_bytes = image_path.read_bytes()
    if not image_bytes:
        raise RuntimeError(f"ImageNet-100 image is empty: {image_path}")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    print(f"  image={image_path}")

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
    # Hard-exit: pyarrow/datasets streaming can leave non-joinable IO threads
    # that deadlock normal interpreter shutdown after a partial stream read.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
