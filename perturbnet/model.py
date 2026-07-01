from __future__ import annotations

import torch
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l

WEIGHTS = EfficientNet_V2_L_Weights.IMAGENET1K_V1
LABELS = [label for label in WEIGHTS.meta.get("categories", [])]
PREPROCESS = WEIGHTS.transforms()


def normalize_prediction_label(raw_label: str) -> str:
    return raw_label.strip().lower().replace("_", " ")


def _label_aliases(label: str) -> list[str]:
    normalized = normalize_prediction_label(label)
    aliases = [normalized]
    aliases.extend(part.strip() for part in normalized.split(",") if part.strip())
    return aliases


LABEL_TO_INDEX = {
    alias: idx
    for idx, label in enumerate(LABELS)
    for alias in _label_aliases(label)
}


def load_efficientnet_v2_l(device: torch.device) -> torch.nn.Module:
    try:
        model = efficientnet_v2_l(weights=WEIGHTS)
    except Exception as exc:
        raise RuntimeError("EfficientNetV2-L pretrained weights are required for scoring.") from exc
    return model.to(device).eval()


def resolve_target_index(target_label: str) -> int | None:
    return LABEL_TO_INDEX.get(normalize_prediction_label(target_label))


def _preprocess_for_efficientnet_v2_l(image_bchw: torch.Tensor) -> torch.Tensor:
    # Use torchvision's canonical transform pipeline for this exact weights variant.
    return PREPROCESS(image_bchw)


def predict_index(model: torch.nn.Module, image_chw: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(_preprocess_for_efficientnet_v2_l(image_chw.unsqueeze(0)))
        return int(logits.argmax(dim=1).item())


def logits_for_images(model: torch.nn.Module, image_bchw: torch.Tensor) -> torch.Tensor:
    return model(_preprocess_for_efficientnet_v2_l(image_bchw))


def label_for_index(index: int) -> str:
    if 0 <= int(index) < len(LABELS):
        return normalize_prediction_label(LABELS[int(index)])
    return str(index)


def predict_label(model: torch.nn.Module, image_chw: torch.Tensor) -> str:
    idx = predict_index(model=model, image_chw=image_chw)
    return label_for_index(idx)

