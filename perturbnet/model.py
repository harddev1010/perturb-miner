from __future__ import annotations

import torch
from torchvision.models import EfficientNet_B5_Weights, efficientnet_b5
from torchvision.transforms.functional import resize

WEIGHTS = EfficientNet_B5_Weights.IMAGENET1K_V1
LABELS = [label.lower() for label in WEIGHTS.meta.get("categories", [])]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
MEAN = torch.tensor(WEIGHTS.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor(WEIGHTS.transforms().std, dtype=torch.float32).view(1, 3, 1, 1)

def load_efficientnet_b5(device: torch.device) -> torch.nn.Module:
    try:
        model = efficientnet_b5(weights=WEIGHTS)
    except Exception:
        # Keep model family stable even if pretrained weights are unavailable.
        model = efficientnet_b5(weights=None)
    return model.to(device).eval()


def resolve_target_index(target_label: str) -> int | None:
    return LABEL_TO_INDEX.get(target_label.strip().lower())


def normalize_prediction_label(raw_label: str) -> str:
    return raw_label.strip().lower().replace("_", " ")


def _preprocess_for_efficientnet_b5(image_bchw: torch.Tensor) -> torch.Tensor:
    resized = resize(image_bchw, [456, 456], antialias=True)
    mean = MEAN.to(device=resized.device, dtype=resized.dtype)
    std = STD.to(device=resized.device, dtype=resized.dtype)
    return (resized - mean) / std


def predict_index(model: torch.nn.Module, image_chw: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(_preprocess_for_efficientnet_b5(image_chw.unsqueeze(0)))
        return int(logits.argmax(dim=1).item())


def logits_for_images(model: torch.nn.Module, image_bchw: torch.Tensor) -> torch.Tensor:
    return model(_preprocess_for_efficientnet_b5(image_bchw))


def predict_label(model: torch.nn.Module, image_chw: torch.Tensor) -> str:
    idx = predict_index(model=model, image_chw=image_chw)
    if 0 <= idx < len(LABELS):
        return LABELS[idx]
    return str(idx)

