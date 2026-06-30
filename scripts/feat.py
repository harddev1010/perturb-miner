"""Inspect EfficientNetV2-L's last hidden layer for a single AttackChallenge.

Loads an ``AttackChallenge`` JSON payload, decodes its clean image, runs it
through the model used by the subnet, and reports:

  * how many features live in the last hidden layer (the pooled embedding),
  * the shape of the final classifier weight / bias,
  * how much that last hidden layer contributes to the winning logit, as a
    concrete number (and its share of the total logit).

Usage:
    python scripts/feat.py --input-challenge challenge.json
"""

from __future__ import annotations

import argparse
import json

import torch

from perturbnet.image_io import decode_image_b64
from perturbnet.model import (
    LABELS,
    PREPROCESS,
    load_efficientnet_v2_l,
    normalize_prediction_label,
)
from perturbnet.protocol import AttackChallenge


def _load_challenge(path: str) -> AttackChallenge:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # Synapse subclasses are pydantic models; only feed known fields.
    fields = set(AttackChallenge.model_fields)
    return AttackChallenge(**{k: v for k, v in payload.items() if k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-challenge",
        required=True,
        help="Path to a JSON file containing an AttackChallenge.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    challenge = _load_challenge(args.input_challenge)
    model = load_efficientnet_v2_l(device)

    # ── Final classifier parameters ────────────────────────────────────────
    fc = model.classifier[1]  # Linear(1280, 1000)
    fc_weight = fc.weight.detach()  # (1000, 1280)
    fc_bias = fc.bias.detach()  # (1000,)
    num_features = fc_weight.shape[1]

    # ── Hook the last hidden layer (post avgpool+flatten, pre-linear) ───────
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, output):
        captured["embedding"] = output.detach()

    handle = model.classifier[0].register_forward_hook(hook_fn)

    # ── Forward pass on the challenge's clean image ─────────────────────────
    image_chw = decode_image_b64(challenge.clean_image_b64).to(device)
    with torch.no_grad():
        logits = model(PREPROCESS(image_chw.unsqueeze(0)))
    handle.remove()

    emb = captured["embedding"]  # (1, 1280)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(logits.argmax(dim=1).item())
    pred_label = LABELS[pred_idx] if 0 <= pred_idx < len(LABELS) else str(pred_idx)

    # ── Influence of the last hidden layer on the winning logit ─────────────
    # logit_c = (emb · W[c]) + b[c]
    feature_contrib = float((emb[0] * fc_weight[pred_idx]).sum())
    bias_contrib = float(fc_bias[pred_idx])
    total_logit = feature_contrib + bias_contrib
    share = feature_contrib / total_logit if total_logit != 0 else float("nan")

    # Sanity: recomputed logit should match the model's own output.
    manual_match = bool(
        torch.allclose(logits[0, pred_idx], emb[0] @ fc_weight[pred_idx] + fc_bias[pred_idx], atol=1e-3)
    )

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"task_id:              {challenge.task_id}")
    print(f"model:                {challenge.model_name}")
    print(f"true_label:           {normalize_prediction_label(challenge.true_label)}")
    print(f"device:               {device}")
    print("")
    print("── Last hidden layer ───────────────────────────────────────")
    print(f"features (embedding): {num_features}            shape={tuple(emb.shape)}")
    print(f"fc weight:            {tuple(fc_weight.shape)}")
    print(f"fc bias:              {tuple(fc_bias.shape)}")
    print(f"embedding L2 norm:    {float(emb.norm()):.4f}")
    print("")
    print("── Prediction ──────────────────────────────────────────────")
    print(f"predicted label:      {pred_label} (idx={pred_idx})")
    print(f"confidence (softmax): {float(probs[0, pred_idx]):.6f}")
    print(f"winning logit:        {float(logits[0, pred_idx]):.6f}")
    print(f"manual recompute ok:  {manual_match}")
    print("")
    print("── Influence of last hidden layer on prediction ────────────")
    print(f"feature contribution: {feature_contrib:.6f}  (emb · W[pred])")
    print(f"bias contribution:    {bias_contrib:.6f}  (b[pred])")
    print(f"total logit:          {total_logit:.6f}")
    print(f"feature share:        {share * 100:.2f}%  of the winning logit")


if __name__ == "__main__":
    main()
