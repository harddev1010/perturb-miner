"""Tunable constants for the perturb attack engine.

Every knob is an env var (PERTURB_*) so behavior is switchable at runtime without a redeploy.
Grouped by concern: core / accept-gate / one-shot engine. Defaults are the shipping values.

One engine (see perturb.py): one_shot — a single-pass sparse ±k_min/255 flip (minimal flipping
gradient-prefix + a bounded iterated-FGSM re-linearization fallback).
"""

from __future__ import annotations

import os

import torch


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Core -------------------------------------------------------------------------------
Q = 1.0 / 255.0                                            # one byte in [0,1] space
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)  # validator L∞ cap
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 4.5)   # deadline headroom (base)
# Reserve scales with the measured per-forward cost so larger images / models leave enough post-search
# headroom for serialization + verification (added on top of RESERVE_SECONDS). t_step is one fwd+bwd.
RESERVE_FWD_MULT = _env_float("PERTURB_RESERVE_FWD_MULT", 6.0)
# Deadline gating. The budget guard accounts for BOTH a backward pass (t_step) and a real eval chunk
# (t_eval, measured live): out_of_budget <=> time_left <= 2·t_step + OOT_EVAL_MARGIN·t_eval. And batch_eval
# stops launching new chunks once time_left <= EVAL_TIME_MARGIN·t_eval, so a big candidate list can never
# run past the deadline mid-evaluation.
OOT_EVAL_MARGIN = _env_float("PERTURB_OOT_EVAL_MARGIN", 1.5)
EVAL_TIME_MARGIN = _env_float("PERTURB_EVAL_TIME_MARGIN", 1.25)
# ±1/255 edits on a grid-aligned clean image survive PNG exactly, so the round-trip is identity.
SKIP_ROUNDTRIP = _env_bool("PERTURB_SKIP_ROUNDTRIP", True)

# --- Accept gate (transfer safety) ------------------------------------------------------
# kappa: require margin <= -kappa. With the TF32 envelope on, the dominant drift axis is
# covered by construction, so kappa drops to the residual cushion.
MARGIN_BUFFER = _env_float("PERTURB_MINER_MARGIN_BUFFER", 0.01)
KAPPA_RESID = _env_float("PERTURB_KAPPA_RESID", 0.004)
# cuDNN-TF32 regime for the TF32-sensitive ops (forward + backward). The validator sets no backend
# flags, so it runs PyTorch defaults: cuDNN convolutions use TF32, matmul does not. EfficientNetV2-L is
# conv-dominated, so this mirrors the validator's logits. matmul stays off in all cases (see perturb.py).
# Default: on for CUDA (match the validator), off for CPU (TF32 does not exist there).
TF32_ON = _env_bool("PERTURB_TF32_ON", torch.cuda.is_available())
# Require the flip under BOTH cuDNN-TF32 regimes (CUDA only): the ambient regime (TF32_ON) AND its
# opposite. Brackets the residual cross-GPU/library drift the single regime might not capture.
TF32_ENVELOPE = _env_bool("PERTURB_TF32_ENVELOPE", True)
# 0 -> return only envelope-safe flips (else clean); 1 -> return any margin<0 flip (literal spec).
ALLOW_UNSAFE_FLIP = _env_bool("PERTURB_ALLOW_UNSAFE_FLIP", False)
# Candidates evaluated per forward pass (halved on OOM).
BATCH_SIZE = _env_int("PERTURB_BATCH_SIZE", 32)

# --- one_shot engine --------------------------------------------------------------------
# Re-linearization fallback: max iterated-FGSM steps taken when no single clean-gradient prefix flips
# (curved boundary). Each step re-linearizes in the ±q box; the first float-flip is then sparsified.
MAX_RELIN = _env_int("PERTURB_MAX_RELIN", 10)
