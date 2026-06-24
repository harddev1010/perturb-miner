"""Tunable constants for the perturb attack engine.

Every knob is an env var (PERTURB_*) so behavior is switchable at runtime without a redeploy.
Grouped by concern: core / accept-gate / per-approach. Defaults are the shipping values.

Only three approaches remain (see perturb.py): find_apgd_dlr, find_dct_apgd, find_hybrid.
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


def _env_ints(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return tuple(int(p) for p in raw.replace(" ", "").split(",") if p)
    except ValueError:
        return default


def _env_floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return tuple(float(p) for p in raw.replace(" ", "").split(",") if p)
    except ValueError:
        return default


# --- Core -------------------------------------------------------------------------------
Q = 1.0 / 255.0                                            # one byte in [0,1] space
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)  # validator L∞ cap
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 4.5)   # deadline headroom
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

# --- find_apgd_dlr: exact-byte APGD-DLR over the full ternary cube ----------------------
# Auto-PGD on the DLR loss, every candidate projected to the exact ternary byte cube {-1,0,+1}. Many
# diverse restarts in one batch (zero / CE / DLR / soft / sparse+dense random); the latent step halves
# on stall and the population refreshes around the best real-margin candidate. Returns safe on
# margin<=-kappa, else best margin<0; logs the best margin reached as an infeasibility signal.
APGD_TOPM = _env_int("PERTURB_APGD_TOPM", 5)
APGD_TAU = _env_float("PERTURB_APGD_TAU", 1.0)
APGD_ALPHA0 = _env_float("PERTURB_APGD_ALPHA0", 1.0)          # initial latent step (byte units)
APGD_MIN_ALPHA = _env_float("PERTURB_APGD_MIN_ALPHA", 0.05)
APGD_PATIENCE = _env_int("PERTURB_APGD_PATIENCE", 2)          # stalled iters before halving alpha
APGD_TOPK = _env_int("PERTURB_APGD_TOPK", 4)                  # candidates that get a gradient step per iter
APGD_SPARSE_STARTS = _env_floats("PERTURB_APGD_SPARSE_STARTS", (0.01, 0.05, 0.20))
APGD_DENSE_STARTS = _env_int("PERTURB_APGD_DENSE_STARTS", 2)
APGD_MUT = _env_int("PERTURB_APGD_MUT", 4)                    # ternary mutations around the best each iter
APGD_MAX_ITERS = _env_int("PERTURB_APGD_MAX_ITERS", 0)        # 0 = until budget

# --- find_dct_apgd: low-frequency filtered-gradient APGD-DLR (Method A) ------------------
# Same APGD machinery, but each gradient step is low-pass-filtered through a 2-D DCT: keep a top-left
# coefficient block, inverse-transform, sign. The mask cycles through DCT_MASK_RATIOS (1/8 -> 1/4 -> 3/8)
# so it starts very smooth/global and widens toward medium detail. Search magnitude stays ±1 byte.
DCT_TOPM = _env_int("PERTURB_DCT_TOPM", 5)
DCT_TAU = _env_float("PERTURB_DCT_TAU", 1.0)
DCT_ALPHA0 = _env_float("PERTURB_DCT_ALPHA0", 1.0)           # initial latent step (byte units)
DCT_MIN_ALPHA = _env_float("PERTURB_DCT_MIN_ALPHA", 0.05)
DCT_PATIENCE = _env_int("PERTURB_DCT_PATIENCE", 2)           # stalled iters before halving alpha
DCT_TOPK = _env_int("PERTURB_DCT_TOPK", 4)                   # candidates that get a gradient step per iter
DCT_SPARSE_STARTS = _env_floats("PERTURB_DCT_SPARSE_STARTS", (0.05, 0.20))
DCT_DENSE_STARTS = _env_int("PERTURB_DCT_DENSE_STARTS", 1)
DCT_MUT = _env_int("PERTURB_DCT_MUT", 2)                     # ternary mutations around the best each iter
DCT_MAX_ITERS = _env_int("PERTURB_DCT_MAX_ITERS", 0)        # 0 = until budget
# Retained low-frequency block per dimension, as a fraction of H/W (top-left ceil(ratio·H)×ceil(ratio·W)).
DCT_MASK_RATIOS = _env_floats("PERTURB_DCT_MASK_RATIOS", (0.125, 0.25, 0.375))
# Sparse top-k one-shot probe: keep only the top (ratio·valid) filtered-saliency channels (k-ladder), so
# the DCT stage can land a SPARSE flip directly instead of signing the whole filtered gradient (dense).
DCT_PREFIX_RATIOS = _env_floats("PERTURB_DCT_PREFIX_RATIOS", (0.001, 0.003, 0.01, 0.03, 0.10, 0.30))

# --- find_hybrid: APGD-DLR -> DCT-APGD -> targeted-DLR repair -> RMSE prune --------------
# The recommended configuration. Wall-clock fractions of the remaining budget split phases 1-3; phase 4
# (pruning) takes the remainder and runs against the real deadline. Any phase that finds an envelope-safe
# flip short-circuits straight to pruning.
HYBRID_APGD_FRAC = _env_float("PERTURB_HYBRID_APGD_FRAC", 0.45)     # phase 1: full APGD-DLR
HYBRID_DCT_FRAC = _env_float("PERTURB_HYBRID_DCT_FRAC", 0.30)       # phase 2: low-frequency DCT-APGD
HYBRID_REPAIR_FRAC = _env_float("PERTURB_HYBRID_REPAIR_FRAC", 0.10) # phase 3: targeted-DLR repair
HYBRID_TARGETS = _env_int("PERTURB_HYBRID_TARGETS", 3)             # runner-up classes attacked in repair
HYBRID_MOMENTUM = _env_float("PERTURB_HYBRID_MOMENTUM", 0.75)      # targeted-repair APGD momentum

# --- compress_l0: exact-byte L0 continuation (the RMSE optimizer, hybrid Phase 4) -------
# Keeps optimizing the SUPPORT after the first flip (reinforce -> adaptive shrink -> slack-aware group
# prune -> one-for-many exchange -> exact leave-one-out) under a locked target margin. At q=1,
# RMSE = q·sqrt(|S|/n), so this is the only stage that actually drives RMSE down. Geometric ladders here
# share PRUNE_LADDER (below) as their base.
L0_RHO0 = _env_float("PERTURB_L0_RHO0", 0.8)            # initial shrink budget k_try = floor(rho·k)
L0_RHO_MIN = _env_float("PERTURB_L0_RHO_MIN", 0.5)      # most aggressive shrink (after repeated success)
L0_RHO_MAX = _env_float("PERTURB_L0_RHO_MAX", 0.95)     # gentlest shrink (after repeated failure)
L0_RHO_STEP = _env_float("PERTURB_L0_RHO_STEP", 0.05)   # rho adaptation step
L0_SWAP_FRACS = _env_floats("PERTURB_L0_SWAP_FRACS", (0.05, 0.10, 0.20))  # weak↔strong swap fractions (reinforce)
L0_EXCHANGE_ADDS = _env_int("PERTURB_L0_EXCHANGE_ADDS", 32)       # top-N unused channels probed as the addition
L0_EXCHANGE_MIN_DROP = _env_int("PERTURB_L0_EXCHANGE_MIN_DROP", 2)  # need >=this removals per addition to net-shrink
L0_LOO_MAX = _env_int("PERTURB_L0_LOO_MAX", 2048)      # max |S| for the exact leave-one-out cleanup pass

# Geometric base for the byte-removal-count ladders (group prune / exchange).
PRUNE_LADDER = _env_int("PERTURB_PRUNE_LADDER", 2)
