"""Tunable constants for the perturb attack engine.

Every knob is an env var (PERTURB_*) so behavior is switchable at runtime without a redeploy.
Grouped by concern: core / accept-gate / feasibility solver. Defaults are the shipping values.

One solver (see perturb.py): find_feasible — Multi-Target Ternary Projected Gradient Search (Problem 1).
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


def _env_strs(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return tuple(p for p in raw.replace(" ", "").split(",") if p)


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

# --- find_feasible: Multi-Target Ternary Projected Gradient Search (Problem 1) -----------
# Gradients propose actions/supports; an exact ternary projection keeps candidates legal; real discrete
# candidates are batch-evaluated and the first envelope-safe flip wins (cost is irrelevant for feasibility).
FEAS_TOPM = _env_int("PERTURB_FEAS_TOPM", 5)            # top runner-up classes for soft loss + multi-target
FEAS_TAU = _env_float("PERTURB_FEAS_TAU", 1.0)         # soft-margin (logsumexp) temperature
FEAS_LOSSES = _env_strs("PERTURB_FEAS_LOSSES", ("dlr", "soft"))  # untargeted losses, rotated in the loop
# Top-k support sweep: prefix sizes as multiples of the linearized crossing size K_c (medium supports can
# beat the fully dense candidate when dense edits interfere destructively).
FEAS_K_MULTS = _env_floats("PERTURB_FEAS_K_MULTS", (0.5, 0.75, 1.0, 1.25, 1.5))
FEAS_ALPHA0 = _env_float("PERTURB_FEAS_ALPHA0", 1.0)   # initial latent step (byte units)
FEAS_MIN_ALPHA = _env_float("PERTURB_FEAS_MIN_ALPHA", 0.05)
FEAS_PATIENCE = _env_int("PERTURB_FEAS_PATIENCE", 2)   # stalled iters before halving alpha + refresh
FEAS_TOPK = _env_int("PERTURB_FEAS_TOPK", 4)           # lowest-margin parents that get a gradient step
# Block-coordinate (macro coordinate descent) ladder: switch the top-B salient coords toward the flip.
FEAS_BLOCKS = _env_ints("PERTURB_FEAS_BLOCKS", (32, 128, 512, 2048))
FEAS_SPARSE_STARTS = _env_floats("PERTURB_FEAS_SPARSE_STARTS", (0.01, 0.05, 0.20))  # sparse random seeds
FEAS_DENSE_STARTS = _env_int("PERTURB_FEAS_DENSE_STARTS", 2)   # dense random seeds
FEAS_MUT = _env_int("PERTURB_FEAS_MUT", 4)             # gradient-free ternary mutations around the best

# --- find_feasible_upgraded: portfolio + diverse beam + adaptive blocks (Problem 1) ------
# Strict superset of find_feasible's ideas: a loss PORTFOLIO with reward-per-second allocation, a
# dynamically prioritized target pool (by linearized crossing size K̂_c), a diverse de-duplicated beam,
# many structured restarts, per-parent adaptive block sizes with replacement/reversal moves, gradient-
# accuracy + stagnation monitoring, gradient ensembles, support-size neighborhoods, near-tie sampling,
# and spatially structured proposals. Returns the instant an envelope-safe flip is banked.
# Solver selector: "feasible" (default, the simple solver) or "upgraded".
SOLVER = _env_strs("PERTURB_SOLVER", ("feasible",))[0]
FEASUP_LOSSES = _env_strs("PERTURB_FEASUP_LOSSES", ("dlr", "soft", "hard", "ce"))  # rotated portfolio
FEASUP_ENS_LOSSES = _env_strs("PERTURB_FEASUP_ENS_LOSSES", ("dlr", "soft"))        # ensemble members
FEASUP_TAU = _env_float("PERTURB_FEASUP_TAU", 1.0)            # soft-margin temperature
FEASUP_TOPM = _env_int("PERTURB_FEASUP_TOPM", 10)            # target pool size (top wrong classes)
FEASUP_RAND_TARGETS = _env_int("PERTURB_FEASUP_RAND_TARGETS", 2)  # extra random wrong classes in the pool
FEASUP_BEAM = _env_int("PERTURB_FEASUP_BEAM", 16)            # diverse near-flip beam capacity
FEASUP_DUP_JACCARD = _env_float("PERTURB_FEASUP_DUP_JACCARD", 0.9)  # support-overlap dedup threshold
FEASUP_K_MULTS = _env_floats("PERTURB_FEASUP_K_MULTS", (0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0))
FEASUP_BLOCK0 = _env_int("PERTURB_FEASUP_BLOCK0", 128)      # initial per-parent block size
FEASUP_BLOCK_MIN = _env_int("PERTURB_FEASUP_BLOCK_MIN", 16)
FEASUP_BLOCK_MAX = _env_int("PERTURB_FEASUP_BLOCK_MAX", 32768)
FEASUP_TIE_TEMP = _env_float("PERTURB_FEASUP_TIE_TEMP", 1.0)   # near-tie softmax temperature
FEASUP_TIE_MULT = _env_int("PERTURB_FEASUP_TIE_MULT", 4)       # sample top (mult·K) when benefits tie
FEASUP_TIE_VARIANTS = _env_int("PERTURB_FEASUP_TIE_VARIANTS", 2)  # randomized tie variants per proposal
FEASUP_ENS_EVERY = _env_int("PERTURB_FEASUP_ENS_EVERY", 3)    # ensemble candidate every N iters
FEASUP_SPATIAL = _env_bool("PERTURB_FEASUP_SPATIAL", True)    # spatially structured proposals
FEASUP_PATCH = _env_int("PERTURB_FEASUP_PATCH", 8)           # square patch side for spatial proposals
FEASUP_STAGNATION = _env_int("PERTURB_FEASUP_STAGNATION", 20)  # evals w/o improvement before rotating
FEASUP_RESCUE_FRACS = _env_floats("PERTURB_FEASUP_RESCUE_FRACS", (0.05, 0.20))  # rescue mutation fractions
FEASUP_POOL_REFRESH = _env_int("PERTURB_FEASUP_POOL_REFRESH", 8)  # rebuild target pool every N iters
FEASUP_SPARSE_STARTS = _env_floats("PERTURB_FEASUP_SPARSE_STARTS", (0.01, 0.05, 0.20))
FEASUP_DENSE_STARTS = _env_int("PERTURB_FEASUP_DENSE_STARTS", 2)
