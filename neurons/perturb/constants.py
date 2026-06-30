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
KAPPA_RESID = _env_float("PERTURB_KAPPA_RESID", 0.004)   # cold-start kappa for the envelope regime
# Dynamic kappa (calibration.py): under the TF32 envelope, learn kappa online from the observed residual
# between the fast proxy margin and the exact validator-faithful margin (PNG round-trip + worst TF32),
# replacing the fixed KAPPA_RESID with a high-quantile upper bound on real drift. CPU / envelope-off keep
# the static cushion. KAPPA_RESID stays the cold-start fallback until enough residuals are observed.
DYNAMIC_KAPPA = _env_bool("PERTURB_DYNAMIC_KAPPA", True)
KAPPA_FLOOR = _env_float("PERTURB_KAPPA_FLOOR", 0.0005)   # smallest kappa once residuals look stable
KAPPA_CEILING = _env_float("PERTURB_KAPPA_CEILING", 0.02)  # cap against pathological residual spikes
KAPPA_QUANTILE = _env_float("PERTURB_KAPPA_QUANTILE", 0.99)  # residual quantile (20+ samples)
KAPPA_CUSHION = _env_float("PERTURB_KAPPA_CUSHION", 0.0002)  # tiny numerical cushion added to the quantile
KAPPA_SAMPLES = _env_int("PERTURB_KAPPA_SAMPLES", 256)   # rolling residual-history length
KAPPA_SPREAD_COEF = _env_float("PERTURB_KAPPA_SPREAD_COEF", 0.5)  # per-candidate TF32-spread weight
KAPPA_STORE = os.getenv("PERTURB_KAPPA_STORE",
                        os.path.join(os.path.expanduser("~"), ".cache", "perturb", "kappa_calib.json"))
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

# --- Cardinality-continuation ternary search (integrated find + sparsify) ----------------
# Feasibility and L0 minimization are solved together: gradients propose actions, an exact fixed-K
# ternary projection keeps every candidate legal, real candidates are batch-verified, and the support
# budget K is grown to find the first flip then shrunk (geometric + bisection) to minimize |S|.
LOSSES = _env_strs("PERTURB_LOSSES", ("dlr", "soft", "hard", "ce"))  # attack-loss portfolio (rotated)
TAU = _env_float("PERTURB_TAU", 1.0)                  # soft-margin (logsumexp) temperature
TOPM = _env_int("PERTURB_TOPM", 6)                    # target pool size (top wrong classes)
POOL_SEEDS = _env_int("PERTURB_POOL_SEEDS", 4)        # smallest-K̂ targets seeded in Phase A
# Support ladder: prefix sizes as multiples of the linearized crossing size K̂ (a medium support often
# beats the fully dense candidate when dense edits interfere destructively).
K_MULTS = _env_floats("PERTURB_K_MULTS", (0.5, 0.75, 1.0, 1.25, 1.5, 2.0))
ORDER_VARIANTS = _env_int("PERTURB_ORDER_VARIANTS", 1)  # near-tie randomized orders per ladder size
TIE_MULT = _env_int("PERTURB_TIE_MULT", 4)            # sample top (mult·K) when benefits tie
TIE_TEMP = _env_float("PERTURB_TIE_TEMP", 1.0)        # near-tie softmax temperature
# Fixed-K projected ternary search (APGD on a latent u, projected onto {-1,0,+1}^N with |S|_0<=K).
FIXED_ITERS = _env_int("PERTURB_FIXED_ITERS", 6)      # re-linearized iterations per fixed-K call
APGD_ALPHA0 = _env_float("PERTURB_APGD_ALPHA0", 2.0)  # initial latent step (activates top saliency)
APGD_MIN_ALPHA = _env_float("PERTURB_APGD_MIN_ALPHA", 0.25)
APGD_MOMENTUM = _env_float("PERTURB_APGD_MOMENTUM", 0.75)
APGD_PATIENCE = _env_int("PERTURB_APGD_PATIENCE", 2)  # stalled iters before halving alpha
U_CLAMP = _env_float("PERTURB_U_CLAMP", 4.0)          # latent magnitude clamp
# Cardinality continuation: shrink factor, near-flip parents kept for seeding, basin-restart cadence.
CONT_ALPHA = _env_float("PERTURB_CONT_ALPHA", 0.75)   # geometric K shrink on success
FEAS_GROW = _env_float("PERTURB_FEAS_GROW", 2.0)      # geometric K growth while no flip exists yet
PARENTS = _env_int("PERTURB_PARENTS", 4)              # diverse near-flip seeds carried forward
RESTART_EVERY = _env_int("PERTURB_RESTART_EVERY", 12)  # Phase-A restart cadence (new basin)
CONVERGE_PATIENCE = _env_int("PERTURB_CONVERGE_PATIENCE", 3)  # stalled basins before stopping

# --- Dynamic sparse fixed-q framework (Phases A-E) ----------------------------------------
# This is the "DynamicSparseFixedQAttack" engine: Phase A seeds a fixed-K support (clean +
# targeted + random-start + FEATURE-GUIDED candidates), Phases B-D dynamically optimize the
# mask/sign and block-swap the support, Phase E reduces cardinality. Per the current request the
# optimizer (B-E) is IMPLEMENTED but NOT RUN: the engine returns as soon as Phase A finds a flip.
RUN_OPTIM = _env_bool("PERTURB_RUN_OPTIM", False)        # gate Phases B-E (off => return on first flip)
IGNORE_TIMEOUT = _env_bool("PERTURB_IGNORE_TIMEOUT", True)  # ignore the deadline / budget guards
# Master early-return: the instant a RETURNABLE flip is banked (safe flip, or any flip when
# ALLOW_UNSAFE_FLIP), stop everything and return it — no further optimization/sparsification, even if
# RUN_OPTIM is on. Default on. Set to 0 to let the optimizer run (then OPTIM_SECONDS caps the post-flip work).
RETURN_FIRST_FLIP = _env_bool("PERTURB_RETURN_FIRST_FLIP", True)
# Phase-A support sizing + seeding breadth.
K_INIT_FRAC = _env_float("PERTURB_K_INIT_FRAC", 0.1)    # initial support as a fraction of N channels
K_MIN_FRAC = _env_float("PERTURB_K_MIN_FRAC", 0.001)     # cardinality-reduction floor (fraction of N)
TARGET_COUNT = _env_int("PERTURB_FW_TARGET_COUNT", 4)    # target-specific clean gradients in Phase A
RANDOM_START_COUNT = _env_int("PERTURB_FW_RANDOM_STARTS", 4)  # random-start gradient reservoirs
RANDOM_START_FRAC = _env_float("PERTURB_FW_RANDOM_FRAC", 0.05)  # density of each random-start mask
BIG_INIT = _env_float("PERTURB_FW_BIG_INIT", 8.0)        # mask-logit boost for the seeded support
MASK_NOISE = _env_float("PERTURB_FW_MASK_NOISE", 0.01)   # small random noise on mask logits

# --- Feature guidance (Q1: feature-guided candidate selection) ----------------------------
# A spatial relevance map from a hidden conv layer (sum_c |feat * d margin/d feat|), upsampled to
# the input grid, then used to GATE the input-gradient candidate scores. Default mode is the
# safer "spatial gate" (feature map picks regions; the input gradient picks the RGB channel+sign).
FEATURE_GUIDED = _env_bool("PERTURB_FEATURE_GUIDED", True)
FEATURE_GATE = _env_bool("PERTURB_FEATURE_GATE", True)   # True: spatial gate; False: pure multiplicative score
FEATURE_EPS = _env_float("PERTURB_FEATURE_EPS", 0.05)    # epsilon floor in (eps + relevance)^beta
FEATURE_BETA = _env_float("PERTURB_FEATURE_BETA", 1.0)   # relevance exponent
FEATURE_QUOTA_FRAC = _env_float("PERTURB_FEATURE_QUOTA_FRAC", 0.25)  # feature candidates as a fraction of K
FEATURE_PIXEL_QUOTA_FRAC = _env_float("PERTURB_FEATURE_PIXEL_QUOTA_FRAC", 0.5)  # important-pixel gate size (frac of H*W)
FEATURE_REFRESH_INTERVAL = _env_int("PERTURB_FEATURE_REFRESH_INTERVAL", 8)  # refresh during Phase B/C

# --- Phase B (dynamic mask + sign optimization) -------------------------------------------
ETA_MASK = _env_float("PERTURB_FW_ETA_MASK", 0.5)        # mask-logit learning rate
SIGN_MOMENTUM = _env_float("PERTURB_FW_SIGN_MOMENTUM", 0.9)
HISTORY_BETA = _env_float("PERTURB_FW_HISTORY_BETA", 0.9)  # path-EMA decay
TEMPERATURE = _env_float("PERTURB_FW_TEMPERATURE", 1.0)  # straight-through sigmoid temperature
MAX_ITERATIONS = _env_int("PERTURB_FW_MAX_ITERATIONS", 200)  # Phase B-D iterations per fixed K

# --- Phase C (exact block swap) -----------------------------------------------------------
SWAP_INTERVAL = _env_int("PERTURB_FW_SWAP_INTERVAL", 5)  # block-swap cadence (iterations)
PROPOSAL_COUNT = _env_int("PERTURB_FW_PROPOSAL_COUNT", 16)  # exact swap proposals per round
# Block size = how many coords are swapped in/out per round. By default it stays a FLAT fraction of
# K (BLOCK_FRAC), floored at BLOCK_MIN, so the step does NOT shrink while still searching for a flip.
# Set BLOCK_ANNEAL=1 to recover the old iteration-thirds taper (0.05 -> 0.02 -> 0.005 of K), which is
# only useful once a flip already exists and you are sparsifying.
BLOCK_FRAC = _env_float("PERTURB_FW_BLOCK_FRAC", 0.05)   # block as a fraction of K (flat by default)
BLOCK_MIN = _env_int("PERTURB_FW_BLOCK_MIN", 64)         # floor on block size (keeps steps from going tiny)
BLOCK_ANNEAL = _env_bool("PERTURB_FW_BLOCK_ANNEAL", False)  # taper the block over iterations (sparsify mode)

# --- Phase D (partial restart) ------------------------------------------------------------
RESTART_PATIENCE = _env_int("PERTURB_FW_RESTART_PATIENCE", 12)
RESTART_MIN_IMPROVE = _env_float("PERTURB_FW_RESTART_MIN_IMPROVE", 1e-3)
RESTART_TURNOVER_THRESH = _env_float("PERTURB_FW_RESTART_TURNOVER", 0.05)
RESTART_FRACTION = _env_float("PERTURB_FW_RESTART_FRACTION", 0.1)
RESTART_PROPOSALS = _env_int("PERTURB_FW_RESTART_PROPOSALS", 16)

# --- Phase E (cardinality continuation) ---------------------------------------------------
REDUCTION_FRACTION = _env_float("PERTURB_FW_REDUCTION_FRACTION", 0.1)
ITERATIONS_PER_K = _env_int("PERTURB_FW_ITERATIONS_PER_K", 60)

# --- Optim time budget (only when RUN_OPTIM) ----------------------------------------------
# Two-stage clock: the FIND stage (until the first flip is banked) runs with no wall-clock limit
# (it closes on max-iters / convergence). The instant the first flip appears, a budget of
# OPTIM_SECONDS is armed; all optimization after the flip (further refinement + Phase E
# sparsification) must finish within that window.
OPTIM_SECONDS = _env_float("PERTURB_FW_OPTIM_SECONDS", 12.0)
