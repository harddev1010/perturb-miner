"""Tunable constants for the perturb attack engine.

Every knob is an env var (PERTURB_*) so behavior is switchable at runtime without a redeploy.
Grouped by concern: core / accept-gate / per-algorithm. Defaults are the shipping values.
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
RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 3.5)   # deadline headroom
FIND_FLIP_BUDGET = _env_float("PERTURB_FIND_FLIP_BUDGET", 6.0)            # wall-clock for the search
# ±1/255 edits on a grid-aligned clean image survive PNG exactly, so the round-trip is identity.
SKIP_ROUNDTRIP = _env_bool("PERTURB_SKIP_ROUNDTRIP", True)

# --- Accept gate (transfer safety) ------------------------------------------------------
# kappa: require margin <= -kappa. With the TF32 envelope on, the dominant drift axis is
# covered by construction, so kappa drops to the residual cushion.
MARGIN_BUFFER = _env_float("PERTURB_MINER_MARGIN_BUFFER", 0.01)
KAPPA_RESID = _env_float("PERTURB_KAPPA_RESID", 0.002)
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

# --- Algorithm 1: batched multi-loss quantized FGSM -------------------------------------
QFGSM_TOPM = _env_int("PERTURB_QFGSM_TOPM", 5)
QFGSM_TAU = _env_float("PERTURB_QFGSM_TAU", 1.0)

# --- Algorithm 2: quantized one-byte PGD ------------------------------------------------
PGD_T = _env_int("PERTURB_PGD_T", 4)                  # ascent steps
PGD_R = _env_int("PERTURB_PGD_R", 3)                  # random restarts
PGD_ALPHA = _env_float("PERTURB_PGD_ALPHA", 0.5)      # latent step size (byte units)
PGD_ZERO_THRESH = _env_float("PERTURB_PGD_ZERO_THRESH", 0.1)  # shrink weak latent channels to 0
PGD_START_PROB = _env_float("PERTURB_PGD_START_PROB", 0.05)   # sparse random-start density

# --- Algorithm 3: quantized DeepFool/FAB top-M boundary ---------------------------------
BOUNDARY_M = _env_int("PERTURB_BOUNDARY_M", 5)
BOUNDARY_ROUNDS = _env_int("PERTURB_BOUNDARY_ROUNDS", 2)

# --- Algorithm 4: quantized SparseFool/JSMA saliency ------------------------------------
SALIENCY_CHUNKS = _env_ints(
    "PERTURB_SALIENCY_CHUNKS", (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
)
SALIENCY_PASS2_KS = _env_ints("PERTURB_SALIENCY_PASS2_KS", (1024, 4096, 16384, 65536))
SALIENCY_TAU = _env_float("PERTURB_SALIENCY_TAU", 1.0)

# --- Algorithm 5: gradient-seeded Square / block search ---------------------------------
SQUARE_SIZES = _env_ints("PERTURB_SQUARE_SIZES", (64, 32, 16, 8, 4, 2, 1))
SQUARE_BATCH = _env_int("PERTURB_SQUARE_BATCH", 32)
SQUARE_GRAD_PROB = _env_float("PERTURB_SQUARE_GRAD_PROB", 0.7)

# --- find_beam_byte_pgd: Multi-target Beam Byte-PGD (beam search, standalone orchestrator) -----
# Flip-FIRST for high-margin images (m0 ~ 3-7): build a big bank of dense/semi-dense q=1 sign patterns
# from several margin losses (hard top-1, soft top-M, targeted top-2/3), keep a BEAM of the lowest REAL
# margin candidates, re-linearize a soft top-M gradient at each beam anchor, and repeat to the budget.
# Each round mixes "replace" (update an anchor's bytes) and "fresh" (new pattern) variants. Returns
# safe on margin<=-kappa, else the best margin<0 when time runs out. No minimal-prefix search up front.
BEAM_ROUNDS = _env_int("PERTURB_BEAM_ROUNDS", 0)               # beam re-linearization rounds (0 = until budget)
BEAM_SIZE = _env_int("PERTURB_BEAM_SIZE", 4)
BEAM_TOPM = _env_int("PERTURB_BEAM_TOPM", 5)                   # initial soft top-M wrong classes
BEAM_TAU = _env_float("PERTURB_BEAM_TAU", 1.0)                # initial soft tau
BEAM_TARGET_CLASSES = _env_int("PERTURB_BEAM_TARGET_CLASSES", 3)  # stage-1 targeted pair losses (top-1..top-(T-1))
BEAM_ROUND_TOPM = _env_int("PERTURB_BEAM_ROUND_TOPM", 3)      # per-anchor soft top-M in the beam rounds
BEAM_ROUND_TAU = _env_float("PERTURB_BEAM_ROUND_TAU", 0.75)
BEAM_INIT_RATIOS = _env_floats(
    "PERTURB_BEAM_INIT_RATIOS", (0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 0.60, 1.00)
)
BEAM_ROUND_RATIOS = _env_floats("PERTURB_BEAM_ROUND_RATIOS", (0.03, 0.10, 0.30, 0.60, 1.00))
BEAM_REPAIR_PROBS = _env_floats("PERTURB_BEAM_REPAIR_PROBS", (0.01, 0.03, 0.10))  # random repairs on a stalled beam

# --- find_population_pgd: Batched Multi-Target Quantized PGD (population search, standalone) ----
# Flip-first for HIGH-margin images: spend the budget testing a population of q=1 candidates
# (zero + sparse random starts, then dense/semi-dense/PGD/mutation children) under a soft top-M
# gradient re-linearized at the lowest real-margin candidate. Return safe on margin<=-kappa, else
# the best margin<0 when time runs out.
POP_MAX_ROUNDS = _env_int("PERTURB_POP_MAX_ROUNDS", 0)        # 0 = loop until the time budget
POP_TOPM = _env_int("PERTURB_POP_TOPM", 5)
POP_TAU = _env_float("PERTURB_POP_TAU", 1.0)
POP_TAU_NEAR = _env_float("PERTURB_POP_TAU_NEAR", 0.5)        # sharper tau once near the boundary
POP_NEAR_THRESH = _env_float("PERTURB_POP_NEAR_THRESH", 0.2)  # |best margin| below this => near
POP_INIT_DENSITIES = _env_floats("PERTURB_POP_INIT_DENSITIES", (0.01, 0.05, 0.20))
POP_PREFIX_RATIOS = _env_floats(
    "PERTURB_POP_PREFIX_RATIOS", (0.001, 0.005, 0.01, 0.05, 0.20, 0.50, 1.00)
)
POP_PGD_ALPHAS = _env_floats("PERTURB_POP_PGD_ALPHAS", (0.5, 1.0))
POP_DROP_PROBS = _env_floats("PERTURB_POP_DROP_PROBS", (0.05, 0.10, 0.20))

# --- find_ensemble_byte_pgd: beam Byte-PGD + ensemble directions + structured mutation --------
# The merged/strongest version: the beam backbone above, plus (1) a DLR-normalized gradient on the
# clean image + best anchor only, (2) sign-mix candidates (normalized soft + targeted gradients),
# (3) limited opposite-sign probes on the strongest channels, (4) structured byte mutation near the
# best candidate once it is close to the boundary, and (5) margin-drop efficiency pruning that drops
# a gradient direction when the real margin barely follows the linear prediction. Gradient budget is
# capped: round 0 uses 3 grads at clean, later rounds use 2 grads at the best anchor only.
ENS_ROUNDS = _env_int("PERTURB_ENS_ROUNDS", 0)                # 0 = loop until the time budget
ENS_BEAM_SIZE = _env_int("PERTURB_ENS_BEAM_SIZE", 3)
ENS_TOPM = _env_int("PERTURB_ENS_TOPM", 5)
ENS_TAU = _env_float("PERTURB_ENS_TAU", 1.0)
ENS_TAU_NEAR = _env_float("PERTURB_ENS_TAU_NEAR", 0.75)       # sharper tau in the beam rounds
ENS_INIT_RATIOS = _env_floats(
    "PERTURB_ENS_INIT_RATIOS", (0.001, 0.005, 0.01, 0.05, 0.20, 0.50, 1.00)
)
ENS_ROUND_RATIOS = _env_floats("PERTURB_ENS_ROUND_RATIOS", (0.03, 0.10, 0.30, 0.60, 1.00))
ENS_OPP_KS = _env_ints("PERTURB_ENS_OPP_KS", (1024, 4096, 16384))   # opposite-sign probe sizes (top-k channels)
ENS_MUTATE_THRESH = _env_float("PERTURB_ENS_MUTATE_THRESH", 0.5)    # run structured mutation when |best margin| < this
ENS_MUTATE_PROBS = _env_floats("PERTURB_ENS_MUTATE_PROBS", (0.05, 0.10, 0.20))  # drop/flip fraction of active channels
ENS_MUTATE_ADD = _env_ints("PERTURB_ENS_MUTATE_ADD", (256, 1024, 4096))         # add next-ranked inactive channels
ENS_EFF_FRAC = _env_float("PERTURB_ENS_EFF_FRAC", 0.1)         # abandon a direction if real drop < frac * predicted

# --- find_hydra: Q1-Hydra finder (primitive-space search, standalone orchestrator) ------
# Search DIFFERENT STRUCTURES of ±1 patterns, not just amounts of the same channel-prefix structure.
# Candidate families from a couple of gradients (global prefix / spatial tiles / low-frequency / color)
# fill a batched bank; then a primitive-space beam search mutates tiles/low-freq/color/crossover under
# real-margin feedback. Returns safe on margin<=-kappa, else best margin<0 at timeout.
HYDRA_TOPM = _env_int("PERTURB_HYDRA_TOPM", 5)
HYDRA_TAU = _env_float("PERTURB_HYDRA_TAU", 1.0)
HYDRA_PREFIX_RATIOS = _env_floats(
    "PERTURB_HYDRA_PREFIX_RATIOS", (0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 0.60, 1.00)
)
HYDRA_TILE_SIZES = _env_ints("PERTURB_HYDRA_TILE_SIZES", (8, 16, 32, 64))
HYDRA_TILE_COUNTS = _env_ints("PERTURB_HYDRA_TILE_COUNTS", (1, 2, 4, 8, 16, 32, 64))
HYDRA_LF_PERIODS = _env_ints("PERTURB_HYDRA_LF_PERIODS", (4, 8, 16))
HYDRA_BEAM_SIZE = _env_int("PERTURB_HYDRA_BEAM_SIZE", 8)
HYDRA_MUT_PER = _env_int("PERTURB_HYDRA_MUT_PER", 8)           # mutations generated per beam member per round
HYDRA_BEAM_ROUNDS = _env_int("PERTURB_HYDRA_BEAM_ROUNDS", 0)   # 0 = until budget

# --- find_apgd_dlr: Quantized APGD-DLR with batched restarts (standalone orchestrator) --
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

# --- find_apgd_targeted: targeted APGD-L∞ (DLR-T) + momentum + restarts (the "live finder") ----
# AutoAttack-style for confident/high-margin images: target the top runner-up classes with the
# targeted-DLR loss (concentrates the diffuse gradient; DLR doesn't saturate at high confidence),
# momentum + adaptive step + restart-from-best, all batched. Runs to the budget; never bails early.
APGDT_TOPM = _env_int("PERTURB_APGDT_TOPM", 5)
APGDT_TARGETS = _env_int("PERTURB_APGDT_TARGETS", 3)          # # runner-up classes to attack in parallel
APGDT_ALPHA0 = _env_float("PERTURB_APGDT_ALPHA0", 1.0)       # initial latent step
APGDT_MIN_ALPHA = _env_float("PERTURB_APGDT_MIN_ALPHA", 0.1)
APGDT_MOMENTUM = _env_float("PERTURB_APGDT_MOMENTUM", 0.75)
APGDT_PATIENCE = _env_int("PERTURB_APGDT_PATIENCE", 3)        # stalled iters before halve+restart-from-best
APGDT_UNTARGETED = _env_bool("PERTURB_APGDT_UNTARGETED", True)  # also run one untargeted DLR trajectory

# --- find_fmn: Q1-FMN-L1 support finder (continuous L1 "breathing" -> byte snap) ---------
# Fast Minimum-Norm flavor: take a few continuous margin-reducing steps under an ADAPTIVE L1 radius
# (grow when not adversarial, shrink when adversarial), L1-project to stay sparse, then snap the support
# to ±1 byte prefixes and verify. Outer loop restarts to the budget. Continuous delta is never submitted.
FMN_STEPS = _env_int("PERTURB_FMN_STEPS", 8)                 # inner continuous steps per pass
FMN_ALPHA = _env_float("PERTURB_FMN_ALPHA", 0.3)
FMN_EPS0_FRAC = _env_float("PERTURB_FMN_EPS0_FRAC", 0.05)    # initial L1 radius as a fraction of n*q
FMN_GROW = _env_float("PERTURB_FMN_GROW", 1.25)
FMN_SHRINK = _env_float("PERTURB_FMN_SHRINK", 0.85)
FMN_RATIOS = _env_floats("PERTURB_FMN_RATIOS", (0.05, 0.10, 0.20, 0.40, 0.70, 1.00))

# --- find_frank_wolfe: Q1-Block Frank-Wolfe + Sparse-RS rescue --------------------------
# Block top-K FW (steepest-coordinate oracle, K from a schedule) builds the support gradually; each
# round also emits Sparse-RS support mutations (add-ranked / remove / swap / flip). Re-anchors to the
# best byte delta and loops to the budget. All candidates are exact ±k_min byte edits.
FW_K_SCHEDULE = _env_ints("PERTURB_FW_K_SCHEDULE", (2048, 4096, 8192, 16384, 32768, 65536))
FW_DROP_RATIO = _env_float("PERTURB_FW_DROP_RATIO", 0.1)     # weakest-active fraction to drop/replace
FW_RS_MUT = _env_int("PERTURB_FW_RS_MUT", 8)                 # Sparse-RS mutations per round
FW_RS_ADD = _env_int("PERTURB_FW_RS_ADD", 2048)
FW_RS_SWAP = _env_int("PERTURB_FW_RS_SWAP", 1024)

# --- find_alma: Q1-ALMA-lite (augmented Lagrangian -> byte snap) ------------------------
# Augmented-Lagrangian flavor: minimize an L1-ish distance while a multiplier/penalty on the
# not-adversarial constraint grows until the margin is crossed (smooth, less "jumpy" than hard
# thresholding — good for high-entropy multi-competitor cases). Snap the support to byte prefixes.
ALMA_STEPS = _env_int("PERTURB_ALMA_STEPS", 8)
ALMA_ALPHA = _env_float("PERTURB_ALMA_ALPHA", 0.3)
ALMA_LAM0 = _env_float("PERTURB_ALMA_LAM0", 0.1)
ALMA_RHO0 = _env_float("PERTURB_ALMA_RHO0", 1.0)
ALMA_RHO_GROW = _env_float("PERTURB_ALMA_RHO_GROW", 1.5)
ALMA_L1W = _env_float("PERTURB_ALMA_L1W", 0.01)             # weight of the L1 shrink term
ALMA_RATIOS = _env_floats("PERTURB_ALMA_RATIOS", (0.05, 0.10, 0.20, 0.40, 0.70, 1.00))

# ========================================================================================
# optim_rmse_*: POST-FLIP RMSE refinement (minimize |S|, the changed-channel count).
# ----------------------------------------------------------------------------------------
# These do NOT find a flip — they warm-start from the Bank's best flip (safe preferred) and
# shrink it. With every channel pinned at exactly +/-k_min bytes, RMSE = q*sqrt(|S|/n), so
# minimizing RMSE is exactly minimizing |S|. Run one (or several, in order) AFTER a finder in
# the orchestrator switch; each folds its sparser survivors back into the Bank, so the global
# best automatically tracks the sparsest. Acceptance is always the validator-faithful batched
# eval (envelope + SSIM/PSNR + kappa); gradients are used only to RANK and PREDICT.

# Stage A -- gradient-ranked byte rollback (greedy backward elimination).
OPTIM_PRUNE_MAX_ROUNDS = _env_int("PERTURB_OPTIM_PRUNE_MAX_ROUNDS", 0)  # 0 = loop until budget
OPTIM_PRUNE_LADDER = _env_int("PERTURB_OPTIM_PRUNE_LADDER", 2)          # geometric base for removal-count ladder

# Stage B -- one-for-many coordinate exchange (add 1 strong channel, drop >=2 weak ones).
OPTIM_EXCHANGE_MAX_ROUNDS = _env_int("PERTURB_OPTIM_EXCHANGE_MAX_ROUNDS", 0)  # 0 = until budget
OPTIM_EXCHANGE_ADDS = _env_int("PERTURB_OPTIM_EXCHANGE_ADDS", 32)       # top-N unused channels probed as the addition
OPTIM_EXCHANGE_MIN_DROP = _env_int("PERTURB_OPTIM_EXCHANGE_MIN_DROP", 2)  # need >=this removals per addition to net-shrink

# Warm-start FMN-L2 -> snap -> prune (radius "breathing" around the boundary).
OPTIM_FMN_L2_STEPS = _env_int("PERTURB_OPTIM_FMN_L2_STEPS", 12)
OPTIM_FMN_L2_ALPHA = _env_float("PERTURB_OPTIM_FMN_L2_ALPHA", 0.5)      # normalized delta-step size (in units of q)
OPTIM_FMN_L2_GAMMA = _env_float("PERTURB_OPTIM_FMN_L2_GAMMA", 0.10)     # eps shrink/grow rate
OPTIM_FMN_L2_RATIOS = _env_floats("PERTURB_OPTIM_FMN_L2_RATIOS", (0.10, 0.25, 0.50, 0.75, 1.00))

# Warm-start FAB-L2 -> snap -> prune (boundary linearization, biased toward the clean image).
OPTIM_FAB_L2_STEPS = _env_int("PERTURB_OPTIM_FAB_L2_STEPS", 12)
OPTIM_FAB_L2_ALPHA_MAX = _env_float("PERTURB_OPTIM_FAB_L2_ALPHA_MAX", 0.5)  # max bias toward the clean projection
OPTIM_FAB_L2_ETA = _env_float("PERTURB_OPTIM_FAB_L2_ETA", 0.10)        # overshoot back into the adversarial region
OPTIM_FAB_L2_RATIOS = _env_floats("PERTURB_OPTIM_FAB_L2_RATIOS", (0.10, 0.25, 0.50, 0.75, 1.00))

# sigma-zero: differentiable-L0 surrogate descent (support finder).
OPTIM_SIGMA_STEPS = _env_int("PERTURB_OPTIM_SIGMA_STEPS", 20)
OPTIM_SIGMA_ETA0 = _env_float("PERTURB_OPTIM_SIGMA_ETA0", 1.0)         # initial step (cosine-annealed), units of q
OPTIM_SIGMA_SIGMA = _env_float("PERTURB_OPTIM_SIGMA_SIGMA", 1e-3)      # surrogate smoothing
OPTIM_SIGMA_TAU0 = _env_float("PERTURB_OPTIM_SIGMA_TAU0", 0.3)         # initial relative zeroing threshold
OPTIM_SIGMA_T = _env_float("PERTURB_OPTIM_SIGMA_T", 0.01)             # tau adjustment rate
OPTIM_SIGMA_RATIOS = _env_floats("PERTURB_OPTIM_SIGMA_RATIOS", (0.25, 0.50, 0.75, 1.00))

# FMN-L0: integer support budget eps, shrunk toward the boundary (refiner).
OPTIM_FMN_L0_STEPS = _env_int("PERTURB_OPTIM_FMN_L0_STEPS", 20)
OPTIM_FMN_L0_ALPHA = _env_float("PERTURB_OPTIM_FMN_L0_ALPHA", 0.5)     # normalized delta-step size (units of q)
OPTIM_FMN_L0_GAMMA = _env_float("PERTURB_OPTIM_FMN_L0_GAMMA", 0.10)    # eps (support-count) shrink/grow rate
OPTIM_FMN_L0_RATIOS = _env_floats("PERTURB_OPTIM_FMN_L0_RATIOS", (0.50, 0.75, 1.00))
