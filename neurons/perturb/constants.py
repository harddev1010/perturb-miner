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
RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 2)   # deadline headroom (base)
# Reserve scales with the measured per-forward cost so larger images / models leave enough post-search
# headroom for serialization + verification (added on top of RESERVE_SECONDS). t_step is one fwd+bwd.
RESERVE_FWD_MULT = _env_float("PERTURB_RESERVE_FWD_MULT", 4.0)
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
RUN_OPTIM = _env_bool("PERTURB_RUN_OPTIM", True)        # gate Phases B-E (off => return on first flip)
IGNORE_TIMEOUT = _env_bool("PERTURB_IGNORE_TIMEOUT", False)  # ignore the deadline / budget guards
# Master early-return: the instant a RETURNABLE flip is banked (safe flip, or any flip when
# ALLOW_UNSAFE_FLIP), stop everything and return it — no further optimization/sparsification, even if
# RUN_OPTIM is on. Default on. Set to 0 to let the optimizer run (then OPTIM_SECONDS caps the post-flip work).
RETURN_FIRST_FLIP = _env_bool("PERTURB_RETURN_FIRST_FLIP", False)
# Phase-A support sizing + seeding breadth.
K_INIT_FRAC = _env_float("PERTURB_K_INIT_FRAC", 0.7)    # initial support as a fraction of N channels
K_MIN_FRAC = _env_float("PERTURB_K_MIN_FRAC", 0.001)     # cardinality-reduction floor (fraction of N)
TARGET_COUNT = _env_int("PERTURB_FW_TARGET_COUNT", 3)    # target-specific clean gradients in Phase A
RANDOM_START_COUNT = _env_int("PERTURB_FW_RANDOM_STARTS", 2)  # random-start gradient reservoirs
RANDOM_START_FRAC = _env_float("PERTURB_FW_RANDOM_FRAC", 0.05)  # density of each random-start mask
BIG_INIT = _env_float("PERTURB_FW_BIG_INIT", 2.0)        # mask-logit boost for the seeded support
MASK_NOISE = _env_float("PERTURB_FW_MASK_NOISE", 0.002)   # small random noise on mask logits
# Batched gradients: compute clean + targeted + random-start gradients in ONE forward+backward over a
# replicated-input batch instead of ~9 sequential backwards (T1.2). Big Phase-A speedup on GPU.
BATCHED_GRADS = _env_bool("PERTURB_FW_BATCHED_GRADS", True)
# Normalized-rank beam fusion (T1.3): fuse the per-source rankings by percentile rank instead of
# reranking the union by the clean-gradient score (which deleted target/feature-surfaced coords).
RANK_FUSION = _env_bool("PERTURB_FW_RANK_FUSION", True)
# Grow-until-first-flip ladder (T1.1): build nested geometric supports from the FUSED ranking and grade
# them in the same batched pass; the Bank keeps the sparsest flipping rung, so you land sparse directly
# instead of shrinking from K_init. Falls through to the fixed-K optimizer if no rung flips.
GROW_LADDER = _env_bool("PERTURB_FW_GROW_LADDER", True)
GROW_RUNGS = _env_floats("PERTURB_FW_GROW_RUNGS", (0.01, 0.05, 0.1, 0.2, 0.4, 0.7))  # fractions of N
GROW_VARIANTS = _env_int("PERTURB_FW_GROW_VARIANTS", 1)  # near-tie randomized order variants per rung

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
ETA_MASK = _env_float("PERTURB_FW_ETA_MASK", 4.0)        # mask-logit learning rate
SIGN_MOMENTUM = _env_float("PERTURB_FW_SIGN_MOMENTUM", 0.9)
HISTORY_BETA = _env_float("PERTURB_FW_HISTORY_BETA", 0.9)  # path-EMA decay
TEMPERATURE = _env_float("PERTURB_FW_TEMPERATURE", 1.0)  # straight-through sigmoid temperature
MAX_ITERATIONS = _env_int("PERTURB_FW_MAX_ITERATIONS", 200)  # Phase B-D iterations per fixed K

# --- Phase C (exact block swap) -----------------------------------------------------------
SWAP_INTERVAL = _env_int("PERTURB_FW_SWAP_INTERVAL", 4)  # block-swap cadence (iterations)
PROPOSAL_COUNT = _env_int("PERTURB_FW_PROPOSAL_COUNT", 16)  # exact swap proposals per round
# Block size = how many coords are swapped in/out per round. By default it stays a FLAT fraction of
# K (BLOCK_FRAC), floored at BLOCK_MIN, so the step does NOT shrink while still searching for a flip.
# Set BLOCK_ANNEAL=1 to recover the old iteration-thirds taper (0.05 -> 0.02 -> 0.005 of K), which is
# only useful once a flip already exists and you are sparsifying.
BLOCK_FRAC = _env_float("PERTURB_FW_BLOCK_FRAC", 0.15)   # block as a fraction of K (flat by default)
BLOCK_MIN = _env_int("PERTURB_FW_BLOCK_MIN", 256)         # floor on block size (keeps steps from going tiny)
BLOCK_ANNEAL = _env_bool("PERTURB_FW_BLOCK_ANNEAL", False)  # taper the block over iterations (sparsify mode)

# --- Phase D (partial restart) ------------------------------------------------------------
RESTART_PATIENCE = _env_int("PERTURB_FW_RESTART_PATIENCE", 8)
RESTART_MIN_IMPROVE = _env_float("PERTURB_FW_RESTART_MIN_IMPROVE", 1e-3)
RESTART_TURNOVER_THRESH = _env_float("PERTURB_FW_RESTART_TURNOVER", 0.05)
RESTART_FRACTION = _env_float("PERTURB_FW_RESTART_FRACTION", 0.1)
RESTART_PROPOSALS = _env_int("PERTURB_FW_RESTART_PROPOSALS", 48)

# --- Phase E (cardinality continuation) ---------------------------------------------------
REDUCTION_FRACTION = _env_float("PERTURB_FW_REDUCTION_FRACTION", 0.1)
ITERATIONS_PER_K = _env_int("PERTURB_FW_ITERATIONS_PER_K", 60)

# --- Optim time budget (only when RUN_OPTIM) ----------------------------------------------
# Two-stage clock: the FIND stage (until the first flip is banked) runs with no wall-clock limit
# (it closes on max-iters / convergence). The instant the first flip appears, a budget of
# OPTIM_SECONDS is armed; all optimization after the flip (further refinement + Phase E
# sparsification) must finish within that window.
OPTIM_SECONDS = _env_float("PERTURB_FW_OPTIM_SECONDS", 40.0)

# --- Post-flip objective (maximize the full score, use the whole budget) -------------------------
# The validator scores total = perturbation(L∞,RMSE) + 0.03·clip(-margin/10,0,1) + 0.01·clip(px/8,0,1).
# perturbation(K) is ANALYTIC (RMSE=q·√(K/N)) and RISES as K falls; the margin bonus saturates at CW
# margin <= -CEIL. So among K that still saturate the margin, score = perturbation(K)+0.04 is maximized
# at the SMALLEST such K (=K_sat). env PERTURB_POSTFLIP_STRATEGY:
#   coupled (default): BINARY-SEARCH K_sat — ~log probes (warm-start + deepen at K), each score-ranked
#     into the Bank, which also holds Phase A's low-K bare flips. Lands at the peak within budget and
#     dynamically balances margin vs RMSE per image (hard images where deep margin costs too much RMSE
#     keep the bare flip). Much cheaper than a geometric descent that re-deepens every 10% step.
#   strict: geometric first-flip descent (shallow rungs, cheap) then ONE deepen pass at the settled K.
#   analytic: MODEL-based (fewest expensive probes). Fit margin(K) from a few probes, then MAXIMIZE the
#     analytic score S(K) (perturbation is closed-form q√(K/N); margin bonus from the interpolated margin;
#     novelty saturated) over a fine K grid for FREE, and verify the predicted optimum + neighborhood at
#     full budget. Fewer probes than the binary search -> more optimization depth on the winner.
#   grow: SPARSE-THEN-GROW. Deepen a dense anchor to saturation (so its retention ranks coords by their
#     DEEP-margin value, not a shallow first-flip value), reduce to a lean guided core, then GROW that
#     core back up by adding steepest margin-gain coords (re-linearizing each step), stopping at the
#     score peak (margin saturates => extra coords only cost RMSE). Reaches ~K_sat from BELOW, so it can
#     land a different (often better) coordinate SET than coupled's dense->shrink. No K_sat bisection.
#   both: run coupled, THEN a grow pass seeded from coupled's result (approached from below), folded into
#     the SAME score-ranked Bank. Strictly non-regressing vs coupled (can only raise the returned score),
#     at the cost of the extra grow budget. Recommended A/B target vs coupled.
#   ufs: UNIFIED FRONTIER SCHEDULER — an anytime replacement for the whole post-flip waterfall. One loop
#     over a K-indexed frontier + online margin surrogate; each step runs the highest-expected-gain move
#     (DEEPEN / SHRINK / GROW) against a single shared budget, so budget is never sliced (no starvation)
#     nor left unused (it schedules until the deadline, diversifying on convergence). See the UFS_* block.
POSTFLIP_STRATEGY = os.getenv("PERTURB_POSTFLIP_STRATEGY", "coupled").strip().lower() or "coupled"
ANALYTIC_PROBE_FRACS = _env_floats("PERTURB_ANALYTIC_PROBE_FRACS", (0.5, 0.25))  # K/K_anchor probes that fit margin(K)
MARGIN_DEEPEN_TARGET = _env_float("PERTURB_MARGIN_DEEPEN_TARGET", 10.5)  # CEIL: CW margin <= -this saturates the bonus
KSAT_REL_TOL = _env_float("PERTURB_KSAT_REL_TOL", 0.05)   # coupled: stop binary search when (hi-lo) <= max(KSAT_ABS_TOL, this·hi)
KSAT_ABS_TOL = _env_int("PERTURB_KSAT_ABS_TOL", 8)        # absolute tol floor so small K doesn't over-probe single coords
# Coupled V2 = a fast saturation-boundary LOCATOR followed by a SCORE-peak refinement (the validator
# rewards a score maximum, not the saturation threshold). Boundary probes run cheap (a fraction of the
# per-K iters) just to classify saturate/not; refinement probes around K_sat run the full budget. A
# near-miss (margin reached >= COUPLED_RETRY_FRAC·target) is retried once from a different warm-start
# parent before conceding the bracket (a single fixed-K run can be a false negative). COUPLED_PARENTS is
# a tiny beam of best-score states kept as alternate warm-start parents (breaks single-lineage path
# dependence). COUPLED_REFINE_MULTS are the K/K_sat ratios sampled in the refinement sweep.
# Phase 3 refinement is SCREEN-then-refine (keeps depth where it matters under the 15s budget): all
# COUPLED_REFINE_MULTS ratios of K_sat are screened with the cheap boundary budget, then only the best
# COUPLED_REFINE_FULL are given a full OptimizeFixedK. Ratios are <=1 (below the boundary, where the score
# peak lives — above-boundary K has worse RMSE and is already covered by the binary-search probes). Every
# expensive probe/retry is skipped when its analytic score upper bound can't beat the current Bank best.
COUPLED_BOUNDARY_ITER_FRAC = _env_float("PERTURB_COUPLED_BOUNDARY_ITER_FRAC", 0.4)
COUPLED_REFINE_MULTS = _env_floats("PERTURB_COUPLED_REFINE_MULTS", (0.85, 0.7, 0.55))
COUPLED_REFINE_FULL = _env_int("PERTURB_COUPLED_REFINE_FULL", 2)  # screened candidates given the full budget
COUPLED_RETRY_FRAC = _env_float("PERTURB_COUPLED_RETRY_FRAC", 0.6)
COUPLED_PARENTS = _env_int("PERTURB_COUPLED_PARENTS", 3)
# Budget-aware anchor (coupled Phase 1). The saturating anchor deepens the FIRST flip at its own K to
# CW margin <= CEIL. On a robust image where CEIL is slow/unreachable, a full-budget anchor can eat the
# whole post-flip window buying at most the 0.03 margin bonus while the K/RMSE search — which may raise
# score more — starves. BUT margin deepening in this optimizer is NON-monotonic and often DELAYED (block
# swaps / sign turnover pay off only after tens of iters), so a too-eager early stop bails at margin ~-4/-5
# on images that would have reached CEIL. So: (a) always give the anchor a real chance — run at least
# ANCHOR_MIN_ITERS before ANY slope-based early stop; (b) preserve the optimizer trajectory across chunks
# (continue from the LIVE state, not the best-margin snapshot — see _quick_anchor), so delayed nonlinear
# drops are not thrown away; (c) never bail once the margin is already close to CEIL (<= -ANCHOR_PUSH_MARGIN
# it is nearly saturated — finish it); (d) only after the warmup, stop on a sustained stall (margin gain
# < ANCHOR_MIN_MARGIN_GAIN for ANCHOR_STALL_CHUNKS chunks in a row => ~a 20-iter flat window) or the
# iter/time caps. Whatever depth it reached is a valid anchor; the score-ranked Bank + K sweep take over.
# Set the gate off to restore the old fixed full-budget anchor.
COUPLED_QUICK_ANCHOR = _env_bool("PERTURB_COUPLED_QUICK_ANCHOR", True)
ANCHOR_CHUNK_ITERS = _env_int("PERTURB_ANCHOR_CHUNK_ITERS", 5)         # iters per slope-check chunk (small: fine stall resolution)
ANCHOR_MIN_ITERS = _env_int("PERTURB_ANCHOR_MIN_ITERS", 25)          # never slope-bail before this many iters (give delayed deepening a chance)
ANCHOR_MAX_ITERS = _env_int("PERTURB_ANCHOR_MAX_ITERS", 40)          # hard cap on total anchor iters
ANCHOR_MAX_FRAC = _env_float("PERTURB_ANCHOR_MAX_FRAC", 0.20)        # post-warmup cap: fraction of remaining post-flip budget
ANCHOR_MIN_MARGIN_GAIN = _env_float("PERTURB_ANCHOR_MIN_MARGIN_GAIN", 0.1)  # min best-|margin| drop/chunk to count as progress
ANCHOR_STALL_CHUNKS = _env_int("PERTURB_ANCHOR_STALL_CHUNKS", 4)     # consecutive weak chunks (post-warmup) before bailing
ANCHOR_PUSH_MARGIN = _env_float("PERTURB_ANCHOR_PUSH_MARGIN", 8.0)   # once best margin <= -this, never stall-bail (close to CEIL)
# Phase-3 refine budget is decided PER CANDIDATE by how close it already is to CEIL, not by whether the
# anchor saturated: a screened candidate at CW margin <= -COUPLED_REFINE_DEEP_MARGIN is close enough to be
# worth the full MAX_ITERATIONS deepen; a shallow one gets the bounded ITERATIONS_PER_K pass so it can't
# grind an unreachable CEIL. This lets a good candidate recover even when the anchor itself fell short.
COUPLED_REFINE_DEEP_MARGIN = _env_float("PERTURB_COUPLED_REFINE_DEEP_MARGIN", 6.0)

# --- Sparse-then-grow post-flip (PERTURB_POSTFLIP_STRATEGY=grow|both) -----------------------------
# Grow reaches the score peak from BELOW. From a saturated dense anchor it keeps the top-retention
# GROW_CORE_FRAC·K_anchor coords (a lean guided core whose ranking reflects DEEP-margin value), then
# repeatedly adds a block of the steepest feasible margin-gain inactive coords — block = max(
# GROW_MIN_BATCH, GROW_ADD_FRAC·K_cur) — re-linearizing the gradient at each add. Each cumulative state
# folds through the score-ranked Bank. The climb stops when the margin SATURATES (<= -CEIL: further
# coords only raise RMSE, so the score can only fall) or when GROW_PATIENCE consecutive adds set no new
# Bank best (pre-flip: no new margin low). GROW_MAX_STEPS is a safety cap. A second lineage grows from
# the sparsest banked flip when it is distinct. Overshoot past the peak is pruned by the SupportRefine
# deletion pass that runs after every post-flip strategy, so grow lands at the peak either way.
GROW_CORE_FRAC = _env_float("PERTURB_GROW_CORE_FRAC", 0.15)   # lean core size as a fraction of anchor K
GROW_ADD_FRAC = _env_float("PERTURB_GROW_ADD_FRAC", 0.35)     # per-step block add as a fraction of current K
GROW_MIN_BATCH = _env_int("PERTURB_GROW_MIN_BATCH", 32)       # floor on the per-step block add
GROW_PATIENCE = _env_int("PERTURB_GROW_PATIENCE", 3)          # non-improving adds tolerated before stopping
GROW_MAX_STEPS = _env_int("PERTURB_GROW_MAX_STEPS", 24)       # safety cap on grow steps per lineage

# --- Unified Frontier Scheduler (PERTURB_POSTFLIP_STRATEGY=ufs) -----------------------------------
# ANYTIME post-flip: instead of a fixed waterfall (anchor -> K_sat bisection -> refine sweep), keep a
# small beam ("frontier") of warm-start States at different K, fit an online margin surrogate m̂(K) from
# every (K,margin) seen, and each iteration execute the single move with the highest expected score gain:
#   DEEPEN(node)     — push a node's margin toward CEIL at its own K (raises the 0.03 bonus).
#   SHRINK(node,K')  — warm-start a node down to a surrogate-chosen sparser K' (lower RMSE).
#   GROW(node)       — climb a sparse node up to saturation (Part-2 sparse-then-grow; reaches K̂* from below).
# The surrogate's argmax K̂* = argmax_K [pert(K) + margin_bonus(m̂(K))] steers both SHRINK and GROW toward
# the score peak; a move is skipped when its analytic ceiling can't beat the Bank. Budget is one shared
# pool (no per-phase slices => no starvation, no unused tail). On convergence (no move beats the Bank by
# SCORE_TOL) with budget left, a bounded PartialRestart diversifies to escape local optima. INVARIANT:
# the densest node is never evicted, so a saturatable anchor always survives (the 0.03 bonus stays
# reachable — prevents the sparse-drift regression). Every candidate folds through the score-ranked Bank.
UFS_FRONTIER_CAP = _env_int("PERTURB_UFS_FRONTIER_CAP", 6)    # live warm-start nodes kept in the frontier
UFS_CHUNK_ITERS = _env_int("PERTURB_UFS_CHUNK_ITERS", 30)     # optimizer iters per DEEPEN/SHRINK burst
UFS_GRID = _env_int("PERTURB_UFS_GRID", 48)                   # geometric-K grid resolution for the surrogate argmax
UFS_DEEPEN_MIN_GAIN = _env_float("PERTURB_UFS_DEEPEN_MIN_GAIN", 0.2)  # min |margin| drop for a DEEPEN to count as progress
UFS_RESTARTS = _env_int("PERTURB_UFS_RESTARTS", 3)           # PartialRestart diversifications allowed on convergence
UFS_MAX_ACTIONS = _env_int("PERTURB_UFS_MAX_ACTIONS", 400)   # hard safety cap on scheduled actions per call

# --- C2: diminishing-returns early stop for STANDALONE margin-deepen calls (OptimizeFixedK) --------
# Complements the QuickAnchor (which already stall-stops the anchor). This covers the OTHER deepen calls
# — binary-search retries and Phase-3 refines — so a shallow-but-stalled candidate stops grinding an
# (effectively) unreachable CEIL and hands the remaining budget to the K/RMSE search. The window is set
# LARGER than ANCHOR_CHUNK_ITERS on purpose, so it can never fill (and thus never fire) inside the
# anchor's chunks — that path keeps its own stall logic. FIND (deepen_target None) and pre-floor margins
# are exempt: the floor guarantees a strong margin (>= this depth) even on the earliest possible stop.
DEEPEN_STALL_STOP = _env_bool("PERTURB_DEEPEN_STALL_STOP", True)
DEEPEN_STALL_FLOOR = _env_float("PERTURB_DEEPEN_STALL_FLOOR", 8.0)   # only early-stop once margin <= -this
DEEPEN_STALL_WINDOW = _env_int("PERTURB_DEEPEN_STALL_WINDOW", 12)    # iters in the sliding best-margin window
DEEPEN_STALL_MIN_IMPROVE = _env_float("PERTURB_DEEPEN_STALL_MIN_IMPROVE", 0.3)  # min |margin| gain/window to keep going

# --- C3 (opt-in, default OFF): warm-start the post-flip anchor from the Bank's sparsest flip ---------
# Anchor from the already-banked sparse flip instead of the dense K_init FIND state: tighter binary-
# search bracket + sparse start. OFF by default — the dense anchor doubles as a saturating-upper-bound
# validator (a sparse start cannot distinguish "K too small to saturate" from "image genuinely hard").
POSTFLIP_ANCHOR_FROM_BANK = _env_bool("PERTURB_POSTFLIP_ANCHOR_FROM_BANK", False)

# --- INNER support-quality refinement (all strategies, after the K-search) -----------------------
# The K-search picks a good cardinality; it does NOT guarantee the best coordinates/signs AT that K
# (WarmStartSmallerK ranks by first-order retention; the deepen swaps accept by margin). This pass takes
# the current best flip and, gradient-prefiltered to ONE batched forward per round, tries: EXACT deletion
# of redundant coords (individual + aggregate removals) and score-accepted one-for-one SWAPS (weakest-out
# / strongest-feasible-in). Every candidate folds through the score-ranked Bank, so acceptance is by FULL
# score automatically and it can only RAISE the returned score (or, out of budget, do nothing).
REFINE_SUPPORT = _env_bool("PERTURB_REFINE_SUPPORT", True)
REFINE_ROUNDS = _env_int("PERTURB_REFINE_ROUNDS", 4)                    # deletion+swap rounds (re-linearized)
REFINE_DELETION_POOL = _env_int("PERTURB_REFINE_DELETION_POOL", 128)   # weakest active coords tested for deletion
REFINE_SWAP_POOL = _env_int("PERTURB_REFINE_SWAP_POOL", 128)           # strongest inactive coords for swap-in
REFINE_SWAP_PROPOSALS = _env_int("PERTURB_REFINE_SWAP_PROPOSALS", 128)  # one-for-one swap proposals per round
# strict only: accept a smaller-K rung while its total score is within SCORE_TOL of the best (a positive
# tolerance lets the descent push through score noise / a shallow dip before declaring the peak).
SCORE_TOL = _env_float("PERTURB_SCORE_TOL", 0.0003)
# Novelty floor: the validator's novelty bonus saturates at NOVELTY_TARGET_PIXELS changed spatial
# pixels; pruning below it trades ~0 perturbation gain for lost novelty. Floor cardinality reduction at
# ~3x that in channels (worst-case 3 channels/pixel) so neither strategy churns in the sub-floor regime.
# Correctness is still guaranteed by the score-ranked Bank; this only saves wasted evals.
NOVELTY_TARGET_PIXELS = _env_int("ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS", 8)

# --- q=2 fallback (last resort when no single-byte flip exists) ----------------------------------
# If the whole q=1 (L∞=1/255) search bank is empty, retry ONCE at q=2 (L∞=2/255). Doubling the per-
# coordinate reach makes flipping far easier, so a small support usually flips on the first grow-ladder
# rung. Capped at q=2 (never q>=3): q=2 tops out at ~0.77 total score, but that dwarfs the 0 of a clean
# return (which also drags the validator's 300-sample average). Phase A only, time-boxed — a fast net.
FALLBACK_Q2 = _env_bool("PERTURB_FALLBACK_Q2", True)
FALLBACK_Q2_SECONDS = _env_float("PERTURB_FALLBACK_Q2_SECONDS", 6)    # reserved q=2 budget; reclaimed the instant a q=1 flip banks

# --- Adaptive hyperparameter controller ---------------------------------------------------
# Starts every tunable knob at its env value, then watches the BEST margin over a sliding window.
# On a stall (best margin barely moved) it escalates exploration one level (bigger blocks, higher
# eta/temperature, more proposals, more frequent swaps, larger restarts); on strong progress it
# relaxes a level back toward the env baseline. level L scales a knob by TUNE_FACTOR**L (capped).
# Higher TEMPERATURE at high levels also de-saturates the BIG_INIT logits, unfreezing Phase B.
ADAPTIVE_TUNE = _env_bool("PERTURB_ADAPTIVE_TUNE", True)   # enable the controller (active during optim)
TUNE_INTERVAL = _env_int("PERTURB_TUNE_INTERVAL", 50)     # optim iterations per evaluation window
TUNE_MIN_IMPROVE = _env_float("PERTURB_TUNE_MIN_IMPROVE", 0.02)  # abs best-margin drop/window => progress
TUNE_MIN_REL = _env_float("PERTURB_TUNE_MIN_REL", 0.01)   # relative drop/window => progress
TUNE_GOOD_REL = _env_float("PERTURB_TUNE_GOOD_REL", 0.05)  # relax a level when relative drop exceeds this
TUNE_FACTOR = _env_float("PERTURB_TUNE_FACTOR", 1.5)      # per-level multiplier
TUNE_MAX_LEVEL = _env_int("PERTURB_TUNE_MAX_LEVEL", 6)    # escalation ceiling
TUNE_BLOCK_FRAC_CAP = _env_float("PERTURB_TUNE_BLOCK_FRAC_CAP", 0.25)  # block-fraction ceiling
TUNE_PROPOSAL_CAP = _env_int("PERTURB_TUNE_PROPOSAL_CAP", 128)  # proposal-count ceiling
TUNE_RESTART_FRAC_CAP = _env_float("PERTURB_TUNE_RESTART_FRAC_CAP", 0.5)  # restart-fraction ceiling
TUNE_TEMPERATURE_CAP = _env_float("PERTURB_TUNE_TEMPERATURE_CAP", 16.0)  # temperature ceiling
