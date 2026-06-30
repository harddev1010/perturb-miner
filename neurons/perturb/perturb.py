"""perturb.py — Dynamic Sparse Fixed-q Attack (Phases A-E) with feature guidance.

A complete fixed-q, fixed-K framework where every edit is delta_i in {-q, 0, +q}, q = k_min/255,
and exactly K channels are active during discovery. The perturbed image is always built as
    delta = q * mask * sign ,  x_adv = x + delta
on the exact uint8 byte grid, so every candidate is validator-faithful, and box-clipping directions
are forbidden.

Phases (see the function blocks below):
  * Phase A  — InitializeAttack: seed a fixed-K support from FOUR sources — a clean untargeted
    gradient, target-specific clean gradients, random-start gradient reservoirs, and a
    FEATURE-GUIDED reservoir (Q1) that gates the input-gradient saliency by a hidden-layer spatial
    relevance map. The merged support initializes the continuous mask logits `a` and sign scores `v`.
  * Phase B  — DynamicMaskOptimizationStep: re-linearize the gradient at the CURRENT adversarial
    image, update sign scores (momentum) and the dense mask logits (straight-through), rebuild the
    hard top-K support, and verify exactly.
  * Phase C  — ExactBlockSwap: replace weak selected coordinates with promising unselected ones,
    drawing the swap-in pool from current/clean/path/feature/frontier/random candidates, evaluating
    whole alternative support blocks exactly, and accepting only strict improvements.
  * Phase D  — PartialRestart: when the support stagnates, preserve the strong core and re-seed only
    the weakest fraction.
  * Phase E  — ReduceCardinality: after a flip at K_high, warm-start smaller supports and reoptimize.

CURRENT MODE (per request):
  * Timeouts are IGNORED (K.IGNORE_TIMEOUT) — the engine does not gate on the deadline.
  * The optimizer (Phases B-E) is IMPLEMENTED but NOT RUN. `search()` runs Phase A and returns as
    soon as a flipping candidate is found; OptimizeFixedK / ReduceCardinality sit behind
    `if K.RUN_OPTIM:` (PERTURB_RUN_OPTIM, default off). Flip that env var to enable the full loop.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from perturbnet.model import logits_for_images

from . import constants as K
from .calibration import env_fingerprint, get_calibrator
from .utils import (
    Bank,
    Context,
    apply_delta_bytes,
    batch_eval,
    estimate_k,
    exact_worst_margin,
    logits_of,
    loss_grad,
    margin_and_grad,
    movable,
    out_of_budget,
    top_wrong_classes,
)

logger = logging.getLogger(__name__)

# Match the validator's numeric regime: cuDNN convolutions use TF32 (PERTURB_TF32_ON, default on for
# CUDA); matmul TF32 stays off (the validator default). EfficientNetV2-L is conv-dominated, so cuDNN is
# the axis that matters; the envelope (utils.batch_eval) brackets the residual cross-GPU/library drift.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = K.TF32_ON
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class _FirstFlipFound(Exception):
    """Raised from the eval choke point to unwind the search immediately when PERTURB_RETURN_FIRST_FLIP
    is set and a returnable flip has just been banked. The Bank already holds it, so perturb() returns it."""


def _oob(ctx: Context) -> bool:
    """Two-stage budget gate.

    FIND stage (no flip banked yet): unbounded wall-clock when PERTURB_IGNORE_TIMEOUT is set — the
    search closes on max-iters / convergence, not on a timer (else the original deadline applies).
    OPTIM stage (a flip exists): always enforce the armed deadline (ctx.deadline = flip_time +
    OPTIM_SECONDS), so post-flip optimization is capped regardless of IGNORE_TIMEOUT.
    """
    if ctx.first_flip_time is None:
        return False if K.IGNORE_TIMEOUT else out_of_budget(ctx)
    return out_of_budget(ctx)


def _maybe_arm_optim_deadline(ctx: Context) -> None:
    """Arm the post-flip optimization clock the first time a flip is banked: cap ctx.deadline at
    flip_time + OPTIM_SECONDS so all further optimization (refine + Phase E) finishes within it."""
    if ctx.optim_seconds <= 0.0 or ctx.first_flip_time is not None:
        return
    if ctx.bank.has_flip:
        ctx.first_flip_time = time.time()
        ctx.deadline = min(ctx.deadline, ctx.first_flip_time + ctx.optim_seconds)
        logger.info(f"[optim] first flip banked -> arming {ctx.optim_seconds:.1f}s optim budget "
                    f"(deadline in {ctx.deadline - time.time():.1f}s)")


# ==========================================================================================
# State — the optimization bundle described in the framework's "Common data structures".
# ==========================================================================================
@dataclass
class State:
    a: torch.Tensor                       # continuous mask logits
    v: torch.Tensor                       # continuous sign scores
    mask: torch.Tensor                    # binary hard mask (bool): TopK(a)
    sign: torch.Tensor                    # -1 / 0 / +1 per channel
    margin: float                         # exact margin of the current discrete state
    x_adv: torch.Tensor                   # current adversarial image (chw float)
    path_ema: torch.Tensor               # EMA of candidate usefulness
    path_max: torch.Tensor               # max usefulness observed along the path
    clean_score: torch.Tensor            # candidate score measured at the clean image
    clean_gradient: torch.Tensor         # gradient at the clean image (flat)
    gradient: torch.Tensor | None = None         # gradient at the current x_adv (flat)
    previous_gradient: torch.Tensor | None = None
    previous_mask: torch.Tensor | None = None
    current_score: torch.Tensor | None = None
    turnover: float = 0.0


# ==========================================================================================
# Low-level feasible-direction / score helpers (vectorized over all N channels).
#   q       = ctx.q          (one feasible step in [0,1] space)
#   a step  = ±k_min bytes   (clean_u8 ± k_min)
# A direction is forbidden when it would leave the valid [0,255] byte box.
# ==========================================================================================
def _feasible_dirs(ctx: Context) -> tuple[torch.Tensor, torch.Tensor]:
    """(can_up, can_down): per-channel masks for the +k_min / -k_min byte steps staying in [0,255]."""
    can_up = (ctx.clean_u8 + ctx.k_min) <= 255.0
    can_down = (ctx.clean_u8 - ctx.k_min) >= 0.0
    return can_up, can_down


def _best_direction(ctx: Context, grad_flat: torch.Tensor) -> torch.Tensor:
    """BestDirection (vectorized): the feasible fixed-q sign s maximizing predicted_decrease = -g·q·s.
    Interior coords -> -sign(g); box-clipped coords fall back to the other feasible sign, else 0."""
    can_up, can_down = _feasible_dirs(ctx)
    neg_inf = torch.full_like(grad_flat, float("-inf"))
    score_up = torch.where(can_up, -grad_flat * ctx.q, neg_inf)     # s = +1
    score_down = torch.where(can_down, grad_flat * ctx.q, neg_inf)  # s = -1
    use_up = score_up >= score_down
    sign = torch.where(use_up, 1.0, -1.0).to(grad_flat.dtype)
    sign[~(can_up | can_down)] = 0.0
    return sign


def _candidate_scores(ctx: Context, grad_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ComputeCandidateScores: (score, preferred_sign). score_i = max(-g_i·q·s_i, 0) at the best
    feasible s_i; for an unconstrained interior coord this is simply q·|g_i|."""
    sign = _best_direction(ctx, grad_flat)
    score = (-grad_flat * ctx.q * sign).clamp(min=0.0)
    return score, sign


def _feasible_sign_from_value(ctx: Context, v_flat: torch.Tensor) -> torch.Tensor:
    """FeasibleSignFromValue: sign(v) if that byte step is feasible, else the opposite if feasible, else 0."""
    can_up, can_down = _feasible_dirs(ctx)
    sign = v_flat.sign().to(v_flat.dtype)
    bad_up = (sign > 0) & ~can_up
    bad_down = (sign < 0) & ~can_down
    sign[bad_up] = torch.where(can_down[bad_up], -1.0, 0.0).to(sign.dtype)
    sign[bad_down] = torch.where(can_up[bad_down], 1.0, 0.0).to(sign.dtype)
    return sign


def _topk_mask(logits: torch.Tensor, budget: int) -> torch.Tensor:
    """TopKMask: a boolean mask with the K largest entries of `logits` set."""
    budget = max(0, min(int(budget), logits.numel()))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    if budget > 0:
        mask[torch.topk(logits, budget).indices] = True
    return mask


def _grad_at(ctx: Context, x_chw: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Exact CW margin and its flat input gradient at x_chw."""
    m, g = margin_and_grad(ctx.model, x_chw, ctx.target_index)
    return m, g.view(-1)


def _make_delta(ctx: Context, mask: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """delta = q * mask * sign as an integer byte delta (±k_min on the active support)."""
    return (mask.to(ctx.clean_u8.dtype) * sign) * float(ctx.k_min)


# ==========================================================================================
# Feature guidance (Q1) — a hidden-layer spatial relevance map gating the input saliency.
# ==========================================================================================
def _feature_relevance(ctx: Context, x_chw: torch.Tensor) -> torch.Tensor | None:
    """Spatial feature relevance upsampled to the input grid, flat over (C,H,W) coords.

    feature_relevance[u,v] = sum_c |feature_map[c,u,v] * d margin / d feature_map[c,u,v]|, normalized,
    then bilinearly upsampled to (H_in, W_in) and tiled across the C input channels so it indexes the
    same flat (c·H·W + h·W + w) layout as ctx.clean_u8. Returns None if no conv layer is hookable.
    """
    model = ctx.model
    layer = getattr(model, "features", None)
    if layer is None:
        return None
    store: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        output.retain_grad()
        store["f"] = output

    handle = layer.register_forward_hook(hook)
    try:
        x = x_chw.detach().clone().requires_grad_(True)
        logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
        others = logits.clone()
        others[ctx.target_index] = float("-inf")
        margin = logits[ctx.target_index] - others.max()
        model.zero_grad(set_to_none=True)
        margin.backward()
        feat = store.get("f")
        fgrad = None if feat is None else feat.grad
        if feat is None or fgrad is None:
            return None
        # [1,C,H,W] -> spatial relevance [H,W]
        rel = (feat * fgrad).abs().sum(dim=1)[0]
    finally:
        handle.remove()

    rel = rel / (rel.max() + 1e-12)
    h_in, w_in = int(ctx.shape[1]), int(ctx.shape[2])
    up = F.interpolate(rel[None, None], size=(h_in, w_in), mode="bilinear", align_corners=False)[0, 0]
    up = up / (up.max() + 1e-12)
    # Tile the (H*W) spatial map across the C channels to match the (c·H·W + ...) flat layout.
    return up.reshape(-1).repeat(int(ctx.shape[0])).to(ctx.clean_u8.dtype)


def _feature_score(ctx: Context, x_chw: torch.Tensor, clean_score: torch.Tensor) -> torch.Tensor | None:
    """Combine spatial feature relevance with the per-channel input saliency.

    Gate mode (default, safer): keep clean_score only inside the most-relevant pixels (feature maps
    locate regions; the input gradient still decides which RGB channel + direction is useful).
    Pure mode:  feature_score[i] = clean_score[i] * (eps + relevance[h,w])^beta.
    """
    relevance = _feature_relevance(ctx, x_chw)
    if relevance is None:
        return None
    if K.FEATURE_GATE:
        hw = int(ctx.shape[1]) * int(ctx.shape[2])
        spatial = relevance[:hw]
        pixel_quota = max(1, int(round(K.FEATURE_PIXEL_QUOTA_FRAC * hw)))
        keep_pix = torch.zeros_like(spatial, dtype=torch.bool)
        keep_pix[torch.topk(spatial, min(pixel_quota, hw)).indices] = True
        keep = keep_pix.repeat(int(ctx.shape[0]))
        score = clean_score.clone()
        score[~keep] = 0.0
        return score
    return clean_score * (K.FEATURE_EPS + relevance).pow(K.FEATURE_BETA)


# ==========================================================================================
# Candidate construction for seeding (gradient -> latent; a fixed-K projection is a legal ternary
# candidate). A "field" latent = move_dir · |g|: sign(field) is the toward-flip action, |field| = |g|.
# ==========================================================================================
def _field(ctx: Context, x: torch.Tensor, kind: str, targets: list[int]):
    """One attack loss at x -> (value, latent), latent = move_dir · |g| (toward-flip per channel)."""
    if kind == "soft":
        value, move_dir, score = loss_grad(ctx.model, x, ctx.target_index, "soft",
                                            top_wrong=targets, tau=K.TAU)
    else:
        value, move_dir, score = loss_grad(ctx.model, x, ctx.target_index, kind)
    return value, move_dir * score


def _legal(ctx: Context, latent: torch.Tensor):
    """Mask out box-clipped actions; return (|latent| with illegal coords zeroed, legal count)."""
    mag = latent.abs().clone()
    mag[~movable(ctx.clean.view(-1), latent.sign())] = 0.0
    return mag, int((mag > 0).sum().item())


def _project_topk(ctx: Context, latent: torch.Tensor, budget: int) -> torch.Tensor:
    """Pi_{T_K}: the K legal coords of largest |latent|, each stepped ±k_min along sign(latent)."""
    mag, valid = _legal(ctx, latent)
    budget = min(max(1, budget), valid)
    d = torch.zeros_like(ctx.clean_u8)
    if budget < 1:
        return d
    idx = torch.topk(mag, budget).indices
    d[idx] = latent.sign()[idx] * float(ctx.k_min)
    return d


def _project_sampled(ctx: Context, latent: torch.Tensor, budget: int) -> torch.Tensor:
    """Near-tie variant: draw K coords from the top (TIE_MULT·K) by softmax(|latent|/temp)."""
    mag, valid = _legal(ctx, latent)
    budget = min(max(1, budget), valid)
    d = torch.zeros_like(ctx.clean_u8)
    if budget < 1:
        return d
    order = torch.argsort(mag, descending=True)
    pool = order[:min(valid, max(budget, K.TIE_MULT * budget))]
    w = torch.softmax(mag[pool] / max(K.TIE_TEMP, 1e-6), dim=0)
    pick = pool[torch.multinomial(w, budget, replacement=False)]
    d[pick] = latent.sign()[pick] * float(ctx.k_min)
    return d


def _kc_of(ctx: Context, latent: torch.Tensor, value: float) -> int:
    """Linearized crossing size K_hat = min{K : q·sum_{i<=K}|latent|_(i) >= value + kappa}."""
    mag, valid = _legal(ctx, latent)
    if valid == 0:
        return 0
    order = torch.argsort(mag, descending=True)
    return estimate_k(value + max(ctx.kappa, 0.0), mag[order][:valid], ctx.q)


def _support_ladder(ctx: Context, latent: torch.Tensor, value: float) -> list[torch.Tensor]:
    """A multi-scale support sweep around K_hat: exact top-K plus randomized near-tie orders at each
    scale, plus the fully dense legal support. Returns byte deltas (one candidate each)."""
    _, valid = _legal(ctx, latent)
    if valid == 0:
        return []
    kc = _kc_of(ctx, latent, value)
    out: list[torch.Tensor] = []
    seen: set[int] = set()
    for f in K.K_MULTS:
        budget = min(max(1, int(round(f * kc))), valid)
        if budget in seen:
            continue
        seen.add(budget)
        out.append(_project_topk(ctx, latent, budget))
        for _ in range(K.ORDER_VARIANTS):
            out.append(_project_sampled(ctx, latent, budget))
    out.append(_project_topk(ctx, latent, valid))  # fully dense legal support
    return out


# ==========================================================================================
# Evaluation + bank folding (validator-faithful, envelope-aware).
# ==========================================================================================
def _eval(ctx: Context, deltas: list[torch.Tensor]) -> list[dict]:
    """Verify a batch of byte deltas (worst-case margin, SSIM/PSNR, L∞ band), attach each delta to
    its result, and fold into the Bank."""
    deltas = [d for d in deltas if d is not None]
    if not deltas:
        return []
    cands = [apply_delta_bytes(ctx.clean_u8, d, ctx.shape) for d in deltas]
    res = batch_eval(ctx, cands)
    for r, d in zip(res, deltas):
        r["delta"] = d
    ctx.bank.consider(res)
    # Master early-return: stop the instant a returnable flip exists (skips optim entirely).
    if K.RETURN_FIRST_FLIP and _incumbent(ctx) is not None:
        raise _FirstFlipFound()
    _maybe_arm_optim_deadline(ctx)
    return res


def _eval_states(ctx: Context, masks: list[torch.Tensor], signs: list[torch.Tensor]) -> list[dict]:
    """BatchEvaluateMargins: verify a batch of (mask, sign) ternary states exactly via the Bank."""
    return _eval(ctx, [_make_delta(ctx, m, s) for m, s in zip(masks, signs)])


def _evaluate_state(ctx: Context, mask: torch.Tensor, sign: torch.Tensor) -> tuple[float, torch.Tensor, dict | None]:
    """EvaluateState: build delta, apply, verify -> (margin, x_adv, result)."""
    res = _eval(ctx, [_make_delta(ctx, mask, sign)])
    if not res:
        x_adv = apply_delta_bytes(ctx.clean_u8, _make_delta(ctx, mask, sign), ctx.shape)
        return float("inf"), x_adv, None
    return res[0]["margin"], res[0]["cand"], res[0]


def _incumbent(ctx: Context) -> dict | None:
    """The returnable incumbent: smallest envelope-safe flip, or (if unsafe allowed) smallest flip."""
    if ctx.bank.best_safe is not None:
        return ctx.bank.best_safe
    return ctx.bank.best_flip if ctx.allow_unsafe else None


# ==========================================================================================
# Phase A — InitializeAttack (multi-source seeding, including feature guidance).
# ==========================================================================================
def _top_indices(score: torch.Tensor, count: int) -> torch.Tensor:
    count = max(0, min(int(count), score.numel()))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    return torch.topk(score, count).indices


def _random_start_grad(ctx: Context) -> torch.Tensor:
    """A1/A3 helper: gradient at a random sparse ±k_min start (a fresh, off-clean basin)."""
    n = ctx.clean_u8.numel()
    cnt = max(1, int(round(K.RANDOM_START_FRAC * n)))
    idx = torch.randperm(n, device=ctx.clean_u8.device)[:cnt]
    rnd_mask = torch.zeros(n, dtype=torch.bool, device=ctx.clean_u8.device)
    rnd_mask[idx] = True
    rnd_sign = _feasible_sign_from_value(ctx, torch.randn(n, device=ctx.clean_u8.device))
    x_rand = apply_delta_bytes(ctx.clean_u8, _make_delta(ctx, rnd_mask, rnd_sign), ctx.shape)
    _, g = _grad_at(ctx, x_rand)
    return g


def InitializeAttack(ctx: Context, targets: list[int], K_init: int) -> tuple[State, list[dict]]:
    """Phase A. Build a fixed-K support from clean + targeted + random-start + feature-guided
    candidates, initialize the continuous logits/signs, and (so a flip can be found without the
    optimizer) verify each source's support ladder into the Bank. Returns (state, seed_results)."""
    n = ctx.clean_u8.numel()
    candidate_lists: list[torch.Tensor] = []
    aggregate_gradient = torch.zeros(n, device=ctx.clean_u8.device)
    seed_deltas: list[torch.Tensor] = []

    # A1. Standard untargeted clean gradient.
    clean_margin, clean_gradient = _grad_at(ctx, ctx.clean)
    clean_score, clean_sign = _candidate_scores(ctx, clean_gradient)
    candidate_lists.append(_top_indices(clean_score, round(0.5 * K_init)))
    aggregate_gradient += clean_gradient
    seed_deltas += _support_ladder(ctx, clean_sign * clean_score, clean_margin)
    logger.info(f"[phaseA] A1 clean gradient: margin={clean_margin:.4f} "
                f"legal={(clean_score > 0).sum().item()} top={round(0.5 * K_init)}")

    # A2. Target-specific clean gradients.
    per_target = max(1, round(K_init / (4 * max(1, K.TARGET_COUNT))))
    for c in targets[:K.TARGET_COUNT]:
        value, latent = _field(ctx, ctx.clean, f"pair:{c}", targets)
        # latent = move_dir·|g| = -sign(g)·|g| = -g, so the raw gradient is -latent.
        tgrad = -latent
        tscore, _ = _candidate_scores(ctx, tgrad)
        candidate_lists.append(_top_indices(tscore, per_target))
        aggregate_gradient += tgrad
        seed_deltas += _support_ladder(ctx, latent, value)
        logger.info(f"[phaseA] A2 target={c}: value={value:.4f} kc={_kc_of(ctx, latent, value)} top={per_target}")

    # A3. Random-start gradient reservoirs.
    per_random = max(1, round(K_init / (4 * max(1, K.RANDOM_START_COUNT))))
    for r in range(K.RANDOM_START_COUNT):
        rgrad = _random_start_grad(ctx)
        rscore, rsign = _candidate_scores(ctx, rgrad)
        candidate_lists.append(_top_indices(rscore, per_random))
        seed_deltas += _support_ladder(ctx, rsign * rscore, clean_margin)
        logger.info(f"[phaseA] A3 random-start {r + 1}/{K.RANDOM_START_COUNT}: top={per_random}")

    # A4. Feature-guided candidate reservoir (Q1).
    if K.FEATURE_GUIDED:
        fscore = _feature_score(ctx, ctx.clean, clean_score)
        if fscore is not None:
            fquota = max(1, round(K.FEATURE_QUOTA_FRAC * K_init))
            candidate_lists.append(_top_indices(fscore, fquota))
            seed_deltas += _support_ladder(ctx, clean_sign * fscore, clean_margin)
            logger.info(f"[phaseA] A4 feature-guided ({'gate' if K.FEATURE_GATE else 'pure'}): "
                        f"top={fquota} nonzero={(fscore > 0).sum().item()}")
        else:
            logger.info("[phaseA] A4 feature-guided: no hookable conv layer -> skipped")
    else:
        logger.info("[phaseA] A4 feature-guided: disabled")

    # A4'/merge. Union of candidate lists -> highest-ranked K distinct coords; fill from clean_score.
    union = torch.cat([c for c in candidate_lists if c.numel() > 0]) if candidate_lists else \
        torch.empty(0, dtype=torch.long, device=ctx.clean_u8.device)
    union = torch.unique(union)
    if union.numel() >= K_init:
        # rank the union by clean_score and keep the top K
        order = torch.argsort(clean_score[union], descending=True)
        initial_support = union[order][:K_init]
    else:
        fill = _top_indices(clean_score, n)
        merged = torch.cat([union, fill])
        # preserve order: dedup keeping first occurrence (union coords first, then clean_score fill)
        seen = torch.zeros(n, dtype=torch.bool, device=ctx.clean_u8.device)
        keep: list[int] = []
        for i in merged.tolist():
            if not seen[i]:
                seen[i] = True
                keep.append(i)
            if len(keep) >= K_init:
                break
        initial_support = torch.tensor(keep, dtype=torch.long, device=ctx.clean_u8.device)
    logger.info(f"[phaseA] merged support: union={union.numel()} -> initial_support={initial_support.numel()} "
                f"(K_init={K_init})")

    # A5. Initialize mask logits: small noise + a large boost on the seeded support.
    a = K.MASK_NOISE * torch.randn(n, device=ctx.clean_u8.device)
    a[initial_support] += K.BIG_INIT

    # A6. Initialize sign scores v = -aggregate_gradient -> feasible signs; hard mask = TopK(a).
    v = -aggregate_gradient
    sign = _feasible_sign_from_value(ctx, v)
    mask = _topk_mask(a, K_init)

    path_ema = clean_score.clone()
    path_max = clean_score.clone()

    margin, x_adv, _ = _evaluate_state(ctx, mask, sign)

    # Verify every seed candidate so a flip can be banked during Phase A alone (optimizer off).
    seed_results = _eval(ctx, seed_deltas)
    flips = sum(1 for r in seed_results if r.get("quality"))
    best_seed = min((r["margin"] for r in seed_results), default=float("inf"))
    logger.info(f"[phaseA] seeded+verified {len(seed_deltas)} candidates: quality_flips={flips} "
                f"best_margin={best_seed:.4f} init_state_margin={margin:.4f}")

    state = State(
        a=a, v=v, mask=mask, sign=sign, margin=margin, x_adv=x_adv,
        path_ema=path_ema, path_max=path_max, clean_score=clean_score, clean_gradient=clean_gradient,
        gradient=clean_gradient.clone(),
    )
    return state, seed_results


# ==========================================================================================
# Phase B — DynamicMaskOptimizationStep (re-linearized straight-through update).
# ==========================================================================================
def DynamicMaskOptimizationStep(ctx: Context, state: State, K_cur: int) -> State:
    """One dynamic step: recompute the gradient at the CURRENT x_adv, update sign scores and the dense
    mask logits (straight-through), rebuild the hard support, and verify exactly."""
    # B1. Current adversarial image.
    state.mask = _topk_mask(state.a, K_cur)
    delta = _make_delta(ctx, state.mask, state.sign)
    state.x_adv = apply_delta_bytes(ctx.clean_u8, delta, ctx.shape)

    # B2. Exact current margin + gradient.
    current_margin, gradient = _grad_at(ctx, state.x_adv)
    current_score, current_preferred_sign = _candidate_scores(ctx, gradient)

    # B3. Update path history.
    state.path_ema = K.HISTORY_BETA * state.path_ema + (1 - K.HISTORY_BETA) * current_score
    state.path_max = torch.maximum(state.path_max, current_score)

    # B4. Update sign scores for all coordinates.
    state.v = K.SIGN_MOMENTUM * state.v + (1 - K.SIGN_MOMENTUM) * (-gradient)
    new_sign = _feasible_sign_from_value(ctx, state.v)
    proposed_sign = state.sign.clone()
    inactive = ~state.mask
    proposed_sign[inactive] = new_sign[inactive]                       # always refresh inactive coords
    retention = -gradient * ctx.q * state.sign                          # refresh harmful active coords
    harmful = state.mask & (retention < 0)
    proposed_sign[harmful] = new_sign[harmful]

    # B5. Dense mask-logit update (straight-through sigmoid).
    mask_gradient = ctx.q * proposed_sign * gradient
    soft_mask = torch.sigmoid(state.a / max(K.TEMPERATURE, 1e-6))
    derivative = soft_mask * (1 - soft_mask) / max(K.TEMPERATURE, 1e-6)
    state.a = state.a - K.ETA_MASK * mask_gradient * derivative

    # B6. Build the new hard support; new coords use their current preferred signs.
    proposed_mask = _topk_mask(state.a, K_cur)
    newly_active = proposed_mask & ~state.mask
    proposed_sign[newly_active] = current_preferred_sign[newly_active]

    # B7. Exact evaluation.
    proposed_margin, proposed_x_adv, _ = _evaluate_state(ctx, proposed_mask, proposed_sign)

    # B8. Commit.
    state.previous_mask = state.mask
    state.previous_gradient = gradient
    state.turnover = float((proposed_mask ^ state.previous_mask).sum().item()) / max(1, K_cur)
    state.mask = proposed_mask
    state.sign = proposed_sign
    state.margin = proposed_margin
    state.x_adv = proposed_x_adv
    state.current_score = current_score
    state.gradient = gradient
    return state


# ==========================================================================================
# Phase C — ExactBlockSwap (swap-in / swap-out pools + exact batched proposals).
# ==========================================================================================
def _top_restricted(score: torch.Tensor, allowed: torch.Tensor, count: int) -> torch.Tensor:
    """TopIndicesRestricted: the top-`count` indices of `score` restricted to `allowed` coords."""
    count = max(0, int(count))
    if count == 0 or allowed.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    vals = score[allowed]
    take = min(count, allowed.numel())
    return allowed[torch.topk(vals, take).indices]


def _bottom_restricted(score: torch.Tensor, allowed: torch.Tensor, count: int) -> torch.Tensor:
    """BottomIndicesRestricted: the lowest-`count` indices of `score` restricted to `allowed`."""
    count = max(0, int(count))
    if count == 0 or allowed.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    vals = score[allowed]
    take = min(count, allowed.numel())
    return allowed[torch.topk(-vals, take).indices]


def BuildSwapInPool(ctx: Context, state: State, block: int) -> torch.Tensor:
    """C_in = current ∪ clean ∪ path ∪ peak ∪ feature ∪ frontier ∪ random candidates (unselected)."""
    selected = state.mask
    unselected = (~selected).nonzero(as_tuple=True)[0]
    pool: list[torch.Tensor] = []
    cur = state.current_score if state.current_score is not None else state.clean_score
    pool.append(_top_restricted(cur, unselected, 4 * block))
    pool.append(_top_restricted(state.clean_score, unselected, 2 * block))
    pool.append(_top_restricted(state.path_ema, unselected, max(1, (3 * block) // 2)))
    pool.append(_top_restricted(state.path_max, unselected, max(1, (3 * block) // 2)))
    if K.FEATURE_GUIDED:
        fscore = _feature_score(ctx, state.x_adv, cur)
        if fscore is not None:
            pool.append(_top_restricted(fscore, unselected, 2 * block))
    if state.previous_gradient is not None and state.gradient is not None:
        frontier = ctx.q * (state.gradient - state.previous_gradient).abs()
        pool.append(_top_restricted(frontier, unselected, 2 * block))
    merged = torch.unique(torch.cat([p for p in pool if p.numel() > 0])) if any(p.numel() for p in pool) \
        else torch.empty(0, dtype=torch.long, device=ctx.clean_u8.device)
    # random exploration over the still-unselected, not-already-pooled coords
    remaining = unselected[~torch.isin(unselected, merged)]
    if remaining.numel() > 0:
        rcount = min(block, remaining.numel())
        rnd = remaining[torch.randperm(remaining.numel(), device=remaining.device)[:rcount]]
        merged = torch.unique(torch.cat([merged, rnd]))
    return merged


def BuildSwapOutPool(ctx: Context, state: State, out_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Removable selected coords: lowest retention = -g·q·sign (small/negative => removable)."""
    selected = state.mask.nonzero(as_tuple=True)[0]
    retention = torch.full_like(state.a, float("inf"))
    if state.gradient is not None and selected.numel() > 0:
        retention[selected] = -state.gradient[selected] * ctx.q * state.sign[selected]
    out_pool = _bottom_restricted(retention, selected, out_size)
    return out_pool, retention


def ExactBlockSwap(ctx: Context, state: State, block: int) -> tuple[State, bool]:
    """Replace weak selected coords with promising unselected ones; accept only exact improvements."""
    swap_in = BuildSwapInPool(ctx, state, block)
    swap_out, retention = BuildSwapOutPool(ctx, state, 4 * block)
    cur = state.current_score if state.current_score is not None else state.clean_score
    proposals: list[tuple[torch.Tensor, torch.Tensor]] = []

    remove_set = _bottom_restricted(retention, swap_out, block)
    proposals.append((remove_set, _top_restricted(cur, swap_in, block)))            # C1 current-gradient
    proposals.append((remove_set, _top_restricted(state.path_ema, swap_in, block)))  # C2 path-average
    proposals.append((remove_set, _top_restricted(state.path_max, swap_in, block)))  # C3 historical-max

    # C4 randomized top-band proposals.
    while len(proposals) < K.PROPOSAL_COUNT and swap_in.numel() > 0 and swap_out.numel() > 0:
        rin = swap_in[torch.randperm(swap_in.numel(), device=swap_in.device)[:min(block, swap_in.numel())]]
        rout = swap_out[torch.randperm(swap_out.numel(), device=swap_out.device)[:min(block, swap_out.numel())]]
        proposals.append((rout, rin))

    # C5 build exact candidate states.
    masks, signs = [], []
    for remove_set, add_set in proposals:
        m = state.mask.clone()
        s = state.sign.clone()
        m[remove_set] = False
        if add_set.numel() > 0 and state.gradient is not None:
            m[add_set] = True
            s[add_set] = _best_direction(ctx, state.gradient)[add_set]
        masks.append(m)
        signs.append(s)

    # C6 batch exact forward.
    res = _eval_states(ctx, masks, signs)
    if not res:
        return state, False
    margins = [r["margin"] for r in res]
    best = int(torch.tensor(margins).argmin().item())

    # C7 accept only a strict improvement.
    if margins[best] < state.margin:
        state.mask = masks[best]
        state.sign = signs[best]
        state.margin = margins[best]
        state.x_adv = res[best]["cand"]
        # synchronize the continuous logits with the new hard support
        threshold = torch.kthvalue(state.a, max(1, state.a.numel() - int(state.mask.sum().item()) + 1)).values \
            if state.mask.any() else state.a.median()
        boost = 1e-3
        state.a[masks[best] & ~state.previous_mask if state.previous_mask is not None else masks[best]] = threshold + boost
        logger.info(f"[phaseC] block-swap accepted: margin={state.margin:.4f} block={block} "
                    f"proposal={best}/{len(proposals)}")
        return state, True
    return state, False


# ==========================================================================================
# Phase D — PartialRestart (preserve the strong core, re-seed only the weakest fraction).
# ==========================================================================================
def ShouldRestart(margin_history: list[float], turnover_history: list[float]) -> bool:
    if len(margin_history) < K.RESTART_PATIENCE:
        return False
    recent_m = margin_history[-K.RESTART_PATIENCE:]
    recent_t = turnover_history[-K.RESTART_PATIENCE:]
    improvement = recent_m[0] - min(recent_m)
    avg_turnover = sum(recent_t) / max(1, len(recent_t))
    return improvement < K.RESTART_MIN_IMPROVE and avg_turnover < K.RESTART_TURNOVER_THRESH


def PartialRestart(ctx: Context, state: State, K_cur: int) -> State:
    """Replace the weakest restart_fraction of the support with reservoir candidates (best of several)."""
    restart_count = max(1, round(K.RESTART_FRACTION * K_cur))
    selected = state.mask.nonzero(as_tuple=True)[0]
    retention = torch.full_like(state.a, float("inf"))
    if state.gradient is not None and selected.numel() > 0:
        retention[selected] = -state.gradient[selected] * ctx.q * state.sign[selected]
    weak = _bottom_restricted(retention, selected, restart_count)
    core = selected[~torch.isin(selected, weak)]

    reservoir = BuildSwapInPool(ctx, state, restart_count)
    cur = state.current_score if state.current_score is not None else state.clean_score
    masks, signs = [], []
    for p in range(K.RESTART_PROPOSALS):
        if p == 0:
            repl = _top_restricted(cur, reservoir, restart_count)
        elif p == 1:
            repl = _top_restricted(state.path_ema, reservoir, restart_count)
        elif p == 2:
            repl = _top_restricted(state.path_max, reservoir, restart_count)
        else:
            take = min(restart_count, reservoir.numel())
            repl = reservoir[torch.randperm(reservoir.numel(), device=reservoir.device)[:take]] \
                if reservoir.numel() > 0 else reservoir
        m = torch.zeros_like(state.mask)
        m[core] = True
        if repl.numel() > 0:
            m[repl] = True
        s = state.sign.clone()
        if repl.numel() > 0 and state.gradient is not None:
            s[repl] = _best_direction(ctx, state.gradient)[repl]
        masks.append(m)
        signs.append(s)

    res = _eval_states(ctx, masks, signs)
    if res:
        margins = [r["margin"] for r in res]
        best = int(torch.tensor(margins).argmin().item())
        state.mask = masks[best]
        state.sign = signs[best]
        state.margin = margins[best]
        state.x_adv = res[best]["cand"]
        logger.info(f"[phaseD] partial restart: replaced {restart_count}/{K_cur} margin={state.margin:.4f}")
    return state


# ==========================================================================================
# Fixed-K optimizer (Phases B + C + D). IMPLEMENTED but only run when K.RUN_OPTIM is set.
# ==========================================================================================
def OptimizeFixedK(ctx: Context, initial_state: State, K_cur: int, max_iterations: int) -> tuple[State, bool]:
    state = copy.deepcopy(initial_state)
    best_state = copy.deepcopy(initial_state)
    margin_history: list[float] = []
    turnover_history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        if _oob(ctx):
            break
        state = DynamicMaskOptimizationStep(ctx, state, K_cur)
        margin_history.append(state.margin)
        turnover_history.append(state.turnover)
        if state.margin < best_state.margin:
            best_state = copy.deepcopy(state)
        if best_state.margin < 0:
            logger.info(f"[optimK] flip at K={K_cur} iter={iteration} margin={best_state.margin:.4f}")
            return best_state, True

        if iteration % K.SWAP_INTERVAL == 0 and not _oob(ctx):
            block = _scheduled_block_size(iteration, K_cur, max_iterations)
            state, _ = ExactBlockSwap(ctx, state, block)
            if state.margin < best_state.margin:
                best_state = copy.deepcopy(state)
            if best_state.margin < 0:
                return best_state, True

        if ShouldRestart(margin_history, turnover_history):
            state = PartialRestart(ctx, state, K_cur)
            margin_history.clear()
            turnover_history.clear()
            if state.margin < best_state.margin:
                best_state = copy.deepcopy(state)
            if best_state.margin < 0:
                return best_state, True

    return best_state, best_state.margin < 0


def _scheduled_block_size(iteration: int, K_cur: int, max_iterations: int) -> int:
    """Coords swapped per round. Flat BLOCK_FRAC·K (floored at BLOCK_MIN) by default, so the step does
    NOT shrink while we are still trying to flip. BLOCK_ANNEAL restores the old iteration taper, which
    only makes sense for sparsifying after a flip already exists."""
    if K.BLOCK_ANNEAL:
        frac = iteration / max(1, max_iterations)
        f = K.BLOCK_FRAC if frac <= 0.3 else (K.BLOCK_FRAC * 0.4 if frac <= 0.7 else K.BLOCK_FRAC * 0.1)
    else:
        f = K.BLOCK_FRAC
    block = max(K.BLOCK_MIN, round(f * K_cur))
    return max(1, min(block, K_cur))


# ==========================================================================================
# Phase E — ReduceCardinality (warm-start smaller supports + reoptimize).
# ==========================================================================================
def WarmStartSmallerK(ctx: Context, successful_state: State, K_new: int) -> State:
    selected = successful_state.mask.nonzero(as_tuple=True)[0]
    _, gradient = _grad_at(ctx, successful_state.x_adv)
    retention = torch.full_like(successful_state.a, float("-inf"))
    retention[selected] = -gradient[selected] * ctx.q * successful_state.sign[selected]
    retained = _top_restricted(retention, selected, K_new)
    new_state = copy.deepcopy(successful_state)
    new_state.mask = torch.zeros_like(successful_state.mask)
    new_state.mask[retained] = True
    new_state.gradient = gradient
    margin, x_adv, _ = _evaluate_state(ctx, new_state.mask, new_state.sign)
    new_state.margin = margin
    new_state.x_adv = x_adv
    logger.info(f"[phaseE] warm-start K_new={K_new} retained={retained.numel()} margin={margin:.4f}")
    return new_state


def ReduceCardinality(ctx: Context, successful_state: State, K_start: int, minimum_K: int) -> tuple[State, int]:
    best_success = copy.deepcopy(successful_state)
    K_current = K_start
    reduction_fraction = K.REDUCTION_FRACTION
    while K_current > minimum_K and not _oob(ctx):
        reduction = max(1, round(reduction_fraction * K_current))
        K_new = max(minimum_K, K_current - reduction)
        warm = WarmStartSmallerK(ctx, best_success, K_new)
        optimized, success = OptimizeFixedK(ctx, warm, K_new, K.ITERATIONS_PER_K)
        if success:
            best_success = optimized
            K_current = K_new
            logger.info(f"[phaseE] reduced to K={K_current} margin={best_success.margin:.4f}")
        else:
            reduction_fraction *= 0.5
            logger.info(f"[phaseE] reduction failed at K_new={K_new}; halving step -> {reduction_fraction:.4f}")
            if reduction_fraction < 1e-3:
                break
    return best_success, K_current


# ==========================================================================================
# Orchestrator — DynamicSparseFixedQAttack.
# ==========================================================================================
def search(ctx: Context) -> None:
    """Run Phase A, then (if enabled) the optimizer. With PERTURB_RETURN_FIRST_FLIP (default on) the
    search unwinds the instant a returnable flip is banked — the _FirstFlipFound raised from _eval is
    caught here and the Bank already holds the flip. Phases B-E are gated behind K.RUN_OPTIM."""
    n = ctx.clean_u8.numel()
    K_init = max(1, int(round(K.K_INIT_FRAC * n)))
    K_min = max(1, int(round(K.K_MIN_FRAC * n)))

    targets = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.TOPM)
    logger.info(f"[search] N={n} K_init={K_init} ({100 * K.K_INIT_FRAC:.1f}%) K_min={K_min} "
                f"targets={targets} run_optim={K.RUN_OPTIM} ignore_timeout={K.IGNORE_TIMEOUT} "
                f"return_first_flip={K.RETURN_FIRST_FLIP}")

    try:
        # ---- Phase A: initialization (multi-source seeding, including feature guidance) ----
        state, _ = InitializeAttack(ctx, targets, K_init)

        inc = _incumbent(ctx)
        if inc is not None:
            logger.info(f"[search] Phase A found a returnable flip: channels={inc['nz']} "
                        f"margin={inc['margin']:.4f}")

        # ---- Phases B-E: IMPLEMENTED but NOT RUN unless explicitly enabled ----
        if K.RUN_OPTIM:
            successful_state, success = OptimizeFixedK(ctx, state, K_init, K.MAX_ITERATIONS)
            logger.info(f"[search] OptimizeFixedK(K={K_init}) success={success} "
                        f"margin={successful_state.margin:.4f}")
            if success:
                final_state, final_K = ReduceCardinality(ctx, successful_state, K_init, K_min)
                logger.info(f"[search] ReduceCardinality -> K={final_K} margin={final_state.margin:.4f}")
        else:
            logger.info("[search] optimizer disabled (PERTURB_RUN_OPTIM=0): returning Phase-A incumbent")
    except _FirstFlipFound:
        inc = _incumbent(ctx)
        if inc is not None:
            logger.info(f"[search] RETURN_FIRST_FLIP: returning first flip immediately "
                        f"(channels={inc['nz']} margin={inc['margin']:.4f})")
        else:
            logger.info("[search] RETURN_FIRST_FLIP: returning first flip immediately")


# ==========================================================================================
# Public entry point — same signature as the legacy miner's perturb().
# ==========================================================================================
def perturb(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
    timeout_seconds: float = 15.0,
    reserve_seconds: float | None = None,
    start_time: float | None = None,
    steps: int | None = None,  # legacy, unused
) -> torch.Tensor:
    """Find a sparse envelope-safe ±k_min/255 flip via the Dynamic Sparse Fixed-q framework (Phase A
    seeding; Phases B-E implemented but gated off by default). Returns the clean image if none found."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(K.MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))  # fixed unit step (typically 1)
    q = k_min * K.Q

    # Byte-space: snap clean to its uint8 grid; every edit is an integer BYTE step on this.
    clean_u8 = torch.round(clean.view(-1) * 255.0)
    envelope = K.TF32_ENVELOPE and device.type == "cuda"

    use_dynamic = K.DYNAMIC_KAPPA and envelope
    calib = get_calibrator(env_fingerprint(model, clean.shape)) if use_dynamic else None
    if calib is not None:
        kappa = calib.global_kappa()
    else:
        kappa = K.KAPPA_RESID if envelope else K.MARGIN_BUFFER

    # One gradient evaluation up front: clean margin m0 + boundary gradient g0, and a t_step gate.
    g_t0 = time.time()
    m0, g0 = margin_and_grad(model, clean, target_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = max(1e-4, time.time() - g_t0)

    if reserve_seconds is None:
        reserve_seconds = K.RESERVE_SECONDS
    reserve_seconds = float(reserve_seconds) + K.RESERVE_FWD_MULT * t_step
    # Timeouts ignored per request: push the deadline far out so neither the budget guard nor
    # batch_eval's deadline-aware chunking stops the search early.
    if K.IGNORE_TIMEOUT:
        deadline = t_start + 1e9
        logger.info("[perturb] IGNORE_TIMEOUT on: deadline disabled")
    else:
        deadline = t_start + max(0.05, float(timeout_seconds) - reserve_seconds)

    # Read the LIVE ctx.deadline so the post-flip arming (which lowers ctx.deadline) takes effect for
    # both the loop budget guard and batch_eval's deadline-aware chunking.
    def time_left() -> float:
        return ctx.deadline - time.time()

    ctx = Context(
        model=model, device=device, clean=clean, clean_u8=clean_u8, shape=clean.shape,
        target_index=target_index, k_min=k_min, q=q, floor=floor, cap=cap, kappa=kappa,
        skip_roundtrip=K.SKIP_ROUNDTRIP, tf32_on=K.TF32_ON, envelope=envelope,
        allow_unsafe=K.ALLOW_UNSAFE_FLIP, deadline=deadline, t_step=t_step, time_left=time_left,
        bank=Bank(), m0=m0, g0=g0.view(-1), dynamic_kappa=use_dynamic,
        optim_seconds=(K.OPTIM_SECONDS if K.RUN_OPTIM else 0.0),
    )

    logger.info(f"[perturb] start: m0={m0:.4f} k_min={k_min} q={q:.6f} kappa={kappa:.4f} "
                f"tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} "
                f"feature_guided={K.FEATURE_GUIDED} "
                f"{'optim_budget=%.1fs' % K.OPTIM_SECONDS if K.RUN_OPTIM else 'optim=off'}")

    search(ctx)

    chosen = ctx.bank.result(ctx.allow_unsafe)

    kappa_n = calib.n_samples if calib is not None else 0
    if calib is not None and chosen is not None:
        try:
            m_exact = exact_worst_margin(ctx, chosen["cand"])
            calib.update(chosen["margin"], m_exact)
            calib.save()
        except Exception as err:
            logger.debug(f"[kappa] calibration update skipped: {err}")

    if chosen is None:
        logger.info(
            f"[perturb] no {'' if ctx.allow_unsafe else 'safe '}flip -> clean "
            f"(m0={m0:.4f} elapsed={time.time() - t_start:.3f}s has_flip={ctx.bank.has_flip} "
            f"tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} "
            f"kappa={kappa:.4f}{f'~n{kappa_n}' if use_dynamic else ''})"
        )
        return clean.detach().clamp(0.0, 1.0)

    pct = 100.0 * chosen["nz"] / max(1, clean_u8.numel())
    logger.info(
        f"[perturb] flip channels={chosen['nz']} ({pct:.2f}%) margin={chosen['margin']:.4f} "
        f"rmse={chosen['rmse']:.6f} linf={chosen['linf']:.6f} elapsed={time.time() - t_start:.3f}s "
        f"m0={m0:.4f} tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} "
        f"kappa={kappa:.4f}{f'~n{kappa_n}' if use_dynamic else ''} safe={chosen is ctx.bank.best_safe}"
    )
    return chosen["cand"].detach().clamp(0.0, 1.0)
