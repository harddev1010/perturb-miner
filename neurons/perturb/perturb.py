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
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from . import constants as K
from .calibration import env_fingerprint, get_calibrator
from .utils import (
    Bank,
    Context,
    apply_delta_bytes,
    batch_eval,
    count_bwd,
    estimate_k,
    exact_worst_margin,
    fwd_logits,
    logits_of,
    loss_grad,
    margin_and_grad,
    movable,
    out_of_budget,
    passes,
    reset_passes,
    top_wrong_classes,
    validator_score,
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
        # Reclaim the q2 reserve: a q=1 flip exists, so the fallback won't run — arm against the FULL
        # hard_deadline (not the fb-reduced ctx.deadline), so post-flip optimization uses the whole budget.
        limit = ctx.hard_deadline if ctx.hard_deadline > 0.0 else ctx.deadline
        ctx.deadline = min(limit, ctx.first_flip_time + ctx.optim_seconds)
        logger.info(f"[optim] first flip banked -> arming {ctx.optim_seconds:.1f}s optim budget "
                    f"(deadline in {ctx.deadline - time.time():.1f}s)")


# ==========================================================================================
# Adaptive hyperparameter controller — start at the env values, escalate on stall, relax on progress.
# Hot-path knobs are read through _p(ctx, NAME), which returns the live (tuned) value when a tuner is
# attached and enabled, else the static K.NAME default.
# ==========================================================================================
_TUNABLE = ("ETA_MASK", "TEMPERATURE", "BLOCK_FRAC", "BLOCK_MIN",
            "PROPOSAL_COUNT", "SWAP_INTERVAL", "RESTART_FRACTION")


def _p(ctx: Context, name: str):
    """Live value of a tunable knob (tuner override if present/enabled), else the static K default."""
    t = getattr(ctx, "tuner", None)
    if t is not None and t.enabled and name in t.params:
        return t.params[name]
    return getattr(K, name)


class AdaptiveTuner:
    """Watches the best margin over a sliding window and rescales the exploration knobs.

    level L maps every knob to base · TUNE_FACTOR**L (with per-knob caps; SWAP_INTERVAL divides so it
    gets MORE frequent). On a stalled window (best margin barely moved) level += 1; on a strongly
    improving window level -= 1. Escalation makes the steps bigger and de-saturates Phase B (higher
    temperature), exactly the levers needed when block-swap plateaus on a hard image."""

    def __init__(self) -> None:
        self.enabled = K.ADAPTIVE_TUNE
        self.base = {k: getattr(K, k) for k in _TUNABLE}
        self.params = dict(self.base)
        self.level = 0
        self.iters = 0
        self.best = float("inf")
        self.window_start_best = float("inf")

    def _apply(self) -> None:
        f = K.TUNE_FACTOR ** self.level
        self.params["ETA_MASK"] = self.base["ETA_MASK"] * f
        self.params["TEMPERATURE"] = min(self.base["TEMPERATURE"] * f, K.TUNE_TEMPERATURE_CAP)
        self.params["BLOCK_FRAC"] = min(self.base["BLOCK_FRAC"] * f, K.TUNE_BLOCK_FRAC_CAP)
        self.params["BLOCK_MIN"] = int(round(self.base["BLOCK_MIN"] * f))
        self.params["PROPOSAL_COUNT"] = min(int(round(self.base["PROPOSAL_COUNT"] * f)), K.TUNE_PROPOSAL_CAP)
        self.params["SWAP_INTERVAL"] = max(1, int(round(self.base["SWAP_INTERVAL"] / f)))
        self.params["RESTART_FRACTION"] = min(self.base["RESTART_FRACTION"] * f, K.TUNE_RESTART_FRAC_CAP)

    def _log(self, why: str) -> None:
        p = self.params
        logger.debug(f"[tune] {why} level={self.level} eta={p['ETA_MASK']:.2f} temp={p['TEMPERATURE']:.2f} "
                     f"block_frac={p['BLOCK_FRAC']:.3f} block_min={p['BLOCK_MIN']} "
                     f"proposals={p['PROPOSAL_COUNT']} swap_int={p['SWAP_INTERVAL']} "
                     f"restart_frac={p['RESTART_FRACTION']:.3f}")

    def observe(self, best_margin: float) -> None:
        """Call once per optim iteration with the current global-best margin."""
        if not self.enabled:
            return
        self.iters += 1
        self.best = min(self.best, best_margin)
        if self.window_start_best == float("inf"):
            self.window_start_best = self.best
            return
        if self.iters % max(1, K.TUNE_INTERVAL) != 0:
            return
        improvement = self.window_start_best - self.best
        rel = improvement / max(abs(self.window_start_best), 1e-6)
        if improvement < K.TUNE_MIN_IMPROVE and rel < K.TUNE_MIN_REL:
            decision = "escalate" if self.level < K.TUNE_MAX_LEVEL else "stall@max"
            if self.level < K.TUNE_MAX_LEVEL:
                self.level += 1
                self._apply()
        elif rel > K.TUNE_GOOD_REL and self.level > 0:
            self.level -= 1
            self._apply()
            decision = "relax"
        else:
            decision = "hold"
        # Per-window heartbeat: always log so the controller's decision is visible even when it holds.
        self._log(f"window @iter={self.iters} best={self.best:.4f} Δ={improvement:.4f} "
                  f"rel={rel:.3f} -> {decision}")
        self.window_start_best = self.best


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


def _batched_grads(ctx: Context, images: list[torch.Tensor],
                   specs: list[tuple[str, int | None]]) -> tuple[list[float], torch.Tensor]:
    """T1.2: gradients for many independent (image, loss) sources in ONE forward+backward.

    images[i] is an (C,H,W) input; specs[i] is ("hard", None) for the CW margin or ("pair", c) for the
    pairwise margin z_y - z_c. The losses are row-separable, so backward of their sum yields each row's
    own input gradient. Returns (values, grads[B, N]). Replaces ~9 sequential backwards in Phase A."""
    batch = torch.stack([im.detach() for im in images], dim=0).to(ctx.device).requires_grad_(True)
    logits = fwd_logits(ctx.model, batch)  # [B, num_classes]
    y = ctx.target_index
    losses, values = [], []
    for i, (kind, c) in enumerate(specs):
        row = logits[i]
        if kind == "pair":
            v = row[y] - row[c]
        else:  # hard CW margin
            others = row.clone()
            others[y] = float("-inf")
            v = row[y] - others.max()
        losses.append(v)
        values.append(float(v.item()))
    grads = torch.autograd.grad(torch.stack(losses).sum(), batch)[0].detach().view(len(images), -1)
    count_bwd(len(images))
    return values, grads


def _percentile_rank(score: torch.Tensor) -> torch.Tensor:
    """T1.3: map a score vector to per-coordinate percentile rank in [0,1] (higher score => higher rank).
    Scale-free, so beams on different numeric scales (hard/soft/DLR/pairwise) fuse without one dominating."""
    n = score.numel()
    if n <= 1:
        return torch.zeros_like(score)
    ranks = score.argsort().argsort().to(score.dtype)
    return ranks / (n - 1)


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
    # T3.1: the clean-image relevance map is invariant — compute it once and reuse across Phase-A
    # (re-)seeds and restarts. Swap-in pools pass x_adv (not clean), so they still recompute.
    is_clean = x_chw is ctx.clean
    if is_clean and ctx.clean_relevance is not None:
        return ctx.clean_relevance
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
        logits = fwd_logits(model, x.unsqueeze(0))[0]
        others = logits.clone()
        others[ctx.target_index] = float("-inf")
        margin = logits[ctx.target_index] - others.max()
        model.zero_grad(set_to_none=True)
        margin.backward()
        count_bwd(1)
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
    relevance = up.reshape(-1).repeat(int(ctx.shape[0])).to(ctx.clean_u8.dtype)
    if is_clean:
        ctx.clean_relevance = relevance
    return relevance


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
    """The returnable incumbent: highest-score envelope-safe flip, or (if unsafe allowed) best flip."""
    if ctx.bank.best_safe is not None:
        return ctx.bank.best_safe
    return ctx.bank.best_flip if ctx.allow_unsafe else None


def _pad_novelty(ctx: Context, chosen: dict) -> dict:
    """The validator's novelty bonus saturates at NOVELTY_TARGET_PIXELS changed spatial pixels; a flip
    touching fewer forfeits up to the full novelty weight (0.01) for ~0 perturbation gain. The Phase-E
    floor only stops pruning below it — it can't lift a flip that Phase A already landed sparse (common
    on easy images). If the chosen flip is under the floor, add single-byte flips on fresh pixels in the
    margin-DECREASING direction (-sign(g0)), which only deepens the flip so safety holds, then keep the
    result iff the score-ranked Bank prefers it."""
    target = max(1, int(K.NOVELTY_TARGET_PIXELS))
    delta = chosen.get("delta")
    if delta is None or int(chosen.get("pixels", target)) >= target or ctx.g0 is None:
        return chosen
    channels = int(ctx.shape[0])
    hw = ctx.clean_u8.numel() // channels
    delta = delta.clone()
    touched = (delta.reshape(channels, hw).abs() > 0).any(dim=0)   # [hw] pixels already changed
    need = target - int(touched.sum().item())
    if need <= 0:
        return chosen
    gm = ctx.g0.reshape(channels, hw)
    best_chan = gm.abs().argmax(dim=0)                             # strongest channel per fresh pixel
    strength = gm.abs().max(dim=0).values.masked_fill(touched, float("-inf"))
    added = 0
    for p in torch.argsort(strength, descending=True).tolist():
        if added >= need or strength[p].item() == float("-inf"):
            break
        idx = int(best_chan[p].item()) * hw + p
        delta[idx] = (-1.0 if ctx.g0[idx] > 0 else 1.0) * ctx.k_min
        added += 1
    try:
        _eval(ctx, [delta])
    except _FirstFlipFound:
        pass
    return ctx.bank.result(ctx.allow_unsafe) or chosen


def _q2_fallback(ctx: Context, hard_deadline: float) -> dict | None:
    """Strong last resort when the q=1 search banked nothing. Here "q=2" is a MAXIMUM L∞ budget of 2/255
    (k_min=2) — a SPARSE ±2/255 support, not every pixel at 2/255. Two stages inside the reserve:

      1) GUARANTEED backstop: one DENSE feasible descent candidate (every coord stepped -sign(g0)·2/255).
         At double reach this flips almost any image in a SINGLE eval, so we always bank *something*
         (score ~0.6-0.7) rather than returning clean (0) — even on a 1s tail.
      2) If budget remains: the FULL search at q=2 (Phase A + optimizer + score-gated reduction) to find
         a SPARSE, higher-score q=2 flip. The score-ranked Bank keeps the best of the two.

    Reuses the clean gradient (∇margin is step-size independent). Hard-capped at q=2 (never q>=3): a q=2
    flip tops out ~0.77 total, but that dwarfs the 0 (and 300-window drag) of a clean return."""
    fb_deadline = min(hard_deadline, time.time() + K.FALLBACK_Q2_SECONDS)
    if fb_deadline - time.time() <= 2.0 * ctx.t_step:
        logger.info("[q2-fallback] no time left for q=2 retry")
        return None
    ctx2 = replace(
        ctx, k_min=2, q=2 * K.Q, bank=Bank(), first_flip_time=None,
        deadline=fb_deadline, time_left=lambda: fb_deadline - time.time(),
    )
    logger.info(f"[q2-fallback] retry at q=2 budget={fb_deadline - time.time():.2f}s")
    try:
        # Stage 1 — dense feasible descent backstop (one eval, near-guaranteed flip at double reach).
        if ctx2.g0 is not None:
            sign = _best_direction(ctx2, ctx2.g0)
            dense = _make_delta(ctx2, torch.ones_like(ctx2.clean_u8, dtype=torch.bool), sign)
            _eval(ctx2, [dense])
            inc = ctx2.bank.result(ctx2.allow_unsafe)
            if inc is not None:
                logger.info(f"[q2-fallback] dense backstop nz={inc['nz']} margin={inc['margin']:.4f} "
                            f"score={inc['score']:.4f}")
        # Stage 2 — full q=2 search for a sparse, higher-score flip (Bank keeps the best).
        if not _oob(ctx2):
            search(ctx2)
    except _FirstFlipFound:
        pass
    except Exception as err:  # a fallback must never take the whole call down
        logger.warning(f"[q2-fallback] failed: {err}")
    chosen = ctx2.bank.result(ctx2.allow_unsafe)
    if chosen is not None:
        chosen = _pad_novelty(ctx2, chosen)
        logger.info(f"[q2-fallback] flip channels={chosen['nz']} pixels={chosen['pixels']} "
                    f"margin={chosen['margin']:.4f} score={chosen['score']:.4f}")
    else:
        logger.info("[q2-fallback] no q=2 flip either -> clean")
    return chosen


# ==========================================================================================
# Phase A — InitializeAttack (multi-source seeding, including feature guidance).
# ==========================================================================================
def _top_indices(score: torch.Tensor, count: int) -> torch.Tensor:
    count = max(0, min(int(count), score.numel()))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    return torch.topk(score, count).indices


def _rand_mask(ctx: Context, cnt: int) -> torch.Tensor:
    """Boolean mask over `cnt` random coordinates (for a fresh off-clean random-start basin)."""
    n = ctx.clean_u8.numel()
    idx = torch.randperm(n, device=ctx.clean_u8.device)[:max(1, min(cnt, n))]
    m = torch.zeros(n, dtype=torch.bool, device=ctx.clean_u8.device)
    m[idx] = True
    return m


def InitializeAttack(ctx: Context, targets: list[int], K_init: int) -> tuple[State, list[dict]]:
    """Phase A. Build a fixed-K support from clean + targeted + random-start + feature-guided
    candidates, initialize the continuous logits/signs, and (so a flip can be found without the
    optimizer) verify each source's support ladder into the Bank. Returns (state, seed_results)."""
    t0 = time.time()
    n = ctx.clean_u8.numel()
    seed_deltas: list[torch.Tensor] = []
    beams: list[torch.Tensor] = []        # per-source score vectors, fused by percentile rank (T1.3)
    candidate_lists: list[torch.Tensor] = []
    tgt = targets[:K.TARGET_COUNT]

    # ---- A1/A2/A3 gradients: one batched forward+backward for clean + targets + random starts (T1.2) ----
    rand_images = [
        apply_delta_bytes(
            ctx.clean_u8,
            _make_delta(
                ctx,
                _rand_mask(ctx, max(1, int(round(K.RANDOM_START_FRAC * n)))),
                _feasible_sign_from_value(ctx, torch.randn(n, device=ctx.clean_u8.device)),
            ),
            ctx.shape,
        )
        for _ in range(K.RANDOM_START_COUNT)
    ]
    if K.BATCHED_GRADS:
        images = [ctx.clean] + [ctx.clean] * len(tgt) + rand_images
        specs = [("hard", None)] + [("pair", c) for c in tgt] + [("hard", None)] * len(rand_images)
        values, grads = _batched_grads(ctx, images, specs)
        clean_margin, clean_gradient = values[0], grads[0]
        tgt_grads = [grads[1 + j] for j in range(len(tgt))]
        tgt_values = [values[1 + j] for j in range(len(tgt))]
        rand_grads = [grads[1 + len(tgt) + r] for r in range(len(rand_images))]
    else:
        clean_margin, clean_gradient = _grad_at(ctx, ctx.clean)
        tgt_grads, tgt_values = [], []
        for c in tgt:
            v, latent = _field(ctx, ctx.clean, f"pair:{c}", targets)
            tgt_grads.append(-latent); tgt_values.append(v)
        rand_grads = [_grad_at(ctx, xr)[1] for xr in rand_images]

    aggregate_gradient = clean_gradient.clone()

    # A1. Untargeted clean beam.
    clean_score, clean_sign = _candidate_scores(ctx, clean_gradient)
    beams.append(clean_score)
    candidate_lists.append(_top_indices(clean_score, round(0.5 * K_init)))
    seed_deltas += _support_ladder(ctx, clean_sign * clean_score, clean_margin)
    logger.debug(f"[phaseA] A1 clean gradient: margin={clean_margin:.4f} "
                 f"legal={(clean_score > 0).sum().item()} top={round(0.5 * K_init)}")

    # A2. Target-specific beams (pairwise margins kept separate for generation; T2.2).
    per_target = max(1, round(K_init / (4 * max(1, K.TARGET_COUNT))))
    for c, tgrad, tval in zip(tgt, tgt_grads, tgt_values):
        tscore, tsign = _candidate_scores(ctx, tgrad)
        beams.append(tscore)
        candidate_lists.append(_top_indices(tscore, per_target))
        aggregate_gradient += tgrad
        seed_deltas += _support_ladder(ctx, tsign * tscore, tval)
        logger.debug(f"[phaseA] A2 target={c}: value={tval:.4f} kc={_kc_of(ctx, tsign * tscore, tval)} top={per_target}")

    # A3. Random-start beams.
    per_random = max(1, round(K_init / (4 * max(1, K.RANDOM_START_COUNT))))
    for r, rgrad in enumerate(rand_grads):
        rscore, rsign = _candidate_scores(ctx, rgrad)
        beams.append(rscore)
        candidate_lists.append(_top_indices(rscore, per_random))
        seed_deltas += _support_ladder(ctx, rsign * rscore, clean_margin)
        logger.debug(f"[phaseA] A3 random-start {r + 1}/{K.RANDOM_START_COUNT}: top={per_random}")

    # A4. Feature-guided beam (Q1); clean map cached (T3.1).
    if K.FEATURE_GUIDED:
        fscore = _feature_score(ctx, ctx.clean, clean_score)
        if fscore is not None:
            fquota = max(1, round(K.FEATURE_QUOTA_FRAC * K_init))
            beams.append(fscore)
            candidate_lists.append(_top_indices(fscore, fquota))
            seed_deltas += _support_ladder(ctx, clean_sign * fscore, clean_margin)
            logger.debug(f"[phaseA] A4 feature-guided ({'gate' if K.FEATURE_GATE else 'pure'}): "
                         f"top={fquota} nonzero={(fscore > 0).sum().item()}")
        else:
            logger.debug("[phaseA] A4 feature-guided: no hookable conv layer -> skipped")
    else:
        logger.debug("[phaseA] A4 feature-guided: disabled")

    # ---- Normalized-rank beam fusion (T1.3): fuse beams by percentile rank, not by clean_score ----
    if K.RANK_FUSION and beams:
        fused_score = torch.stack([_percentile_rank(b) for b in beams], dim=0).sum(dim=0)
    else:
        fused_score = clean_score
    agg_sign = _best_direction(ctx, aggregate_gradient)  # toward-flip signs for the grow ladder

    # ---- Grow-until-first-flip ladder (T1.1): nested geometric supports from the FUSED ranking, graded
    # in the same batched pass; the Bank keeps the sparsest flipping rung so we land sparse directly. ----
    if K.GROW_LADDER:
        fused_order = torch.argsort(fused_score, descending=True)
        rung_ks = sorted({max(1, min(int(round(f * n)), n)) for f in K.GROW_RUNGS})
        for kr in rung_ks:
            idx = fused_order[:kr]
            m = torch.zeros(n, dtype=torch.bool, device=ctx.clean_u8.device)
            m[idx] = True
            seed_deltas.append(_make_delta(ctx, m, agg_sign))
            for _ in range(K.GROW_VARIANTS):
                seed_deltas.append(_project_sampled(ctx, agg_sign * fused_score, kr))
        logger.debug(f"[phaseA] grow ladder: rungs={rung_ks} (fused-ranked, batched)")

    # A4'/merge. Union of per-beam quotas -> K distinct coords, trimmed/filled by the FUSED rank (T1.3).
    union = torch.cat([c for c in candidate_lists if c.numel() > 0]) if candidate_lists else \
        torch.empty(0, dtype=torch.long, device=ctx.clean_u8.device)
    union = torch.unique(union)
    if union.numel() >= K_init:
        order = torch.argsort(fused_score[union], descending=True)
        initial_support = union[order][:K_init]
    else:
        fill = _top_indices(fused_score, n)
        merged = torch.cat([union, fill])
        seen = torch.zeros(n, dtype=torch.bool, device=ctx.clean_u8.device)
        keep: list[int] = []
        for i in merged.tolist():
            if not seen[i]:
                seen[i] = True
                keep.append(i)
            if len(keep) >= K_init:
                break
        initial_support = torch.tensor(keep, dtype=torch.long, device=ctx.clean_u8.device)
    logger.debug(f"[phaseA] merged support: union={union.numel()} -> initial_support={initial_support.numel()} "
                 f"(K_init={K_init}) fusion={'on' if K.RANK_FUSION else 'off'}")

    # A5. Initialize mask logits: small noise + a large boost on the seeded support.
    a = K.MASK_NOISE * torch.randn(n, device=ctx.clean_u8.device)
    a[initial_support] += K.BIG_INIT

    # A6. Initialize sign scores v = -aggregate_gradient -> feasible signs; hard mask = TopK(a).
    v = -aggregate_gradient
    sign = _feasible_sign_from_value(ctx, v)
    mask = _topk_mask(a, K_init)

    path_ema = clean_score.clone()
    path_max = clean_score.clone()

    # Verify ALL seed candidates in ONE batched pass: the grow-ladder rungs and per-source ladders
    # together with the K_init init-state. The Bank keeps the HIGHEST-SCORE flip (its _better orders by
    # the full validator score), and with RETURN_FIRST_FLIP the whole batch is graded before the sentinel
    # unwinds — so we return the best-scoring flipping rung, never the dense init-state just because it
    # was graded first.
    init_delta = _make_delta(ctx, mask, sign)
    seed_deltas.append(init_delta)
    seed_results = _eval(ctx, seed_deltas)
    init_res = seed_results[-1] if len(seed_results) == len(seed_deltas) else None
    if init_res is not None:
        margin, x_adv = init_res["margin"], init_res["cand"]
    else:
        margin, x_adv, _ = _evaluate_state(ctx, mask, sign)
    flips = sum(1 for r in seed_results if r.get("quality"))
    best_seed = min((r["margin"] for r in seed_results), default=float("inf"))
    logger.debug(f"[phaseA] seeded+verified {len(seed_deltas)} candidates: quality_flips={flips} "
                 f"best_margin={best_seed:.4f} init_state_margin={margin:.4f}")
    k_flip = ctx.bank.best_flip["nz"] if ctx.bank.best_flip is not None else None
    logger.info(f"[phaseA] done: tried={len(seed_deltas)} candidates found_flip={flips > 0}"
                + (f" K_flip={k_flip}" if flips > 0 else "")
                + f" spent={time.time() - t0:.3f}s")

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

    # B5. Dense mask-logit update (straight-through sigmoid). eta/temperature are live (adaptive).
    temperature = max(_p(ctx, "TEMPERATURE"), 1e-6)
    mask_gradient = ctx.q * proposed_sign * gradient
    soft_mask = torch.sigmoid(state.a / temperature)
    derivative = soft_mask * (1 - soft_mask) / temperature
    state.a = state.a - _p(ctx, "ETA_MASK") * mask_gradient * derivative

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
    while len(proposals) < _p(ctx, "PROPOSAL_COUNT") and swap_in.numel() > 0 and swap_out.numel() > 0:
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
        n = state.mask.numel()
        (logger.info if ctx.log_swaps else logger.debug)(
            f"[phaseC] swap: cands={len(proposals)} block={block} "
            f"({100.0 * block / max(1, n):.2f}% of N) margin={state.margin:.4f}")
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
    restart_count = max(1, round(_p(ctx, "RESTART_FRACTION") * K_cur))
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
        logger.debug(f"[phaseD] partial restart: replaced {restart_count}/{K_cur} margin={state.margin:.4f}")
    return state


# ==========================================================================================
# Fixed-K optimizer (Phases B + C + D). IMPLEMENTED but only run when K.RUN_OPTIM is set.
# ==========================================================================================
def _reached(margin: float, deepen_target: float | None) -> bool:
    """Stop this fixed-K optimize call? Requires a flip (margin<0) and, when deepen_target is set, a CW
    margin at least that deep (<= deepen_target, a negative number). deepen_target=None => stop at the
    FIRST flip (used by the FIND stage and strict-mode reduce rungs — RMSE is handled separately, so
    there is no reason to keep grinding margin at a large K)."""
    if margin >= 0.0:
        return False
    return deepen_target is None or margin <= deepen_target


def OptimizeFixedK(ctx: Context, initial_state: State, K_cur: int, max_iterations: int,
                   deepen_target: float | None = None, return_state: bool = False):
    """Fixed-K optimizer (Phases B/C/D). Returns (best_state, success). With return_state=True also returns
    the LIVE end-of-run state as a third element — used by the chunked QuickAnchor to CONTINUE the optimizer
    trajectory across chunks (best_state is the deepest snapshot; resuming from it would discard the
    temporarily-uphill exploration that precedes a delayed nonlinear margin drop)."""
    state = copy.deepcopy(initial_state)
    best_state = copy.deepcopy(initial_state)
    margin_history: list[float] = []
    turnover_history: list[float] = []
    stall_win: list[float] = []   # C2: sliding best-margin window for the deepen early-stop

    ran = 0

    def _out(ok: bool):
        ctx.last_iters = ran
        return (best_state, ok, state) if return_state else (best_state, ok)

    for iteration in range(1, max_iterations + 1):
        ran = iteration
        if _oob(ctx):
            break
        state = DynamicMaskOptimizationStep(ctx, state, K_cur)
        margin_history.append(state.margin)
        turnover_history.append(state.turnover)
        if state.margin < best_state.margin:
            best_state = copy.deepcopy(state)
        # Feed the adaptive controller the global-best margin (escalate on stall, relax on progress).
        if ctx.tuner is not None:
            ctx.tuner.observe(best_state.margin)
        if _reached(best_state.margin, deepen_target):
            logger.debug(f"[optimK] flip at K={K_cur} iter={iteration} margin={best_state.margin:.4f}")
            return _out(True)

        if iteration % max(1, int(_p(ctx, "SWAP_INTERVAL"))) == 0 and not _oob(ctx):
            block = _scheduled_block_size(ctx, iteration, K_cur, max_iterations)
            state, _ = ExactBlockSwap(ctx, state, block)
            if state.margin < best_state.margin:
                best_state = copy.deepcopy(state)
            if _reached(best_state.margin, deepen_target):
                return _out(True)

        if ShouldRestart(margin_history, turnover_history):
            state = PartialRestart(ctx, state, K_cur)
            margin_history.clear()
            turnover_history.clear()
            if state.margin < best_state.margin:
                best_state = copy.deepcopy(state)
            if _reached(best_state.margin, deepen_target):
                return _out(True)

        # C2: diminishing-returns early stop for standalone deepen calls (deepen_target set). Once a
        # good-enough margin (<= -DEEPEN_STALL_FLOOR) is banked AND it has stalled over the window, return
        # so the caller redirects the rest of the budget to the RMSE (cardinality) search. The window is
        # larger than ANCHOR_CHUNK_ITERS, so this never fires inside _quick_anchor's chunks (that path runs
        # its own stall logic). FIND (deepen_target None) is exempt — it already stops at the first flip.
        if (K.DEEPEN_STALL_STOP and deepen_target is not None
                and best_state.margin <= -K.DEEPEN_STALL_FLOOR):
            stall_win.append(best_state.margin)
            if len(stall_win) > max(2, int(K.DEEPEN_STALL_WINDOW)):
                stall_win.pop(0)
            if (len(stall_win) >= max(2, int(K.DEEPEN_STALL_WINDOW))
                    and stall_win[0] - min(stall_win) < K.DEEPEN_STALL_MIN_IMPROVE):
                logger.debug(f"[optimK] deepen stalled at K={K_cur} margin={best_state.margin:.4f} "
                             f"iter={iteration} -> early stop (redirect budget to RMSE)")
                return _out(True)

    return _out(best_state.margin < 0)


def _scheduled_block_size(ctx: Context, iteration: int, K_cur: int, max_iterations: int) -> int:
    """Coords swapped per round. Flat BLOCK_FRAC·K (floored at BLOCK_MIN) by default, so the step does
    NOT shrink while we are still trying to flip. BLOCK_ANNEAL restores the old iteration taper, which
    only makes sense for sparsifying after a flip already exists. BLOCK_FRAC/BLOCK_MIN are live (adaptive)."""
    block_frac = _p(ctx, "BLOCK_FRAC")
    if K.BLOCK_ANNEAL:
        frac = iteration / max(1, max_iterations)
        f = block_frac if frac <= 0.3 else (block_frac * 0.4 if frac <= 0.7 else block_frac * 0.1)
    else:
        f = block_frac
    block = max(_p(ctx, "BLOCK_MIN"), round(f * K_cur))
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
    logger.debug(f"[phaseE] warm-start K_new={K_new} retained={retained.numel()} margin={margin:.4f}")
    return new_state


def _novelty_floor_k(K_start: int, K_min: int) -> int:
    """Smallest K the reducers may reach: max(K_min, min(K_start, NOVELTY_TARGET_PIXELS)). Channels are
    scored with their real changed-PIXEL count (batch_eval), and _pad_novelty lifts any sub-target flip
    back to >=8 pixels at the end — so the search may explore down to the pixel target (8 channels on 8
    distinct pixels already saturate novelty) instead of the old conservative 3x-channel floor (#4)."""
    return max(int(K_min), min(int(K_start), max(1, int(K.NOVELTY_TARGET_PIXELS))))


def _state_score(ctx: Context, state: State) -> float:
    """The validator's FULL score for a state's current image (mirrors batch_eval): 0 if not flipped,
    else perturbation(L∞,RMSE) + margin bonus + novelty bonus. Lets the descent compare (K, margin)
    levels on the real objective instead of on 'does it still flip'."""
    if state.margin >= 0.0:
        return 0.0
    diff = state.x_adv - ctx.clean
    changed = diff.abs() > 0.5 * ctx.q
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    pixels = int(changed.any(dim=0).sum().item())
    return validator_score(linf, rmse, state.margin, pixels, ctx.cap)


def ReduceCardinality(ctx: Context, flip_state: State, K_start: int, minimum_K: int,
                      deepen_per_level: bool) -> tuple[State, int]:
    """Score-gated cardinality descent (the 'margin-vs-RMSE rate' made explicit). Geometrically probe
    smaller supports and accept a smaller K only while the TOTAL score does not fall by more than
    SCORE_TOL (a smaller K at equal score is strictly better). Reduction stops at the score PEAK — the K
    where the margin lost by removing another coord costs more perturbation than it buys — NOT at the
    minimum flipping K. When deepen_per_level, re-deepen the margin toward the CEIL at each probed K
    before scoring it, so the descent sees the deep-margin-at-moderate-K points the old first-flip rungs
    never generated. The score-ranked Bank still keeps the global best regardless."""
    minimum_K = _novelty_floor_k(K_start, minimum_K)
    per_level_iters = max(1, int(K.ITERATIONS_PER_K))
    deepen_target = -K.MARGIN_DEEPEN_TARGET if deepen_per_level else None

    best_state = copy.deepcopy(flip_state)
    if deepen_per_level and not _oob(ctx):
        # Settle the starting level so its score reflects the achievable margin, not the bare flip.
        best_state, _ = OptimizeFixedK(ctx, best_state, max(1, int(best_state.mask.sum().item())),
                                       per_level_iters, deepen_target=deepen_target)
    best_score = _state_score(ctx, best_state)
    K_current = max(1, int(best_state.mask.sum().item()))
    reduction_fraction = K.REDUCTION_FRACTION
    misses = 0

    while K_current > minimum_K and not _oob(ctx):
        reduction = max(1, round(reduction_fraction * K_current))
        K_new = max(minimum_K, K_current - reduction)
        warm = WarmStartSmallerK(ctx, best_state, K_new)
        optimized, success = OptimizeFixedK(ctx, warm, K_new, per_level_iters, deepen_target=deepen_target)
        cand_score = _state_score(ctx, optimized) if success else -1.0
        if success and cand_score >= best_score - K.SCORE_TOL:
            best_state, best_score, K_current = optimized, max(best_score, cand_score), K_new
            misses = 0
            logger.debug(f"[reduce] K={K_current} margin={optimized.margin:.4f} score={cand_score:.4f}")
        else:
            # Smaller K lost score (or would not flip): try a finer step, then accept the score peak.
            reduction_fraction *= 0.5
            misses += 1
            logger.debug(f"[reduce] reject K_new={K_new} (score={cand_score:.4f} vs {best_score:.4f}); "
                         f"finer step -> {reduction_fraction:.4f}")
            if misses >= 2 or reduction_fraction < 1e-3:
                break
    return best_state, K_current


# ==========================================================================================
# INNER support-quality refinement (all strategies). Fixes the coords/signs at the chosen K.
# ==========================================================================================
def SupportRefine(ctx: Context) -> None:
    """Refine the coordinate SET at the chosen K on the current best flip. The outer K-search decides
    HOW MANY coords; this decides WHICH — where the engine was weakest (first-order retention + margin-
    only swaps). Two moves per round, gradient-prefiltered into ONE batched forward:

      * EXACT deletion — batch-test removing the weakest active coords (individually AND a few aggregate
        removals), catching redundancy the first-order retention misranks (nonlinear response).
      * Score swaps — batch weakest-out / strongest-feasible-in one-for-one replacements.

    Every candidate folds through the score-ranked Bank, so acceptance is by FULL validator score for
    free — it can only raise the returned score (or, out of budget, do nothing). Re-linearizes at the new
    best each round and stops when a round yields no score gain."""
    if not K.REFINE_SUPPORT:
        return
    best = ctx.bank.best_safe
    if best is None or best.get("delta") is None:
        return
    d = best["delta"]
    x_adv = best["cand"]
    prev = float(best["score"])
    logger.info(f"[refine] SupportRefine start: rounds<={max(1, int(K.REFINE_ROUNDS))} "
                f"K={best['nz']} score={prev:.4f}")
    t_sr = time.time()
    rounds = 0
    for _ in range(max(1, int(K.REFINE_ROUNDS))):
        if _oob(ctx):
            break
        rounds += 1
        mask = d != 0
        sign = d.sign()
        active = mask.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        _m, g = _grad_at(ctx, x_adv)
        retention = -g[active] * ctx.q * sign[active]                 # small/negative => removable
        weak = active[torch.argsort(retention)][: max(1, int(K.REFINE_DELETION_POOL))]
        cands: list[torch.Tensor] = []
        # (B) exact deletion: individual removals + a few aggregate removals of the weakest coords.
        for i in weak.tolist():
            c = d.clone(); c[i] = 0.0; cands.append(c)
        for frac in (0.5, 0.25, 0.1):
            k = max(1, int(frac * weak.numel()))
            c = d.clone(); c[weak[:k]] = 0.0; cands.append(c)
        # (C) score-accepted swaps: weakest active out, strongest feasible-descent inactive in.
        in_sign = _best_direction(ctx, g)
        in_gain = (-g * ctx.q * in_sign).clamp(min=0.0)
        in_gain[mask] = float("-inf")                                 # inactive coords only
        n_in = min(int(K.REFINE_SWAP_POOL), int((~mask).sum().item()))
        if n_in > 0:
            strong_in = torch.topk(in_gain, n_in).indices
            n_pairs = min(int(weak.numel()), int(strong_in.numel()), int(K.REFINE_SWAP_PROPOSALS))
            for t in range(n_pairs):
                i = int(weak[t]); j = int(strong_in[t])
                c = d.clone(); c[i] = 0.0; c[j] = float(in_sign[j]) * float(ctx.k_min); cands.append(c)
        _eval(ctx, cands)                                             # batched; Bank accepts by full score
        cur = ctx.bank.best_safe
        if cur is None or float(cur["score"]) <= prev + 1e-6:
            break                                                    # no score gain this round -> stop
        prev = float(cur["score"]); d = cur["delta"]; x_adv = cur["cand"]
    fin = ctx.bank.best_safe
    if fin is not None:
        logger.info(f"[refine] SupportRefine done: rounds_ran={rounds} spent={time.time() - t_sr:.3f}s "
                    f"best=({fin['nz']}ch pixels={fin['pixels']} margin={fin['margin']:.4f} score={fin['score']:.4f})")


# ==========================================================================================
# Post-flip strategies (env PERTURB_POSTFLIP_STRATEGY). Both leave the best (K, margin) in the Bank.
# ==========================================================================================
def PostFlipStrict(ctx: Context, flip_state: State, K_start: int, K_min: int) -> None:
    """Reduce first with first-flip rungs (cheap, shallow), score-gated, then one deepen pass at the
    settled K. Simpler/faster than coupled; may settle at a slightly smaller K with a shallower margin."""
    final_state, final_K = ReduceCardinality(ctx, flip_state, K_start, K_min, deepen_per_level=False)
    logger.info(f"[strict] score-peak K={final_K} margin={final_state.margin:.4f}")
    if not _oob(ctx) and final_state.margin > -K.MARGIN_DEEPEN_TARGET:
        deepened, _ = OptimizeFixedK(ctx, final_state, final_K, K.MAX_ITERATIONS,
                                     deepen_target=-K.MARGIN_DEEPEN_TARGET)
        logger.info(f"[strict] deepened at K={final_K} margin={deepened.margin:.4f}")


def _k_of(state: State) -> int:
    return max(1, int(state.mask.sum().item()))


def _saturates(state: State, target: float) -> bool:
    """A flip whose margin reached the saturation target (full margin bonus)."""
    return state.margin < 0.0 and state.margin <= target


def _probe_k(ctx: Context, src: State, K_new: int, iters: int, target: float) -> State:
    """Warm-start `src` down to K_new (keep its top-retention coords) and re-optimize/deepen there. Every
    candidate it evaluates is folded into the score-ranked Bank; returns the optimized state."""
    warm = WarmStartSmallerK(ctx, src, K_new)
    cand, _ = OptimizeFixedK(ctx, warm, K_new, iters, deepen_target=target)
    return cand


def _remember_parent(ctx: Context, parents: list, state: State, cap: int) -> None:
    """Keep up to `cap` highest-score states as alternate warm-start parents (a tiny beam that breaks the
    single-lineage path dependence of a pure nested reduction)."""
    parents.append((_state_score(ctx, state), state))
    parents.sort(key=lambda t: t[0], reverse=True)
    del parents[cap:]


def _best_parent(parents: list, min_k: int) -> State:
    """Best-score parent whose support is large enough (K >= min_k) to warm-start DOWN to min_k; falls
    back to the largest-K parent if none qualifies."""
    eligible = [(s, st) for s, st in parents if _k_of(st) >= min_k]
    if eligible:
        return max(eligible, key=lambda t: t[0])[1]
    return max(parents, key=lambda t: _k_of(t[1]))[1]


def _bank_best_score(ctx: Context) -> float:
    return float(ctx.bank.best_safe["score"]) if ctx.bank.best_safe is not None else -1.0


def _score_upper_bound(ctx: Context, K_new: int) -> float:
    """Analytic MAX score achievable at cardinality K_new: perturbation at the fixed L∞=q and
    RMSE=q·√(K/N), plus the FULL margin and novelty bonuses. Cheap (no model eval) — if this ceiling
    can't beat the current Bank best, an expensive probe/retry at K_new is pointless and is skipped."""
    rmse = ctx.q * math.sqrt(max(0, int(K_new)) / max(1, ctx.clean_u8.numel()))
    return validator_score(ctx.q, rmse, -10.0, 10 ** 9, ctx.cap)  # margin -10 => full bonus; huge px => full novelty


def _state_from_delta(ctx: Context, template: State, res: dict) -> State:
    """Reconstruct an optimizer State from a banked result's byte delta, borrowing the path/clean fields
    from `template` (Phase A's state). Used by the opt-in C3 anchor-from-bank warm start (default off):
    the mask/sign come from the sparse banked flip, and the mask logits are rebuilt so TopK(a) reproduces
    that support. margin/x_adv are taken straight from the banked result (already exact-evaluated)."""
    st = copy.deepcopy(template)
    delta = res["delta"]
    st.mask = delta != 0
    st.sign[st.mask] = delta.sign()[st.mask].to(st.sign.dtype)
    st.a = K.MASK_NOISE * torch.randn_like(st.a)
    st.a[st.mask] += K.BIG_INIT
    st.margin = float(res["margin"])
    st.x_adv = res["cand"]
    return st


def _quick_anchor(ctx: Context, flip_state: State, target: float) -> tuple[State, bool]:
    """Budget-aware anchor (coupled Phase 1). Deepen the first flip at its OWN K toward `target`, giving
    the optimizer a REAL chance to reach CEIL (margin deepening here is non-monotonic and often delayed)
    while still bailing on genuinely hopeless images so the K/RMSE search isn't starved.

    Design (see constants ANCHOR_*):
      * warmup     — run at least ANCHOR_MIN_ITERS before ANY slope-based early stop;
      * trajectory — continue each chunk from the LIVE state (return_state=True), NOT the best-margin
                     snapshot, so a temporarily-uphill run that precedes a delayed drop is not discarded;
      * near-CEIL  — once best margin <= -ANCHOR_PUSH_MARGIN, never stall-bail (almost saturated, finish);
      * stall      — post-warmup, stop after ANCHOR_STALL_CHUNKS chunks whose best-margin gain stayed
                     < ANCHOR_MIN_MARGIN_GAIN (~a flat 20-iter window); at fixed K the best-margin slope
                     IS the score slope, so this is the slope + score-ROI gate in one;
      * caps       — hard ANCHOR_MAX_ITERS, and post-warmup a fraction-of-budget time cap (ANCHOR_MAX_FRAC).

    Returns (best_anchor, saturated). Every probe is folded into the score-ranked Bank; the returned
    best_anchor is the deepest snapshot (a good warm-start parent) — the K sweep decides the winner."""
    if _saturates(flip_state, target):                       # already deep — bonus ~maxed, no work
        return flip_state, True
    Kc = _k_of(flip_state)
    chunk = max(1, int(K.ANCHOR_CHUNK_ITERS))
    min_iters = max(0, int(K.ANCHOR_MIN_ITERS))
    max_iters = max(min_iters, chunk, int(K.ANCHOR_MAX_ITERS))
    push = -abs(float(K.ANCHOR_PUSH_MARGIN))                  # best margin <= push => never stall-bail
    t_cap = time.time() + max(0.0, K.ANCHOR_MAX_FRAC) * max(0.0, ctx.time_left())
    logger.info(f"[postflip:E1] QuickAnchor start: iters<={max_iters} K_flip={Kc}")
    t_e1 = time.time()
    live = best = flip_state                                  # `live` carries the trajectory; `best` the depth
    prev_margin = best.margin
    weak = done = 0
    while done < max_iters and not _oob(ctx):
        best_c, _, live = OptimizeFixedK(ctx, live, Kc, chunk, deepen_target=target, return_state=True)
        done += chunk
        if best_c.margin < best.margin:
            best = best_c
        if _saturates(best, target):
            logger.info(f"[postflip:E1] QuickAnchor done: iters_ran={done} spent={time.time() - t_e1:.3f}s "
                        f"saturates=True margin={best.margin:.4f}")
            return best, True
        gain = prev_margin - best.margin                     # >= 0: deepening of the BEST margin this chunk
        prev_margin = best.margin
        weak = weak + 1 if gain < K.ANCHOR_MIN_MARGIN_GAIN else 0
        # Early stops apply only AFTER the warmup and only while NOT already near CEIL.
        if done >= min_iters and best.margin > push:
            if time.time() >= t_cap:
                logger.debug(f"[quick-anchor] time cap iter={done} margin={best.margin:.4f}")
                break
            if weak >= max(1, int(K.ANCHOR_STALL_CHUNKS)):
                logger.debug(f"[quick-anchor] stalled (gain<{K.ANCHOR_MIN_MARGIN_GAIN} x{weak}) iter={done} "
                             f"margin={best.margin:.4f}")
                break
    sat = _saturates(best, target)
    logger.info(f"[postflip:E1] QuickAnchor done: iters_ran={done} spent={time.time() - t_e1:.3f}s "
                f"saturates={sat} margin={best.margin:.4f}")
    return best, sat


def PostFlipCoupled(ctx: Context, flip_state: State, K_start: int, K_min: int) -> None:
    """Three-phase score maximizer. The validator rewards a SCORE peak, not the saturation threshold, so
    binary search is demoted to a fast boundary LOCATOR and a real score sweep decides the winner.

      1) Anchor: deepen the first flip at its own K toward a SATURATING upper bound, but BUDGET-AWARE
         (QuickAnchor) — capped in iters/time and stopped on a weak margin slope, so a hard/unreachable
         CEIL can't consume the whole window. If it can't saturate, there is no K_sat -> score sweep with
         a bounded per-K refine.
      2) Locate K_sat: bracket-validated binary search with adaptive (current-bracket) tolerance, cheap
         classification probes, and a single retry from a diverse parent on an ambiguous near-miss (a
         fixed-K run can be a false negative, so `lo=mid` is not applied blindly).
      3) Refine: sample the real score curve at K/K_sat in COUPLED_REFINE_MULTS with the full per-K
         budget, warm-started from a small best-score parent beam.

    Every probe is score-ranked into the Bank (which also holds Phase A's low-K bare flips); the returned
    answer is the highest-score SAFE candidate, not the last binary-search state. No fixed margin-vs-RMSE
    priority — the Bank's full score arbitrates per image."""
    target = -K.MARGIN_DEEPEN_TARGET
    b_iters = max(1, int(K.ITERATIONS_PER_K * K.COUPLED_BOUNDARY_ITER_FRAC))  # cheap classify probes
    r_iters = max(1, int(K.ITERATIONS_PER_K))                                 # intensive refine probes
    K_floor = _novelty_floor_k(K_start, K_min)

    # --- Phase 1: anchor -> a validated saturating upper bound. Budget-aware: deepen only until the
    #     marginal return dries up (QuickAnchor), so a hard/unreachable CEIL can't eat the whole window. -
    parents: list = []
    # C3 (opt-in, default OFF): if the Bank already holds a flip sparser than the dense FIND state, anchor
    # from IT instead — tighter binary-search bracket + sparse start. Off by default (the dense anchor
    # doubles as a saturating-upper-bound validator, which a sparse start cannot provide).
    anchor_src = flip_state
    if K.POSTFLIP_ANCHOR_FROM_BANK and ctx.bank.best_safe is not None:
        b = ctx.bank.best_safe
        if b.get("delta") is not None and int(b["nz"]) < _k_of(flip_state):
            anchor_src = _state_from_delta(ctx, flip_state, b)
            logger.debug(f"[coupled] C3 anchor-from-bank: K {_k_of(flip_state)} -> {_k_of(anchor_src)}")
    if K.COUPLED_QUICK_ANCHOR:
        anchor, anchor_ok = _quick_anchor(ctx, anchor_src, target)   # E1 (logs its own start/end)
    else:
        logger.info(f"[postflip:E1] Anchor start: iters={r_iters} K_flip={_k_of(anchor_src)}")
        t_e1 = time.time()
        anchor, _ = OptimizeFixedK(ctx, anchor_src, _k_of(anchor_src), r_iters, deepen_target=target)
        anchor_ok = _saturates(anchor, target)
        logger.info(f"[postflip:E1] Anchor done: iters_ran={ctx.last_iters} spent={time.time() - t_e1:.3f}s "
                    f"saturates={anchor_ok} margin={anchor.margin:.4f}")
    _remember_parent(ctx, parents, anchor, K.COUPLED_PARENTS)
    hi = max(K_floor, _k_of(anchor))

    # --- Phase 2: bracket-validated binary search for K_sat (only under a real saturating anchor). ---
    logger.info(f"[postflip:E2] Ksat-search start: bracket=[{K_floor},{hi}] anchor_saturates={anchor_ok}")
    t_e2 = time.time()
    probes = 0
    K_sat = hi
    if anchor_ok:
        lo, src = K_floor, anchor
        while not _oob(ctx):
            tol = max(int(K.KSAT_ABS_TOL), int(K.KSAT_REL_TOL * hi))   # adaptive rel + absolute floor (#2)
            if hi - lo <= tol:
                break
            mid = (lo + hi) // 2
            probes += 1
            cand = _probe_k(ctx, src, mid, b_iters, target)
            _remember_parent(ctx, parents, cand, K.COUPLED_PARENTS)
            if _saturates(cand, target):
                src, hi = cand, mid                        # smaller K still saturates -> go lower
            elif (cand.margin <= K.COUPLED_RETRY_FRAC * target
                  and _score_upper_bound(ctx, mid) > _bank_best_score(ctx) + K.SCORE_TOL):
                # Near-miss whose ceiling can still beat the Bank: may be an optimizer false negative ->
                # retry once from a diverse parent with the full budget before conceding the bracket (#5).
                probes += 1
                retry = _probe_k(ctx, _best_parent(parents, mid), mid, r_iters, target)
                _remember_parent(ctx, parents, retry, K.COUPLED_PARENTS)
                if _saturates(retry, target):
                    src, hi = retry, mid
                else:
                    lo = mid
            else:
                lo = mid                                   # confident non-saturation (or can't beat Bank)
            logger.debug(f"[coupled] probe K={mid} margin={cand.margin:.4f} "
                         f"score={_state_score(ctx, cand):.4f} bracket=[{lo},{hi}]")
        K_sat = hi
    else:
        # C1: no saturating margin exists, but RMSE is still minimizable. Run a flip-preserving,
        # score-gated cardinality descent (deepen_per_level=False keeps the achieved margin and shrinks K
        # while total score holds within SCORE_TOL). Bank-gated: can only raise the returned score. Center
        # the Phase-3 sweep on the descended (sparse) K instead of the dense anchor.
        logger.debug("[coupled] anchor did not saturate -> flip-preserving score-gated descent (C1)")
        descended, _ = ReduceCardinality(ctx, anchor, _k_of(anchor), K_min, deepen_per_level=False)
        _remember_parent(ctx, parents, descended, K.COUPLED_PARENTS)
        K_sat = _k_of(descended)
    logger.info(f"[postflip:E2] Ksat-search done: probes={probes} K_sat={K_sat} "
                f"spent={time.time() - t_e2:.3f}s")

    # --- Phase 3: SCREEN-then-refine (#3, #7) -> cheaply sample the curve around/below K_sat, skipping
    #     any K whose analytic ceiling can't beat the Bank, then spend the budget only on the best
    #     COUPLED_REFINE_FULL screened states. Preserves broad score-curve discovery without starving the
    #     eventual winner of optimization depth. The per-candidate refine budget is decided by how close
    #     that candidate ALREADY is to CEIL (not by whether the anchor saturated): a candidate at CW margin
    #     <= -COUPLED_REFINE_DEEP_MARGIN is close enough that finishing to CEIL is worth the full budget
    #     (and it returns early on reaching it anyway); a shallow one gets a bounded ITERATIONS_PER_K pass
    #     so it can't grind an unreachable CEIL. This lets a good candidate recover even when the anchor
    #     itself fell short. The score-ranked Bank keeps the true best K either way. --------------------
    hi_cap = _k_of(anchor)
    probe_ks = sorted({min(hi_cap, max(K_floor, int(round(m * K_sat)))) for m in K.COUPLED_REFINE_MULTS})
    logger.info(f"[postflip:E3] refine-sweep start: probe_ks={probe_ks}")
    t_e3 = time.time()
    refined = 0
    screened: list[tuple[float, int, State]] = []
    for kk in probe_ks:
        if _oob(ctx):
            break
        if _score_upper_bound(ctx, kk) <= _bank_best_score(ctx) + K.SCORE_TOL:
            logger.debug(f"[coupled] skip K={kk} (UB {_score_upper_bound(ctx, kk):.4f} <= bank {_bank_best_score(ctx):.4f})")
            continue
        cand = _probe_k(ctx, _best_parent(parents, kk), kk, b_iters, target)   # cheap screen
        _remember_parent(ctx, parents, cand, K.COUPLED_PARENTS)
        screened.append((_state_score(ctx, cand), kk, cand))
        logger.debug(f"[coupled] screen K={kk} margin={cand.margin:.4f} score={_state_score(ctx, cand):.4f}")
    screened.sort(key=lambda t: t[0], reverse=True)
    for _, kk, cand in screened[: max(1, int(K.COUPLED_REFINE_FULL))]:
        if _oob(ctx):
            break
        # Full budget only for candidates already near CEIL; bounded otherwise (per-candidate, not blanket).
        cand_iters = K.MAX_ITERATIONS if cand.margin <= -K.COUPLED_REFINE_DEEP_MARGIN else max(1, int(K.ITERATIONS_PER_K))
        full, _ = OptimizeFixedK(ctx, cand, kk, cand_iters, deepen_target=target)
        _remember_parent(ctx, parents, full, K.COUPLED_PARENTS)
        refined += 1
        logger.debug(f"[coupled] refine K={kk} margin={full.margin:.4f} score={_state_score(ctx, full):.4f} iters={cand_iters}")
    logger.info(f"[postflip:E3] refine-sweep done: screened={len(screened)} refined={refined} "
                f"spent={time.time() - t_e3:.3f}s")

    best = ctx.bank.best_safe
    if best is not None:
        logger.info(f"[postflip] result: K_sat~{K_sat} best=({best['nz']}ch pixels={best['pixels']} "
                    f"margin={best['margin']:.4f} score={best['score']:.4f})")
    else:
        logger.info("[postflip] result: no safe flip banked")


def _interp_margin(K_new: int, samples: list[tuple[int, float]]) -> float:
    """Estimate margin at K_new from (K, margin) probe samples (margin is more negative at larger K).
    Piecewise-linear within the sampled range; LINEAR-EXTRAPOLATED below the smallest sample using the
    two lowest points, because margin degrades toward 0 (the flip dies) as K shrinks — clamping there
    would falsely predict a surviving deep flip at tiny K. Clamped above the largest (we don't probe
    above the anchor, and margin only deepens there)."""
    pts = sorted(samples)
    if len(pts) == 1:
        return pts[0][1]
    if K_new <= pts[0][0]:
        (k0, m0), (k1, m1) = pts[0], pts[1]
        return m0 + ((m1 - m0) / max(1, (k1 - k0))) * (K_new - k0)   # extrapolate the low-K trend
    if K_new >= pts[-1][0]:
        return pts[-1][1]
    for (k0, m0), (k1, m1) in zip(pts, pts[1:]):
        if k0 <= K_new <= k1:
            t = (K_new - k0) / max(1, (k1 - k0))
            return m0 + t * (m1 - m0)
    return pts[-1][1]


def PostFlipAnalytic(ctx: Context, flip_state: State, K_start: int, K_min: int) -> None:
    """Model-based score maximizer (Family 2). perturbation(K)=f(q√(K/N)) is closed-form, so given a
    cheap model of margin(K) the WHOLE score S(K) is a known 1-D function. We fit margin(K) from an
    anchor + a couple of spread probes, maximize the predicted S(K) over a fine grid for FREE (no model
    evals), and verify the predicted optimum + a small neighborhood at full budget — the Bank keeps the
    actual best. Fewer expensive probes than the binary search, so more depth lands on the winner. If the
    margin model mispredicts, the neighborhood verification + the Bank still recover a good candidate."""
    target = -K.MARGIN_DEEPEN_TARGET
    b_iters = max(1, int(K.ITERATIONS_PER_K * K.COUPLED_BOUNDARY_ITER_FRAC))
    r_iters = max(1, int(K.ITERATIONS_PER_K))
    K_floor = _novelty_floor_k(K_start, K_min)
    n = max(1, ctx.clean_u8.numel())

    # 1. Anchor + spread probes -> (K, margin) samples that define the margin(K) model.
    anchor, _ = OptimizeFixedK(ctx, flip_state, _k_of(flip_state), r_iters, deepen_target=target)
    K_a = _k_of(anchor)
    parents: list = []
    _remember_parent(ctx, parents, anchor, K.COUPLED_PARENTS)
    samples: list[tuple[int, float]] = [(K_a, anchor.margin)]
    for frac in K.ANALYTIC_PROBE_FRACS:
        if _oob(ctx):
            break
        kk = min(K_a, max(K_floor, int(round(frac * K_a))))
        cand = _probe_k(ctx, _best_parent(parents, kk), kk, b_iters, target)
        _remember_parent(ctx, parents, cand, K.COUPLED_PARENTS)
        samples.append((kk, cand.margin))
        logger.info(f"[analytic] sample K={kk} margin={cand.margin:.4f}")

    # 2. Maximize predicted S(K) over a geometric grid (analytic — no model evals).
    lo_k, hi_k = K_floor, K_a
    if hi_k > lo_k:
        grid = sorted({min(hi_k, max(lo_k, int(round(lo_k * (hi_k / lo_k) ** (i / 63.0))))) for i in range(64)})
    else:
        grid = [lo_k]

    def pred(kk: int) -> float:
        m = _interp_margin(kk, samples)
        if m >= 0.0:
            return 0.0                                   # predicted not-flipped -> no score (don't chase tiny K)
        rmse = ctx.q * math.sqrt(kk / n)
        return validator_score(ctx.q, rmse, m, 10 ** 9, ctx.cap)

    K_star = max(grid, key=pred)
    logger.info(f"[analytic] predicted K*={K_star} pred_score={pred(K_star):.4f} samples={len(samples)}")

    # 3. Verify K* + a small neighborhood at full budget (Bank keeps the actual best; skip by UB).
    verify_ks = sorted({min(hi_k, max(lo_k, int(round(f * K_star)))) for f in (1.0, 0.85, 1.15)})
    for kk in verify_ks:
        if _oob(ctx):
            break
        if _score_upper_bound(ctx, kk) <= _bank_best_score(ctx) + K.SCORE_TOL:
            continue
        cand = _probe_k(ctx, _best_parent(parents, kk), kk, r_iters, target)
        _remember_parent(ctx, parents, cand, K.COUPLED_PARENTS)
        logger.info(f"[analytic] verify K={kk} margin={cand.margin:.4f} score={_state_score(ctx, cand):.4f}")

    best = ctx.bank.best_safe
    if best is not None:
        logger.info(f"[analytic] K*~{K_star} best=({best['nz']}ch pixels={best['pixels']} "
                    f"margin={best['margin']:.4f} score={best['score']:.4f})")
    else:
        logger.info("[analytic] no safe flip banked")


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

        # ---- Optimizer (gated by RUN_OPTIM) ----
        # FIND: get the first flip at K_init (deepen_target=None). No margin-grinding at this large K —
        # RMSE (cardinality) is the bigger lever and the post-flip strategy handles both.
        if K.RUN_OPTIM:
            logger.info(f"[find-flip] start: iters={K.MAX_ITERATIONS} K_init={K_init} target=flip(margin<0)")
            t_ff = time.time()
            ctx.log_swaps = True
            successful_state, success = OptimizeFixedK(ctx, state, K_init, K.MAX_ITERATIONS)
            ctx.log_swaps = False
            logger.info(f"[find-flip] done: found={success} iters_ran={ctx.last_iters} "
                        f"spent={time.time() - t_ff:.3f}s margin={successful_state.margin:.4f}"
                        + (f" K_flip={_k_of(successful_state)}" if success else ""))
            if success:
                if K.POSTFLIP_STRATEGY == "strict":
                    PostFlipStrict(ctx, successful_state, K_init, K_min)
                elif K.POSTFLIP_STRATEGY == "analytic":
                    PostFlipAnalytic(ctx, successful_state, K_init, K_min)
                else:
                    PostFlipCoupled(ctx, successful_state, K_init, K_min)
                # INNER refinement on the chosen K (all strategies): exact deletion + score swaps.
                SupportRefine(ctx)
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
    reset_passes()
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
        deadline = hard_deadline = t_start + 1e9
        logger.info("[perturb] IGNORE_TIMEOUT on: deadline disabled")
    else:
        hard_deadline = t_start + max(0.05, float(timeout_seconds) - reserve_seconds)
        # Reserve a small slice for a possible q=2 fallback so a genuinely-unflippable-at-q=1 image (whose
        # q=1 search would otherwise grind to the hard deadline) still gets its retry. Only when a q=2 step
        # is actually possible (k_min currently 1). The reserve is reclaimed by the fallback (up to hard_deadline).
        fb_reserve = K.FALLBACK_Q2_SECONDS if (K.FALLBACK_Q2 and k_min < 2) else 0.0
        deadline = max(t_start + 0.05, hard_deadline - fb_reserve)

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
        optim_seconds=(K.OPTIM_SECONDS if K.RUN_OPTIM else 0.0), hard_deadline=hard_deadline,
        tuner=AdaptiveTuner(),
    )

    logger.info(f"[perturb] start: m0={m0:.4f} k_min={k_min} q={q:.6f} kappa={kappa:.4f} "
                f"tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} "
                f"feature_guided={K.FEATURE_GUIDED} "
                f"{'optim_budget=%.1fs' % K.OPTIM_SECONDS if K.RUN_OPTIM else 'optim=off'}")

    search(ctx)

    chosen = ctx.bank.result(ctx.allow_unsafe)
    if chosen is not None:
        chosen = _pad_novelty(ctx, chosen)
    elif K.FALLBACK_Q2 and k_min < 2:
        chosen = _q2_fallback(ctx, hard_deadline)

    kappa_n = calib.n_samples if calib is not None else 0
    if calib is not None and chosen is not None:
        try:
            m_exact = exact_worst_margin(ctx, chosen["cand"])
            calib.update(chosen["margin"], m_exact)
            calib.save()
        except Exception as err:
            logger.debug(f"[kappa] calibration update skipped: {err}")

    n_fwd, n_bwd = passes()
    if chosen is None:
        logger.info(
            f"[perturb] no {'' if ctx.allow_unsafe else 'safe '}flip -> clean "
            f"(m0={m0:.4f} elapsed={time.time() - t_start:.3f}s has_flip={ctx.bank.has_flip} "
            f"fwd={n_fwd} bwd={n_bwd} "
            f"tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} "
            f"kappa={kappa:.4f}{f'~n{kappa_n}' if use_dynamic else ''})"
        )
        return clean.detach().clamp(0.0, 1.0)

    pct = 100.0 * chosen["nz"] / max(1, clean_u8.numel())
    logger.info(
        f"[perturb] flip channels={chosen['nz']} ({pct:.2f}%) pixels={chosen.get('pixels', -1)} "
        f"margin={chosen['margin']:.4f} score={chosen.get('score', 0.0):.4f} "
        f"rmse={chosen['rmse']:.6f} linf={chosen['linf']:.6f} elapsed={time.time() - t_start:.3f}s "
        f"fwd={n_fwd} bwd={n_bwd} "
        f"m0={m0:.4f} tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} "
        f"kappa={kappa:.4f}{f'~n{kappa_n}' if use_dynamic else ''} safe={chosen is ctx.bank.best_safe}"
    )
    return chosen["cand"].detach().clamp(0.0, 1.0)
