"""perturb.py — anytime banked operator loop over a fixed-q sparse attack.

A fixed-q framework where every edit is delta_i in {-q, 0, +q}, q = k_min/255, and the perturbed image
is always built as
    delta = q * mask * sign ,  x_adv = x + delta
on the exact uint8 byte grid, so every candidate is validator-faithful and box-clipping directions are
forbidden.

ENGINE (anytime_search): a single phase-less controller replaces the old staged find -> saturate ->
reduce pipeline. It seeds two shared Pareto banks (pre-flip `frontier`, flipped `flipwork`) from Phase A
(InitializeAttack), then until the deadline repeatedly selects a diverse PARENT and runs the operator
with the best expected score-gain-per-second for ONE small quantum, folding every candidate through the
banks. The immutable submission Bank (utils.Bank) always holds the best exact-score validated flip.

Operators (interruptible, each reuses one numerical routine):
  * push / deepen        — OptimizeFixedK: cross the boundary (pre-flip) or deepen the margin (post-flip).
  * prune / ksat         — WarmStartSmallerK + re-optimize: shrink the support toward the score peak.
  * swap                 — ExactBlockSwap: replace weak active coords with promising inactive ones.
  * grow                 — _grow_to_saturation: climb a sparse flip up to the peak (from below).
  * refine / cleanup     — exact deletion (+ score swaps); cleanup is the margin-preserving pre-flip form.
  * restart              — PartialRestart: keep the strong core, re-seed the weakest fraction.

The surrogate m̂(K) (isotonic, anchored at the clean (0, m0) point) steers the prune/ksat targets. Budget
is one shared pool (no stage boundaries => no starvation, no unused tail); the returned answer is always
the best VALIDATED flip found, never merely whatever the last stage produced.
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
            anytime_search(ctx2)
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
    """Fixed-K optimizer (re-linearized mask/sign steps + block swaps + restarts). Returns (best_state,
    success); the push/deepen/prune operators call it for one bounded quantum. With return_state=True it
    also returns the LIVE end-of-run state as a third element, to CONTINUE the trajectory across quanta
    (best_state is the deepest snapshot; resuming from it would discard temporarily-uphill exploration
    that precedes a delayed nonlinear margin drop)."""
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

        # C2: diminishing-returns early stop for deepen quanta (deepen_target set). Once a good-enough
        # margin (<= -DEEPEN_STALL_FLOOR) is banked AND it has stalled over the window, return so the
        # controller can redirect budget to another operator. FIND/push (deepen_target None) is exempt —
        # it already stops at the first flip.
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
# WarmStartSmallerK — shrink a flip to a smaller support (top-retention coords); the prune/ksat operators.
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


def _k_of(state: State) -> int:
    return max(1, int(state.mask.sum().item()))


def _bank_best_score(ctx: Context) -> float:
    return float(ctx.bank.best_safe["score"]) if ctx.bank.best_safe is not None else -1.0


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


# ==========================================================================================
# Sparse-then-grow post-flip (PERTURB_POSTFLIP_STRATEGY=grow|both). Reach the score peak from BELOW:
# a lean guided core grown in margin-gain order until the margin saturates. A structural complement to
# coupled's dense->shrink — a different route to ~K_sat that can land a better coordinate SET.
# ==========================================================================================
def _grow_to_saturation(ctx: Context, core: State, target: float) -> State:
    """Grow a sparse `core` UP toward the score peak. Each step re-linearizes the gradient at the current
    grown image, adds a block of the steepest feasible margin-gain INACTIVE coords (block = max(
    GROW_MIN_BATCH, GROW_ADD_FRAC·K_cur)), and folds the new cumulative state through the score-ranked
    Bank. Climbs while it makes progress; stops at saturation (margin <= target => extra coords only
    raise RMSE) or after GROW_PATIENCE non-improving adds. Progress is a NEW Bank best once flipped, or a
    NEW margin low while still pre-flip (so the pre-flip prefix isn't mistaken for a stall). Returns the
    final grown state (the Bank holds the actual best across the whole trajectory)."""
    state = copy.deepcopy(core)
    baseline = _bank_best_score(ctx)      # score to beat once flipped
    prev_margin = state.margin
    misses = steps = grew = 0
    while not _oob(ctx) and steps < max(1, int(K.GROW_MAX_STEPS)):
        steps += 1
        if state.margin <= target:        # saturated: the score peak is at/below here -> stop growing
            break
        inactive = ~state.mask
        n_inactive = int(inactive.sum().item())
        if n_inactive <= 0:
            break
        _m, g = _grad_at(ctx, state.x_adv)
        in_sign = _best_direction(ctx, g)
        in_gain = (-g * ctx.q * in_sign).clamp(min=0.0)
        in_gain[state.mask] = float("-inf")                        # inactive coords only
        add = min(n_inactive, max(int(K.GROW_MIN_BATCH), int(round(K.GROW_ADD_FRAC * _k_of(state)))))
        cand = torch.topk(in_gain, add).indices
        cand = cand[in_gain[cand] > 0]                             # only coords with a real feasible gain
        if cand.numel() == 0:
            break
        state.mask = state.mask.clone(); state.mask[cand] = True
        state.sign = state.sign.clone(); state.sign[cand] = in_sign[cand]
        state.gradient = g
        state.margin, state.x_adv, _ = _evaluate_state(ctx, state.mask, state.sign)   # folds to Bank
        grew += 1
        if state.margin < 0.0:
            cur = _bank_best_score(ctx)
            improved = cur > baseline + 1e-9
            baseline = max(baseline, cur)
        else:
            improved = state.margin < prev_margin - 1e-6           # pre-flip: closing on the boundary
        prev_margin = state.margin
        misses = 0 if improved else misses + 1
        if misses >= max(1, int(K.GROW_PATIENCE)):
            break
    logger.debug(f"[grow] climb: steps={grew} K={_k_of(state)} margin={state.margin:.4f} "
                 f"score={_state_score(ctx, state):.4f}")
    return state


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


def _ufs_geom_grid(lo: int, hi: int, n: int) -> list[int]:
    """A geometric K grid on [lo, hi] for the surrogate argmax (cheap, no model evals)."""
    lo = max(1, int(lo)); hi = max(lo, int(hi))
    if hi == lo:
        return [lo]
    return sorted({min(hi, max(lo, int(round(lo * (hi / lo) ** (i / max(1, n - 1)))))) for i in range(max(2, n))})


def _isotonic_nonincreasing(pts: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Pool-Adjacent-Violators fit of margin(K) constrained NON-INCREASING in K (a larger support can
    always match or beat a smaller one's margin, so the achievable margin only deepens with K). Pools
    adjacent points that violate monotonicity into their mean, smoothing optimizer-noise wiggles that
    would otherwise make the piecewise-linear surrogate pick a wrong K̂*. `pts` is sorted ascending by K."""
    blocks: list[list] = []                          # each: [sum_m, count, [ks...]]
    for k, m in pts:
        blocks.append([m, 1, [k]])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s2, c2, k2 = blocks.pop(); s1, c1, k1 = blocks.pop()   # earlier mean < later => violation
            blocks.append([s1 + s2, c1 + c2, k1 + k2])
    out: list[tuple[int, float]] = []
    for s, c, ks in blocks:
        v = s / c
        out.extend((k, v) for k in ks)
    return out


def _ufs_margin_hat(K_new: int, samples: dict, m0: float) -> float:
    """Surrogate margin at K_new: anchor the fit at the KNOWN clean point (K=0, m0) — margin at zero
    changed coords IS the clean margin — then isotonic-monotonize the observed (K, margin) probes and
    interpolate. The (0, m0) anchor removes the dangerous below-range extrapolation of the raw piecewise-
    linear interp (which mispredicted a flip at tiny K); between (0, m0>0) and the sparsest flip it now
    interpolates the real flip boundary for free."""
    pts = sorted(samples.items())
    if not pts or pts[0][0] > 0:
        pts = [(0, float(m0))] + pts
    return _interp_margin(K_new, _isotonic_nonincreasing(pts))


# ==========================================================================================
# ANYTIME BANKED OPERATOR LOOP (the engine). Replaces the staged find -> saturate -> reduce pipeline
# AND every post-flip strategy with ONE phase-less controller over phase-aware, interruptible operators
# and shared Pareto banks. Each iteration selects a diverse PARENT from the banks, runs the operator with
# the best expected score-gain-per-second for one small QUANTUM, and folds every produced candidate
# through the banks. ctx.bank stays the immutable submission bank (best exact-score validated flip); the
# two Pareto banks below only hold warm-start parents, so an operator can never lower the returned score.
#
# Deliberate deviation from a pure "margin as safety-slack" model: THIS validator pays a margin BONUS
# (0.03·clip(-margin/10)) up to CW margin -10, so margin depth is a first-class score term, not just
# safety currency. The controller therefore ranks every candidate by the EXACT validator score (which
# already prices the bonus and saturates it past -10), rather than stopping deepening at a safe threshold.
# ==========================================================================================
def _cand_metrics(ctx: Context, state: State) -> dict:
    """The state's validator metrics on its current image (score is 0 unless flipped)."""
    diff = state.x_adv - ctx.clean
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    pixels = int((diff.abs() > 0.5 * ctx.q).any(dim=0).sum().item())
    flip = state.margin < 0.0
    score = validator_score(linf, rmse, state.margin, pixels, ctx.cap) if flip else 0.0
    return {"rmse": rmse, "pixels": pixels, "flip": flip, "safe": state.margin <= -ctx.kappa, "score": score}


class _Cand:
    """A warm-start parent held in a Pareto bank: a State plus its cached validator metrics."""
    __slots__ = ("state", "margin", "rmse", "score", "K", "flip", "safe", "key", "nid")

    def __init__(self, ctx: Context, state: State, nid: int) -> None:
        m = _cand_metrics(ctx, state)
        self.state = state
        self.margin = float(state.margin)
        self.rmse = m["rmse"]
        self.score = m["score"]
        self.K = _k_of(state)
        self.flip = m["flip"]
        self.safe = m["safe"]
        self.key = (self.K, round(self.margin, 3), round(self.rmse, 6))     # coarse dedup fingerprint
        self.nid = nid


class _ParetoBank:
    """Bounded Pareto set over a few metric dims — keeps DIVERSE tradeoffs (a deep-margin candidate with a
    currently-lower score is an excellent prune parent, so it must survive), not just the single best.
    dims = [(attr, +1|-1)]: +1 higher-is-better, -1 lower-is-better. Over the limit, the most crowded
    (metric-space nearest-neighbour) candidate is evicted so the retained set stays spread out."""
    def __init__(self, limit: int, dims: list) -> None:
        self.limit = max(1, int(limit))
        self.dims = dims
        self.items: list[_Cand] = []

    def _dominates(self, a: _Cand, b: _Cand) -> bool:
        strict = False
        for attr, s in self.dims:
            va, vb = getattr(a, attr) * s, getattr(b, attr) * s
            if va < vb:
                return False
            if va > vb:
                strict = True
        return strict

    def insert(self, c: _Cand) -> bool:
        for it in self.items:
            if it.key == c.key or self._dominates(it, c):
                return False
        self.items = [it for it in self.items if not self._dominates(c, it)]
        self.items.append(c)
        if len(self.items) > self.limit:
            self._evict_crowded()
        return True

    def _evict_crowded(self) -> None:
        its = self.items
        span = {attr: max(max(getattr(x, attr) for x in its) - min(getattr(x, attr) for x in its), 1e-9)
                for attr, _ in self.dims}
        def d2(a, b):
            return sum(((getattr(a, attr) - getattr(b, attr)) / span[attr]) ** 2 for attr, _ in self.dims)
        worst, worst_nn = 0, float("inf")
        for i, a in enumerate(its):
            nn = min(d2(a, b) for j, b in enumerate(its) if j != i)
            if nn < worst_nn:
                worst_nn, worst = nn, i
        its.pop(worst)


# ---- Interruptible operators. Each runs ONE quantum and returns the States it produced (all also folded
#      into the submission Bank via _eval inside the reused routines). Parents are never mutated: routines
#      that edit in place get a deepcopy. -------------------------------------------------------------
def _op_push(ctx: Context, state: State, target: float | None) -> list[State]:
    """Margin push/deepen at the state's own K. target=None stops at the first flip (pre-flip FIND);
    target=-CEIL deepens a flip toward the margin-bonus saturation."""
    st, _ = OptimizeFixedK(ctx, state, _k_of(state), max(1, int(K.AL_QUANTUM_ITERS)), deepen_target=target)
    return [st]


def _op_prune(ctx: Context, state: State, K_new: int) -> list[State]:
    """Support prune + repair: warm-start to a smaller K (top-retention coords) then re-deepen there."""
    K_new = max(1, min(_k_of(state) - 1, int(K_new)))
    if K_new < 1:
        return []
    warm = WarmStartSmallerK(ctx, state, K_new)
    st, _ = OptimizeFixedK(ctx, warm, K_new, max(1, int(K.AL_QUANTUM_ITERS)), deepen_target=-K.MARGIN_DEEPEN_TARGET)
    return [warm, st]


def _op_swap(ctx: Context, state: State) -> list[State]:
    """One exact block-swap round (replace weak active coords with promising inactive ones)."""
    st = copy.deepcopy(state)
    _m, st.gradient = _grad_at(ctx, st.x_adv)                          # refresh saliency for the swap pools
    block = max(int(_p(ctx, "BLOCK_MIN")), int(round(_p(ctx, "BLOCK_FRAC") * _k_of(st))))
    st, _ = ExactBlockSwap(ctx, st, max(1, min(block, _k_of(st))))
    return [st]


def _op_grow(ctx: Context, state: State) -> list[State]:
    """Grow a sparse flip up toward the score peak (sparse-then-grow, approached from below)."""
    return [_grow_to_saturation(ctx, copy.deepcopy(state), -K.MARGIN_DEEPEN_TARGET)]


def _op_restart(ctx: Context, state: State) -> list[State]:
    """Diversify: keep the strong core, re-seed the weakest fraction from the reservoirs."""
    st = copy.deepcopy(state)
    if st.gradient is None:
        _m, st.gradient = _grad_at(ctx, st.x_adv)
    return [PartialRestart(ctx, st, _k_of(st))]


def _op_refine(ctx: Context, state: State, preserve_margin: bool = False) -> list[State]:
    """One exact deletion (+ score-swap) round on `state` — the old SupportRefine as a single quantum.
    preserve_margin=True is the PRE-FLIP guard: propose only deletions and keep only those that did not
    worsen the margin (never run unconstrained distortion reduction on an unflipped candidate)."""
    d = _make_delta(ctx, state.mask, state.sign)
    mask = d != 0
    sign = d.sign()
    active = mask.nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        return []
    _m, g = _grad_at(ctx, state.x_adv)
    retention = -g[active] * ctx.q * sign[active]                      # small/negative => removable
    weak = active[torch.argsort(retention)][: max(1, int(K.REFINE_DELETION_POOL))]
    cands: list[torch.Tensor] = []
    for i in weak.tolist():
        c = d.clone(); c[i] = 0.0; cands.append(c)
    for frac in (0.5, 0.25, 0.1):
        k = max(1, int(frac * weak.numel()))
        c = d.clone(); c[weak[:k]] = 0.0; cands.append(c)
    if not preserve_margin:                                            # swaps only make sense post-flip
        in_sign = _best_direction(ctx, g)
        in_gain = (-g * ctx.q * in_sign).clamp(min=0.0)
        in_gain[mask] = float("-inf")
        n_in = min(int(K.REFINE_SWAP_POOL), int((~mask).sum().item()))
        if n_in > 0:
            strong = torch.topk(in_gain, n_in).indices
            for t in range(min(int(weak.numel()), int(strong.numel()), int(K.REFINE_SWAP_PROPOSALS))):
                i, j = int(weak[t]), int(strong[t])
                c = d.clone(); c[i] = 0.0; c[j] = float(in_sign[j]) * float(ctx.k_min); cands.append(c)
    res = _eval(ctx, cands)
    key = (lambda r: r["margin"]) if preserve_margin else (lambda r: -r["score"])
    out: list[State] = []
    for r in sorted((r for r in res if r.get("delta") is not None), key=key)[:3]:
        st = _state_from_delta(ctx, state, r)
        if not preserve_margin or st.margin <= state.margin + 1e-6:
            out.append(st)
    return out


def _eligible(state: State) -> list[str]:
    """Operators valid for the parent's phase. Pre-flip is mostly margin push (never unconstrained
    distortion reduction); Ksat/grow need a robust flipped anchor to be meaningful."""
    if state.margin >= 0.0:
        return ["push", "cleanup", "restart"]
    ops = ["deepen", "prune", "swap", "refine", "restart"]
    if state.margin <= -K.AL_ROBUST_MARGIN:
        ops += ["ksat", "grow"]
    return ops


def _choose_op(ops: list[str], voi: dict, cnt: dict, action: int) -> str:
    """Pick the operator with the best score-gain-per-second (UCB-style exploration bonus); force a
    periodic restart so the search can escape a trapped support/target trajectory."""
    if action % max(1, int(K.AL_RESTART_EVERY)) == 0 and "restart" in ops:
        return "restart"
    untried = [o for o in ops if o not in voi]
    if untried:
        return untried[0]
    best, best_u = ops[0], float("-inf")
    for o in ops:
        u = voi[o] + K.AL_EXPLORE_C * math.sqrt(math.log(action + 1) / max(1, cnt.get(o, 1)))
        if u > best_u:
            best_u, best = u, o
    return best


def _select_parent(frontier: _ParetoBank, flipwork: _ParetoBank, action: int) -> State | None:
    """Parent mixture: mostly the best-score flip, but rotate in the deepest-margin and leanest-RMSE
    flips (excellent prune/deepen parents) and an occasional diverse pick, so the loop does not grind a
    single local optimum. Before any flip exists, take the frontier candidate closest to the boundary."""
    if flipwork.items:
        r = action % 5
        if r <= 1:
            return max(flipwork.items, key=lambda c: c.score).state
        if r == 2:
            return min(flipwork.items, key=lambda c: c.margin).state          # deepest margin
        if r == 3:
            return min(flipwork.items, key=lambda c: c.rmse).state             # leanest RMSE
        pool = flipwork.items + frontier.items
        return pool[action % len(pool)].state
    if frontier.items:
        return min(frontier.items, key=lambda c: c.margin).state              # closest to flipping
    return None


def _run_op(ctx: Context, op: str, state: State, kstar: int, K_floor: int) -> list[State]:
    if op == "push":
        return _op_push(ctx, state, None)
    if op == "deepen":
        return _op_push(ctx, state, -K.MARGIN_DEEPEN_TARGET)
    if op == "prune":
        return _op_prune(ctx, state, max(K_floor, int(round(0.8 * _k_of(state)))))
    if op == "ksat":
        return _op_prune(ctx, state, max(K_floor, min(_k_of(state) - 1, kstar)))
    if op == "swap":
        return _op_swap(ctx, state)
    if op == "grow":
        return _op_grow(ctx, state)
    if op == "refine":
        return _op_refine(ctx, state)
    if op == "cleanup":
        return _op_refine(ctx, state, preserve_margin=True)
    if op == "restart":
        return _op_restart(ctx, state)
    return []


def anytime_search(ctx: Context) -> None:
    """The engine. Seed the banks from Phase A, then run the phase-less operator loop until the deadline.
    The submission Bank (ctx.bank) always holds the best exact-score safe flip found — the returned answer
    is never worse than what any single operator produced, and unused/over-used stage time cannot occur."""
    N = ctx.clean_u8.numel()
    K_init = max(1, int(round(K.K_INIT_FRAC * N)))
    K_floor = _novelty_floor_k(K_init, max(1, int(round(K.K_MIN_FRAC * N))))
    targets = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.TOPM)
    logger.info(f"[anytime] N={N} K_init={K_init} K_floor={K_floor} targets={targets} "
                f"quantum={K.AL_QUANTUM_ITERS} robust<={-K.AL_ROBUST_MARGIN}")

    frontier = _ParetoBank(K.AL_FRONTIER_LIMIT, [("margin", -1), ("rmse", -1)])          # pre-flip
    flipwork = _ParetoBank(K.AL_FLIPWORK_LIMIT, [("score", 1), ("rmse", -1), ("margin", -1)])  # flipped
    samples: dict[int, float] = {}
    nid = [0]

    def observe(st: State) -> None:
        k = _k_of(st)
        if k not in samples or st.margin < samples[k]:
            samples[k] = float(st.margin)

    def score_at(k: int, m: float) -> float:
        if m >= 0.0:
            return 0.0
        return validator_score(ctx.q, ctx.q * math.sqrt(max(0, int(k)) / N), m, 10 ** 9, ctx.cap)

    def K_star() -> int:
        if not samples:
            return K_floor
        grid = _ufs_geom_grid(K_floor, max(samples), int(K.UFS_GRID))
        return max(grid, key=lambda k: score_at(k, _ufs_margin_hat(k, samples, ctx.m0)))

    def ingest(states: list[State]) -> None:
        for st in states:
            if st is None:
                continue
            nid[0] += 1
            observe(st)
            c = _Cand(ctx, st, nid[0])
            (flipwork if c.flip else frontier).insert(c)

    voi: dict = {}
    cnt: dict = {}
    action = 0
    try:
        seed_state, _ = InitializeAttack(ctx, targets, K_init)
        ingest([seed_state])
        for b in (ctx.bank.best_safe, ctx.bank.best_flip):                   # fold Phase-A flips as parents
            if b is not None and b.get("delta") is not None:
                ingest([_state_from_delta(ctx, seed_state, b)])

        while not _oob(ctx) and action < int(K.AL_MAX_ACTIONS):
            if not flipwork.items and not frontier.items:
                break
            action += 1
            parent = _select_parent(frontier, flipwork, action)
            if parent is None:
                break
            op = _choose_op(_eligible(parent), voi, cnt, action)
            kstar = K_star()
            before = _bank_best_score(ctx)
            t_op = time.time()
            ingest(_run_op(ctx, op, parent, kstar, K_floor))
            rate = max(0.0, _bank_best_score(ctx) - before) / max(1e-6, time.time() - t_op)
            voi[op] = 0.5 * voi.get(op, rate) + 0.5 * rate
            cnt[op] = cnt.get(op, 0) + 1
    except _FirstFlipFound:
        pass
    except Exception as err:                                                 # never crash the miner
        logger.warning(f"[anytime] loop aborted after {action} actions: {err}")

    b = ctx.bank.best_safe
    logger.info(f"[anytime] done: actions={action} frontier={len(frontier.items)} flipwork={len(flipwork.items)} "
                + (f"best=({b['nz']}ch pixels={b['pixels']} margin={b['margin']:.4f} score={b['score']:.4f})"
                   if b is not None else "no safe flip banked"))


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

    anytime_search(ctx)

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
