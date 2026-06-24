"""perturb.py — one-shot sparse attack engine (Perturb subnet miner).

A fast, reliable single-pass sparse attack: one gradient, then binary-search the MINIMAL flipping
prefix of the gradient-ranked channels. The linear crossing-size estimate over-counts ~10x near the
boundary (it yields dense, low-score flips), so we DON'T trust it — we verify the actual minimal k.
If no prefix of the clean gradient flips (curved boundary), a bounded iterated-FGSM re-linearization
finds a flipping direction and we sparsify along it the same way.

MATH (CW margin m = logit[true] - max_{j!=true} logit[j], step q = k_min/255):
  first order:  m(x0 + δ) ≈ m0 + gᵀδ,  g = ∇ₓ m(x0); move channel i by δ_i = -q·sign(g_i).
  Rank channels by |g_i| (descending); binary-search the gradient-ranked prefix for the minimal k
  that actually flips, verified on the validator-faithful path.

TRANSFER SAFETY (unchanged from the package): the cuDNN-TF32 ambient regime (PERTURB_TF32_ON) plus a
worst-case TF32 envelope, and a residual-cushion kappa (PERTURB_KAPPA_RESID under the envelope, else
PERTURB_MINER_MARGIN_BUFFER) — a flip counts as "safe" only when its worst-case margin <= -kappa.

BYTE-SPACE: every edit is an exact integer ±k_min byte step on the uint8 grid, so candidates sit
exactly on the k/255 grid and the PNG round-trip is identity (see utils.py / constants.py). Tuning is
via the PERTURB_* env vars in constants.py — no redeploy needed.
"""

from __future__ import annotations

import logging
import math
import time

import torch

from . import constants as K
from .utils import (
    Bank,
    Context,
    apply_byte,
    batch_eval,
    build_sparse_order,
    estimate_k,
    margin_and_grad,
    out_of_budget,
)

logger = logging.getLogger(__name__)

# Match the validator's numeric regime: cuDNN convolutions use TF32 (PERTURB_TF32_ON, default on for
# CUDA); matmul TF32 stays off (the validator default). EfficientNetV2-L is conv-dominated, so cuDNN is
# the axis that matters; the envelope (utils.batch_eval) brackets the residual cross-GPU/library drift.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = K.TF32_ON
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _verify_prefix(ctx: Context, move_dir: torch.Tensor, order: torch.Tensor, k: int) -> bool:
    """Evaluate the top-k gradient-ranked prefix as a ±k_min byte flip, fold it into the bank, and
    return whether it is an in-band, quality flip (envelope worst-case margin < 0)."""
    sel = order[:max(1, k)]
    cand = apply_byte(ctx.clean_u8, move_dir, sel, ctx.k_min, ctx.shape)
    res = batch_eval(ctx, [cand])
    ctx.bank.consider(res)
    return bool(res and res[0]["quality"])


def _min_flip(ctx: Context, move_dir: torch.Tensor, order: torch.Tensor, valid: int, k_seed: int) -> None:
    """Exp-up to a flipping prefix, then bisect DOWN for the minimal flipping k along this order —
    instead of trusting the linear estimate, which over-counts and yields dense flips."""
    if valid <= 0:
        return
    ks = max(1, min(k_seed, valid))
    if _verify_prefix(ctx, move_dir, order, ks):
        lo, hi = 0, ks
    else:
        lo, hi, k = ks, None, min(2 * ks, valid)
        while not out_of_budget(ctx):
            if _verify_prefix(ctx, move_dir, order, k):
                hi = k
                break
            if k >= valid:
                break
            lo, k = k, min(2 * k, valid)
        if hi is None:
            return  # no prefix of this order flips
    while lo + 1 < hi and not out_of_budget(ctx):
        mid = (lo + hi) // 2
        if _verify_prefix(ctx, move_dir, order, mid):
            hi = mid
        else:
            lo = mid


def one_shot(ctx: Context) -> None:
    """One gradient -> minimal flipping prefix (Phase A) -> iterated-FGSM re-linearization (Phase B).

    Banks the best envelope-safe / flipping candidate found within the deadline. Phase A bisects the
    clean hard-margin gradient's ranked prefix for the minimal flipping k; Phase B kicks in only when
    no clean-gradient prefix flips (curved boundary), re-linearizing in the ±q box until a dense flip
    appears, then sparsifying along that better direction."""
    clean_flat = ctx.clean.view(-1)

    # PHASE A: minimal flipping prefix on the clean hard-margin gradient. move_dir = -sign(g0); rank
    # channels by |g0| and seed the search at the linear crossing estimate (an upper bound).
    move_dir = -ctx.g0.sign()
    masked, order, valid = build_sparse_order(ctx.g0.abs(), move_dir, clean_flat)
    if valid == 0:
        logger.info(f"[oneshot] no feasible channels -> clean (m0={ctx.m0:.4f})")
        return
    k_seed = estimate_k(ctx.m0 + max(ctx.kappa, 0.0), masked[order][:valid], ctx.q)
    _min_flip(ctx, move_dir, order, valid, k_seed)

    # PHASE B: re-linearized fallback — iterated FGSM in the ±q box until a (dense) flip, then
    # sparsify along that boundary gradient. Catches images the single clean-gradient prefix misses.
    if not ctx.bank.has_flip:
        step_bytes = torch.zeros_like(ctx.clean_u8)
        for _ in range(K.MAX_RELIN):
            if out_of_budget(ctx):
                break
            x = ((ctx.clean_u8 + step_bytes).clamp(0.0, 255.0) / 255.0).view(ctx.shape)
            m, g = margin_and_grad(ctx.model, x, ctx.target_index)
            gl = g.view(-1)
            if m < 0.0:  # float-flipped at this iterate -> sparsify along its boundary gradient
                md2 = -gl.sign()
                masked2, order2, valid2 = build_sparse_order(gl.abs(), md2, clean_flat)
                _min_flip(ctx, md2, order2, valid2, valid2)
                break
            step_bytes = -float(ctx.k_min) * gl.sign()  # dense single byte-step from clean, box-confined


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
    """One-shot sparse ±k_min/255 flip: minimal flipping gradient-prefix + re-linearization fallback.
    Returns the best envelope-safe flip (or any margin<0 flip when PERTURB_ALLOW_UNSAFE_FLIP=1); else
    the clean image when no flip is found within the deadline."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(K.MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))  # fixed unit step (typically 1)
    q = k_min * K.Q

    # Byte-space: snap clean to its uint8 grid; every edit is an integer BYTE step on this, so
    # candidates are exactly on the k/255 grid and the PNG round-trip is identity.
    clean_u8 = torch.round(clean.view(-1) * 255.0)
    envelope = K.TF32_ENVELOPE and device.type == "cuda"
    kappa = K.KAPPA_RESID if envelope else K.MARGIN_BUFFER

    # One gradient evaluation up front: clean margin m0 + boundary gradient g0, and a t_step gate.
    g_t0 = time.time()
    m0, g0 = margin_and_grad(model, clean, target_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = max(1e-4, time.time() - g_t0)

    # Scale the reserve with the per-forward cost so larger images/models leave enough post-search
    # headroom for serialization + verification; the search deadline shrinks accordingly.
    if reserve_seconds is None:
        reserve_seconds = K.RESERVE_SECONDS
    reserve_seconds = float(reserve_seconds) + K.RESERVE_FWD_MULT * t_step
    deadline = t_start + max(0.05, float(timeout_seconds) - reserve_seconds)

    def time_left() -> float:
        return deadline - time.time()

    ctx = Context(
        model=model, device=device, clean=clean, clean_u8=clean_u8, shape=clean.shape,
        target_index=target_index, k_min=k_min, q=q, floor=floor, cap=cap, kappa=kappa,
        skip_roundtrip=K.SKIP_ROUNDTRIP, tf32_on=K.TF32_ON, envelope=envelope,
        allow_unsafe=K.ALLOW_UNSAFE_FLIP, deadline=deadline, t_step=t_step, time_left=time_left,
        bank=Bank(), m0=m0, g0=g0.view(-1),
    )

    one_shot(ctx)

    chosen = ctx.bank.result(ctx.allow_unsafe)
    if chosen is None:
        logger.info(
            f"[perturb] no {'' if ctx.allow_unsafe else 'safe '}flip -> clean "
            f"(m0={m0:.4f} elapsed={time.time() - t_start:.3f}s has_flip: {ctx.bank.has_flip} "
            f"tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} kappa={kappa:.4f})"
        )
        return clean.detach().clamp(0.0, 1.0)

    pct = 100.0 * chosen["nz"] / max(1, clean_u8.numel())
    logger.info(
        f"[perturb] flip channels={chosen['nz']} ({pct:.2f}%) margin={chosen['margin']:.4f} "
        f"rmse={chosen['rmse']:.6f} linf={chosen['linf']:.6f} elapsed={time.time() - t_start:.3f}s "
        f"m0={m0:.4f} tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'} "
        f"kappa={kappa:.4f} safe={chosen is ctx.bank.best_safe}"
    )
    return chosen["cand"].detach().clamp(0.0, 1.0)
