"""Perturb attack engine (DEV duplicate) — cleaned to a minimal baseline.

Current algorithm: ONE-SHOT + BINARY-SEARCH LINEAR FLIP FINDER.

  1. One gradient at the clean image gives the CW margin m0 and its input gradient g0.
  2. Move direction = -sign(g0): the fixed L∞ descent direction on the margin.
  3. One-shot candidate: step EVERY channel by k_max = round(cap·255) bytes along that direction (a
     full-budget FGSM step). If it does not flip, there is no flip on this line -> return the clean image.
  4. If it flips, binary-search the SMALLEST uniform byte step k in [1, k_max] that still flips. A lower k
     is a shorter step -> smaller L∞/RMSE -> better perturbation score. This is a linear line search: every
     probed candidate is clean + (k/k_max)·(one-shot delta), snapped to the byte grid.
  5. All probes fold through the score-ranked Bank (full validator objective, envelope-safe accept gate),
     so the returned flip is the best-scoring one seen. No flip banked -> return the clean image.

The heavy multi-phase optimizer that used to live here is gone; utils.py / calibration.py are the exact
production primitives (batched envelope-safe evaluator + online kappa calibration).
"""

from __future__ import annotations

import logging
import math
import time

import torch

from . import constants as K
from .calibration import env_fingerprint, get_calibrator
from .utils import (
    Bank,
    Context,
    apply_delta_bytes,
    batch_eval,
    margin_and_grad,
    movable,
    passes,
    reset_passes,
)

logger = logging.getLogger(__name__)

# Match the validator's numeric regime: cuDNN convolutions use TF32 (the validator's PyTorch default);
# matmul TF32 stays off. Set once at import so every forward/backward here matches the validator's logits.
torch.backends.cudnn.allow_tf32 = K.TF32_ON
torch.backends.cuda.matmul.allow_tf32 = False


# ==========================================================================================
# Linear flip finder.
# ==========================================================================================
def _candidate(ctx: Context, move_dir_flat: torch.Tensor, k: int) -> torch.Tensor:
    """clean + k bytes along move_dir on every channel (clamped to the [0,255] box) -> chw float."""
    delta = float(k) * move_dir_flat
    return apply_delta_bytes(ctx.clean_u8, delta, ctx.shape)


def _search(ctx: Context, move_dir_flat: torch.Tensor, k_max: int) -> None:
    """One-shot at k_max; if it flips, binary-search the smallest flipping uniform byte step k.

    Seeds the bracket with a coarse linear grid over (0, k_max], then bisects between the smallest
    flipping k (hi) and the largest non-flipping k (lo). Every candidate is graded by the Bank."""
    # One-shot full-budget step. This is the maximum reach on the line; if it does not flip, nothing on
    # the line does (a bigger step along -sign(g) can only push the margin further down).
    one_shot = _candidate(ctx, move_dir_flat, k_max)
    res = batch_eval(ctx, [one_shot])
    ctx.bank.consider(res)
    if not (res and res[0]["flipped"]):
        return  # no flip on this line

    # Coarse linear grid to tighten the [lo, hi] bracket before bisection (cheap, one batched forward).
    lo, hi = 0, k_max  # lo: largest known non-flip; hi: smallest known flip (k_max flips)
    grid_n = max(1, K.LINE_GRID)
    ks = sorted({max(1, min(k_max, round(k_max * i / (grid_n + 1)))) for i in range(1, grid_n + 1)})
    if ks:
        grid = [_candidate(ctx, move_dir_flat, k) for k in ks]
        results = batch_eval(ctx, grid)
        ctx.bank.consider(results)
        for k, r in zip(ks, results):
            if r["flipped"]:
                hi = min(hi, k)
            else:
                lo = max(lo, k)

    # Bisect the flip boundary in [lo, hi]. Each probe folds into the Bank, so we keep the best-scoring
    # flip even if it is not the exact boundary (the score gate can prefer a slightly larger, deeper flip).
    for _ in range(max(0, K.BISECT_ITERS)):
        if hi - lo <= 1:
            break
        if ctx.time_left() <= 2.0 * ctx.t_step + ctx.t_eval:
            break
        mid = (lo + hi) // 2
        r = batch_eval(ctx, [_candidate(ctx, move_dir_flat, mid)])
        ctx.bank.consider(r)
        if r and r[0]["flipped"]:
            hi = mid
        else:
            lo = mid


# ==========================================================================================
# Public entry point.
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
    """Find an envelope-safe flip via a one-shot FGSM step + binary-search line search. Returns the
    clean image if none is found."""
    t_start = start_time if start_time is not None else time.time()
    reset_passes()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(K.MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))
    k_max = max(k_min, int(cap * 255.0 + 1e-6))
    q = k_min * K.Q

    clean_u8 = torch.round(clean.view(-1) * 255.0)
    envelope = K.TF32_ENVELOPE and device.type == "cuda"

    use_dynamic = K.DYNAMIC_KAPPA and envelope
    calib = get_calibrator(env_fingerprint(model, clean.shape)) if use_dynamic else None
    kappa = calib.global_kappa() if calib is not None else (K.KAPPA_RESID if envelope else K.MARGIN_BUFFER)

    # One gradient up front: clean margin m0 + boundary gradient g0, and a t_step timing gate.
    g_t0 = time.time()
    m0, g0 = margin_and_grad(model, clean, target_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = max(1e-4, time.time() - g_t0)

    if reserve_seconds is None:
        reserve_seconds = K.RESERVE_SECONDS
    reserve_seconds = float(reserve_seconds) + K.RESERVE_FWD_MULT * t_step
    if K.IGNORE_TIMEOUT:
        deadline = hard_deadline = t_start + 1e9
    else:
        deadline = hard_deadline = t_start + max(0.05, float(timeout_seconds) - reserve_seconds)

    def time_left() -> float:
        return ctx.deadline - time.time()

    ctx = Context(
        model=model, device=device, clean=clean, clean_u8=clean_u8, shape=clean.shape,
        target_index=target_index, k_min=k_min, q=q, floor=floor, cap=cap, kappa=kappa,
        skip_roundtrip=K.SKIP_ROUNDTRIP, tf32_on=K.TF32_ON, envelope=envelope,
        allow_unsafe=K.ALLOW_UNSAFE_FLIP, deadline=deadline, t_step=t_step, time_left=time_left,
        bank=Bank(), m0=m0, g0=g0.view(-1), dynamic_kappa=use_dynamic, hard_deadline=hard_deadline,
    )

    logger.info(f"[perturb-dev] start: m0={m0:.4f} k_min={k_min} k_max={k_max} kappa={kappa:.4f} "
                f"tf32={'on' if K.TF32_ON else 'off'} envelope={'on' if envelope else 'off'}")

    # Move direction: -sign(g0) reduces the CW margin. Restrict to channels that will not immediately
    # clip at the box edge (a clipped channel contributes nothing to the step).
    move_dir = -torch.sign(ctx.g0)
    move_dir[~movable(clean.view(-1), move_dir)] = 0.0

    _search(ctx, move_dir, k_max)

    chosen = ctx.bank.result(ctx.allow_unsafe)

    n_fwd, n_bwd = passes()
    if chosen is None:
        logger.info(f"[perturb-dev] no {'' if ctx.allow_unsafe else 'safe '}flip -> clean "
                    f"(m0={m0:.4f} elapsed={time.time() - t_start:.3f}s "
                    f"has_flip={ctx.bank.has_flip} fwd={n_fwd} bwd={n_bwd})")
        return clean.detach().clamp(0.0, 1.0)

    pct = 100.0 * chosen["nz"] / max(1, clean_u8.numel())
    logger.info(f"[perturb-dev] flip channels={chosen['nz']} ({pct:.2f}%) pixels={chosen.get('pixels', -1)} "
                f"margin={chosen['margin']:.4f} score={chosen.get('score', 0.0):.4f} "
                f"linf={chosen['linf']:.5f} rmse={chosen['rmse']:.5f} "
                f"elapsed={time.time() - t_start:.3f}s fwd={n_fwd} bwd={n_bwd}")
    return chosen["cand"].detach().clamp(0.0, 1.0)
