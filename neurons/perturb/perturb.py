"""perturb.py — three q=1 flip-finding approaches + the public entry point.

Only three strategies live here, toggled by commenting/uncommenting one line in perturb():

  1. find_apgd_dlr   — exact-byte APGD on the DLR loss over the full ternary {-1,0,+1} cube.
  2. find_dct_apgd   — low-frequency, filtered-gradient APGD-DLR (Method A: DCT-low-pass the gradient).
  3. find_hybrid     — the recommended combo: APGD-DLR -> DCT-APGD -> targeted-DLR repair -> RMSE prune.

Design: few backward passes, many batched forward checks. Each approach proposes candidate byte
perturbations, batch-evaluates them on the validator-faithful path (envelope + SSIM/PSNR + kappa),
and folds survivors into a shared Bank that tracks the sparsest envelope-safe flip. Every edit is an
exact integer ±k_min byte step on the uint8 grid, so L∞ stays pinned at q and the PNG round-trip is
identity (see neurons/perturb/utils.py and constants.py).

SWITCHES (no redeploy needed):
  * Approach: comment/uncomment the one line at the ORCHESTRATOR SWITCH in perturb().
  * Per-approach tuning + the accept gate: env vars in constants.py.
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
    apply_delta_bytes,
    batch_eval,
    logits_of,
    loss_grad,
    margin_and_grad,
    movable,
    top_wrong_classes,
)

logger = logging.getLogger(__name__)

# Match the validator's numeric regime byte-for-byte. The validator sets no backend flags, so it runs
# PyTorch defaults: cuDNN convolutions use TF32, matmul does NOT. EfficientNetV2-L is conv-dominated, so
# the TF32 axis that matters is cuDNN — we mirror it with PERTURB_TF32_ON (default: on for CUDA, off for
# CPU). matmul stays off (the validator default); the envelope brackets the cuDNN regime only.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = K.TF32_ON
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ==========================================================================================
# Shared helpers
# ==========================================================================================
def _status(ctx: Context) -> str | None:
    if ctx.bank.has_safe:
        return "safe"
    if ctx.bank.has_flip:
        return "flip"
    return None


def _out_of_time(ctx: Context) -> bool:
    return ctx.time_left() <= 2.0 * ctx.t_step


def _consider(ctx: Context, cands: list[torch.Tensor]) -> list[dict]:
    res = batch_eval(ctx, cands)
    ctx.bank.consider(res)
    return res


def _eval_deltas(ctx: Context, deltas: list[torch.Tensor]) -> list[dict]:
    """Eval a list of byte deltas; fold into the bank; return results with their delta attached."""
    if not deltas:
        return []
    cands = [apply_delta_bytes(ctx.clean_u8, d, ctx.shape) for d in deltas]
    res = _consider(ctx, cands)
    for r, d in zip(res, deltas):
        r["delta"] = d
    return res


def _mutate_ternary_latent(base: torch.Tensor, dev) -> torch.Tensor:
    """Random ternary mutation of a latent u: reset a small random subset to {-1,0,+1}."""
    n = base.numel()
    u = base.clone()
    m = torch.rand(n, device=dev) < 0.05
    u[m] = torch.randint(0, 3, (int(m.sum().item()),), device=dev).float() - 1.0
    return u


# ==========================================================================================
# 1. find_apgd_dlr — exact-byte APGD-DLR with batched restarts (full ternary cube).
# ==========================================================================================
def find_apgd_dlr(ctx: Context) -> str | None:
    """Quantized APGD-DLR with batched restarts (standalone approach).

    Auto-PGD on the DLR loss with every candidate projected onto the exact ternary byte cube. Latent
    u∈[-1,1] per channel; evaluated delta = round(u)·k_min. Starts in one batch: zero, CE-sign, DLR-sign,
    soft top-M sign, sparse random (1/5/20%), dense random. Each iter: eval the population, track the
    best REAL margin; if the best stalls APGD_PATIENCE times, halve the latent step and refresh the
    population around the best; otherwise take a DLR gradient step on the top-APGD_TOPK candidates
    (+ sign-only and mixed variants) and add ternary mutations around the best. Returns 'safe' on
    margin<=-kappa, else best margin<0. Logs the best margin as an infeasibility signal (high => the
    one-byte cube likely has no flip for this image)."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    top = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.APGD_TOPM)

    U = [torch.zeros(n, device=dev)]                                       # zero start
    _, md_ce, _ = loss_grad(ctx.model, ctx.clean, ctx.target_index, "ce")
    _, md_dlr, _ = loss_grad(ctx.model, ctx.clean, ctx.target_index, "dlr")
    _, md_soft, _ = loss_grad(ctx.model, ctx.clean, ctx.target_index, "soft", top_wrong=top, tau=K.APGD_TAU)
    U += [md_ce.clone(), md_dlr.clone(), md_soft.clone()]                   # gradient-sign starts
    for p in K.APGD_SPARSE_STARTS:                                         # sparse random starts
        u = torch.zeros(n, device=dev)
        m = torch.rand(n, device=dev) < p
        u[m] = torch.randint(0, 2, (int(m.sum().item()),), device=dev).float() * 2.0 - 1.0
        U.append(u)
    for _ in range(K.APGD_DENSE_STARTS):                                   # dense random starts
        U.append(torch.randint(0, 2, (n,), device=dev).float() * 2.0 - 1.0)

    best_margin = float("inf")
    best_u = None
    alpha = K.APGD_ALPHA0
    stale = 0
    iters = 0
    while not _out_of_time(ctx):
        if K.APGD_MAX_ITERS > 0 and iters >= K.APGD_MAX_ITERS:
            break
        iters += 1
        deltas = [u.round().clamp(-1.0, 1.0) * km for u in U]
        res = _eval_deltas(ctx, deltas)
        if ctx.bank.has_safe:
            return "safe"
        if not res:
            break
        bi = min(range(len(res)), key=lambda i: res[i]["margin"])
        if res[bi]["margin"] < best_margin - 1e-6:
            best_margin, best_u, stale = res[bi]["margin"], U[bi].clone(), 0
        else:
            stale += 1
            if stale >= K.APGD_PATIENCE:                                   # APGD stall -> shrink + refresh
                alpha = max(K.APGD_MIN_ALPHA, alpha * 0.5)
                stale = 0
                if best_u is not None:
                    U = [best_u.clone()] + [_mutate_ternary_latent(best_u, dev) for _ in range(len(U) - 1)]
                continue

        order = sorted(range(len(res)), key=lambda i: res[i]["margin"])[:K.APGD_TOPK]
        new_u: list[torch.Tensor] = []
        if best_u is not None:
            new_u.append(best_u.clone())
        for idx in order:
            if _out_of_time(ctx):
                break
            _, md, _ = loss_grad(ctx.model, res[idx]["cand"], ctx.target_index, "dlr")  # DLR attack dir
            new_u.append((U[idx] + alpha * md).clamp(-1.0, 1.0))
            new_u.append(md.clone())                                        # sign-only
            new_u.append((0.5 * U[idx] + 0.5 * md).clamp(-1.0, 1.0))        # mixed
        for _ in range(K.APGD_MUT):
            new_u.append(_mutate_ternary_latent(best_u if best_u is not None else U[bi], dev))
        U = new_u

    if not ctx.bank.has_flip:
        logger.info(f"[apgd_dlr] no flip; best_margin={best_margin:.4f} "
                    f"({'likely q=1 infeasible' if best_margin > 2.0 else 'near-miss'})")
    return _status(ctx)


# ==========================================================================================
# 2. find_dct_apgd — low-frequency filtered-gradient APGD-DLR (Method A).
# ------------------------------------------------------------------------------------------
# The search MAGNITUDE is still ±1 byte; only the search DIRECTION is restricted to smooth, coordinated
# low-frequency patterns. Each step low-pass-filters the DLR descent direction through a 2-D DCT (keep a
# top-left coefficient block, inverse-transform, take the sign) before the APGD update. Strong on images
# whose useful gradient energy is diffuse / low-frequency.
# ==========================================================================================
_DCT_CACHE: dict = {}


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix D[k,m] (rows = frequency k). Orthonormal => inverse is the transpose."""
    key = (n, str(device), dtype)
    M = _DCT_CACHE.get(key)
    if M is None:
        k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
        m = torch.arange(n, device=device, dtype=dtype).view(1, n)
        M = torch.cos(math.pi * (2.0 * m + 1.0) * k / (2.0 * n)) * math.sqrt(2.0 / n)
        M[0] *= 1.0 / math.sqrt(2.0)
        _DCT_CACHE[key] = M
    return M


def _dct_lowpass(x_chw: torch.Tensor, ratio: float) -> torch.Tensor:
    """2-D DCT low-pass per channel: keep the top-left ceil(ratio·H) × ceil(ratio·W) coefficient block."""
    c, h, w = int(x_chw.shape[0]), int(x_chw.shape[1]), int(x_chw.shape[2])
    dh = _dct_matrix(h, x_chw.device, x_chw.dtype)
    dw = _dct_matrix(w, x_chw.device, x_chw.dtype)
    coef = torch.einsum("kn,cnw->ckw", dh, x_chw)            # DCT along height
    coef = torch.einsum("lw,ckw->ckl", dw, coef)            # DCT along width
    rh = max(1, int(math.ceil(ratio * h)))
    rw = max(1, int(math.ceil(ratio * w)))
    coef[:, rh:, :] = 0.0                                    # mask high vertical frequencies
    coef[:, :, rw:] = 0.0                                    # mask high horizontal frequencies
    coef = torch.einsum("lw,ckl->ckw", dw, coef)            # IDCT along width
    return torch.einsum("kn,ckw->cnw", dh, coef)            # IDCT along height


def _dct_step_dir(ctx: Context, cand: torch.Tensor, ratio: float) -> torch.Tensor:
    """Filtered DLR descent direction: low-pass the toward-flip gradient, sign it, mask box-clipping moves."""
    _, move_dir, score = loss_grad(ctx.model, cand, ctx.target_index, "dlr")
    descent = (move_dir * score).view(ctx.shape)            # signed descent (reduces the DLR loss)
    low = _dct_lowpass(descent, ratio).reshape(-1)
    d = low.sign()
    d[~movable(ctx.clean.view(-1), d)] = 0.0                # drop directions that would clip at [0,1]
    return d


def _dct_sparse_probe(ctx: Context, ratio: float) -> None:
    """One-shot SPARSE top-k candidates from the low-frequency filtered clean gradient. Signing the whole
    filtered gradient activates ~every channel; instead keep only the top-k legal channels by filtered
    saliency over a k-ladder and stop at the first exact-byte flip — sparse-by-construction, not dense."""
    clean_flat = ctx.clean.view(-1)
    km = float(ctx.k_min)
    _, move_dir, score = loss_grad(ctx.model, ctx.clean, ctx.target_index, "dlr")
    descent = (move_dir * score).view(ctx.shape)
    low = _dct_lowpass(descent, ratio).reshape(-1)
    sign = low.sign()
    sal = low.abs()
    sal[~movable(clean_flat, sign)] = -1.0                  # legal, box-feasible channels only
    valid = int((sal > 0).sum().item())
    if valid == 0:
        return
    order = torch.argsort(sal, descending=True)
    cands = []
    for r in K.DCT_PREFIX_RATIOS:
        k = max(1, min(int(round(r * valid)), valid))
        d = torch.zeros_like(ctx.clean_u8)
        idx = order[:k]
        d[idx] = sign[idx] * km
        cands.append(d)
    _eval_deltas(ctx, cands)


def find_dct_apgd(ctx: Context) -> str | None:
    """Low-frequency filtered-gradient APGD-DLR (standalone approach).

    Identical APGD-DLR machinery to find_apgd_dlr, but every gradient step direction is first projected
    onto a retained low-frequency DCT subspace (Method A): DCT2 the descent vector, keep a top-left
    coefficient block, IDCT2, sign. Cycles the mask through DCT_MASK_RATIOS (1/8 -> 1/4 -> 3/8 by default)
    so it starts very smooth/global and widens toward medium detail. Latent u∈[-1,1], delta=round(u)·k_min,
    exact ±k_min byte edits => L∞ stays at q. Returns 'safe' on margin<=-kappa, else best margin<0."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    ratios = list(K.DCT_MASK_RATIOS) or [0.25]
    top = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.DCT_TOPM)

    _dct_sparse_probe(ctx, ratios[0])                                      # sparse top-k one-shot (cheap)
    if ctx.bank.has_safe:
        return "safe"

    U = [torch.zeros(n, device=dev)]                                       # zero start
    U.append(_dct_step_dir(ctx, ctx.clean, ratios[0]))                     # filtered DLR-sign start
    for p in K.DCT_SPARSE_STARTS:                                         # sparse random starts
        u = torch.zeros(n, device=dev)
        m = torch.rand(n, device=dev) < p
        u[m] = torch.randint(0, 2, (int(m.sum().item()),), device=dev).float() * 2.0 - 1.0
        U.append(u)
    for _ in range(K.DCT_DENSE_STARTS):                                   # dense random starts
        U.append(torch.randint(0, 2, (n,), device=dev).float() * 2.0 - 1.0)

    best_margin = float("inf")
    best_u = None
    alpha = K.DCT_ALPHA0
    stale = 0
    iters = 0
    while not _out_of_time(ctx):
        if K.DCT_MAX_ITERS > 0 and iters >= K.DCT_MAX_ITERS:
            break
        ratio = ratios[iters % len(ratios)]
        iters += 1
        deltas = [u.round().clamp(-1.0, 1.0) * km for u in U]
        res = _eval_deltas(ctx, deltas)
        if ctx.bank.has_safe:
            return "safe"
        if not res:
            break
        bi = min(range(len(res)), key=lambda i: res[i]["margin"])
        if res[bi]["margin"] < best_margin - 1e-6:
            best_margin, best_u, stale = res[bi]["margin"], U[bi].clone(), 0
        else:
            stale += 1
            if stale >= K.DCT_PATIENCE:                                    # APGD stall -> shrink + refresh
                alpha = max(K.DCT_MIN_ALPHA, alpha * 0.5)
                stale = 0
                if best_u is not None:
                    U = [best_u.clone()] + [_mutate_ternary_latent(best_u, dev) for _ in range(len(U) - 1)]
                continue

        order = sorted(range(len(res)), key=lambda i: res[i]["margin"])[:K.DCT_TOPK]
        new_u: list[torch.Tensor] = []
        if best_u is not None:
            new_u.append(best_u.clone())
        for idx in order:
            if _out_of_time(ctx):
                break
            d = _dct_step_dir(ctx, res[idx]["cand"], ratio)               # low-frequency DLR attack dir
            new_u.append((U[idx] + alpha * d).clamp(-1.0, 1.0))
            new_u.append(d.clone())                                        # sign-only
            new_u.append((0.5 * U[idx] + 0.5 * d).clamp(-1.0, 1.0))        # mixed
        for _ in range(K.DCT_MUT):
            new_u.append(_mutate_ternary_latent(best_u if best_u is not None else U[bi], dev))
        U = new_u

    if not ctx.bank.has_flip:
        logger.info(f"[dct_apgd] no flip; best_margin={best_margin:.4f} "
                    f"({'likely q=1 infeasible' if best_margin > 2.0 else 'near-miss'})")
    return _status(ctx)


# ==========================================================================================
# 3. find_hybrid — APGD-DLR -> DCT-APGD -> targeted-DLR repair -> RMSE prune.
# ------------------------------------------------------------------------------------------
# The recommended configuration: full APGD-DLR as the main finder, low-frequency APGD as the
# complementary structured search, a short targeted-DLR repair when a clear runner-up exists, and a
# post-success byte-pruning pass as the actual RMSE optimizer. Time is split by wall-clock fractions of
# the remaining budget; pruning always runs last on whatever flip the finders banked.
# ==========================================================================================
def _warm_delta(ctx: Context) -> torch.Tensor | None:
    """Integer byte delta (flat) of the Bank's best flip — safe preferred — or None if no flip yet."""
    best = ctx.bank.best_safe if ctx.bank.best_safe is not None else ctx.bank.best_flip
    if best is None:
        return None
    cand_u8 = torch.round(best["cand"].view(-1) * 255.0)
    return cand_u8 - ctx.clean_u8


def _accept(ctx: Context, r: dict) -> bool:
    """Is candidate r a valid working base to keep refining? Must be a QUALITY (in-band + SSIM/PSNR) flip
    that is also envelope-safe — or, with PERTURB_ALLOW_UNSAFE_FLIP=1, any quality flip. Requiring quality
    (not just margin-safe) stops the L0 stages from reverting past the in-band floor toward the clean image
    (which is 'safe' by margin but out of band). The Bank still independently records the global best."""
    if r.get("safe") and r.get("quality"):
        return True
    return bool(ctx.allow_unsafe and r.get("flipped") and r.get("quality"))


def _targeted_repair(ctx: Context) -> str | None:
    """Phase 3: short targeted Auto-PGD on the targeted-DLR loss against the top runner-up classes.

    Most useful when one competitor is naturally close. Latent u∈[-1,1] per target, momentum + adaptive
    step + restart-from-best, all batched. delta=round(u)·k_min (exact byte edits). Returns 'safe' on
    margin<=-kappa, else the best margin<0 in its slice of the budget."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    top = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index,
                            max(K.HYBRID_TARGETS, K.DCT_TOPM))
    kinds = [f"dlrt:{t}" for t in top[:max(1, K.HYBRID_TARGETS)]]
    states = [{"u": torch.zeros(n, device=dev), "prev": torch.zeros(n, device=dev),
               "alpha": K.APGD_ALPHA0, "best": float("inf"), "stale": 0,
               "ubest": torch.zeros(n, device=dev), "kind": k} for k in kinds]

    while not _out_of_time(ctx):
        deltas = [s["u"].round().clamp(-1.0, 1.0) * km for s in states]
        res = _eval_deltas(ctx, deltas)
        if ctx.bank.has_safe:
            return "safe"
        if not res:
            break
        for s, r in zip(states, res):
            if r["margin"] < s["best"] - 1e-6:
                s["best"], s["stale"], s["ubest"] = r["margin"], 0, s["u"].clone()
            else:
                s["stale"] += 1
                if s["stale"] >= K.APGD_PATIENCE:            # halve step + restart from this state's best
                    s["alpha"] = max(K.APGD_MIN_ALPHA, s["alpha"] * 0.5)
                    s["stale"], s["u"] = 0, s["ubest"].clone()
        for s, r in zip(states, res):
            if _out_of_time(ctx):
                break
            _, move_dir, _ = loss_grad(ctx.model, r["cand"], ctx.target_index, s["kind"])
            z = (s["u"] + s["alpha"] * move_dir).clamp(-1.0, 1.0)            # APGD step (sign direction)
            u_new = (z + K.HYBRID_MOMENTUM * (s["u"] - s["prev"])).clamp(-1.0, 1.0)  # momentum
            s["prev"], s["u"] = s["u"], u_new
    return _status(ctx)


# ------------------------------------------------------------------------------------------
# compress_l0 — exact-byte L0 continuation (THE RMSE optimizer).
# ------------------------------------------------------------------------------------------
# At q=1 every active channel has magnitude exactly one byte, so RMSE = q·sqrt(|S|/n) and the real
# objective is min |S| s.t. the flip holds — NOT "find any flip in the L∞ box" (what the finders do).
# This stage keeps OPTIMIZING THE SUPPORT itself rather than only pruning the support the finders
# happened to land on, escaping the subset trap that plateaus plain backward pruning. Each round, under
# a LOCKED target margin h_t = z_y − z_t (stable support ranking, vs untargeted DLR whose argmax/denom
# switch), it runs:
#   A reinforce the margin at fixed |S| (re-sign + swap weak↔strong) to create deletion slack;
#   B adaptively shrink the cardinality budget k (cheap-subset drop + a fresh relocate candidate);
#   C slack-aware group pruning (revert a whole low-cost group in one forward, β-budgeted by the slack);
#   D one-for-many support exchange (add 1 strong unused channel, drop ≥2 weak) — escapes the subset trap;
#   E exact batched leave-one-out cleanup for the final, uncertain channels.
# Candidates flow through the validator-faithful envelope eval, so the Bank independently records the
# sparsest envelope-safe flip; gradients only RANK/PREDICT. (Our delta is over d=3HW individual channel-
# values, so every top-k op is channel-sparse — never auto-flips all 3 channels of a pixel.)
# ------------------------------------------------------------------------------------------
def _l0_pair(ctx: Context, delta: torch.Tensor, t: int):
    """Locked-target margin h_t=z_y−z_t, its toward-flip move dir, and |∂h_t/∂x| at clean+delta."""
    x = apply_delta_bytes(ctx.clean_u8, delta, ctx.shape)
    return loss_grad(ctx.model, x, ctx.target_index, f"pair:{t}")


def _revert_cost(move_dir, score, delta, idx, km, q) -> torch.Tensor:
    """First-order Δh_t from reverting each channel in idx one byte toward clean. Negative => reverting
    also helps the flip (do it first); positive => it costs margin slack."""
    return move_dir[idx] * score[idx] * torch.sign(delta[idx]) * km * q


def _l0_pick(ctx: Context, res: list[dict], max_k: int, require_accept: bool) -> dict | None:
    """Lowest hard-margin (envelope) candidate that still flips, with |S|<=max_k and (optionally) safe."""
    best = None
    for r in res:
        if not r.get("flipped") or r["nz"] > max_k:
            continue
        if require_accept and not _accept(ctx, r):
            continue
        if best is None or r["margin"] < best["margin"]:
            best = r
    return best


def _l0_reinforce(ctx: Context, delta: torch.Tensor, t: int, km: float) -> torch.Tensor:
    """A: at FIXED |S|, re-sign active channels to the toward-flip direction and swap the weakest active
    channels for the strongest unused ones, keeping the equal-k candidate with the most negative margin.
    Does not shrink |S| — it manufactures the slack that lets later stages delete."""
    clean_flat = ctx.clean.view(-1)
    _, move_dir, score = _l0_pair(ctx, delta, t)
    active = delta != 0
    k = int(active.sum().item())
    cands = [delta]                                            # keep the current base in the race
    d = torch.zeros_like(delta)                               # (1) re-sign all active to toward-flip
    d[active] = move_dir[active] * km
    cands.append(d)
    act_idx = active.nonzero(as_tuple=True)[0]
    if act_idx.numel() > 0:                                    # (2) swap weak active <-> strong unused
        weak = act_idx[torch.argsort(score[act_idx])]
        gabs = score.clone()
        gabs[active] = -1.0
        gabs[~movable(clean_flat, move_dir)] = -1.0
        n_un = int((gabs > 0).sum().item())
        for frac in K.L0_SWAP_FRACS:
            cnt = min(int(frac * act_idx.numel()) + 1, int(weak.numel()), n_un)
            if cnt < 1:
                continue
            strong = torch.topk(gabs, cnt).indices
            d = delta.clone()
            d[weak[:cnt]] = 0.0
            d[strong] = move_dir[strong] * km
            cands.append(d)
    res = _eval_deltas(ctx, cands)
    pick = _l0_pick(ctx, res, max_k=k, require_accept=False)   # equal-k, just want more slack
    return pick["delta"] if pick is not None else delta


def _l0_shrink(ctx: Context, delta: torch.Tensor, t: int, km: float, q: float, rho: float):
    """B: try a smaller cardinality budget k_try=floor(ρ·k). Two candidates: drop the cheapest (k−k_try)
    active channels (subset), and a FRESH top-k_try support from the gradient (relocate — can leave the
    current support entirely). Returns (delta, True) on an accepted strict shrink, else (delta, False)."""
    clean_flat = ctx.clean.view(-1)
    active = delta != 0
    k = int(active.sum().item())
    k_try = max(1, int(math.floor(rho * k)))
    if k_try >= k:
        return delta, False
    _, move_dir, score = _l0_pair(ctx, delta, t)
    act_idx = active.nonzero(as_tuple=True)[0]
    cost = _revert_cost(move_dir, score, delta, act_idx, km, q)
    order = act_idx[torch.argsort(cost)]                       # cheapest-to-remove first
    cand_subset = delta.clone()
    cand_subset[order[: k - k_try]] = 0.0                      # keep the k_try most-expensive-to-remove
    gabs = score.clone()
    gabs[~movable(clean_flat, move_dir)] = -1.0
    n_valid = max(1, int((gabs > 0).sum().item()))
    topk = torch.topk(gabs, min(k_try, n_valid)).indices
    cand_fresh = torch.zeros_like(delta)                       # relocate: fresh top-k_try support
    cand_fresh[topk] = move_dir[topk] * km
    res = _eval_deltas(ctx, [cand_subset, cand_fresh])
    pick = _l0_pick(ctx, res, max_k=k - 1, require_accept=True)
    return (pick["delta"], True) if pick is not None else (delta, False)


def _l0_group_prune(ctx: Context, delta: torch.Tensor, t: int, q: float) -> torch.Tensor:
    """C: slack-aware group deletion. Rank active channels by revert cost (ascending), batch-eval a
    geometric ladder of removal counts (+ the linear-predicted safe prefix) in ONE forward, and take the
    LARGEST removal that stays accept-safe. Repeat to a local minimum or the budget."""
    km = float(ctx.k_min)
    base = float(K.PRUNE_LADDER) if K.PRUNE_LADDER > 1 else 2.0
    cur = delta
    while not _out_of_time(ctx):
        changed = (cur != 0).nonzero(as_tuple=True)[0]
        if changed.numel() == 0:
            break
        val, move_dir, score = _l0_pair(ctx, cur, t)
        if val >= 0.0:                                        # locked target no longer winning -> stop here
            break
        c = _revert_cost(move_dir, score, cur, changed, km, q)
        sort_idx = torch.argsort(c)
        order = changed[sort_idx]
        n = int(order.numel())
        sizes = set()
        s = 1
        while s < n:
            sizes.add(s)
            s = max(s + 1, int(s * base))
        sizes.add(n)
        cum = torch.cumsum(c[sort_idx], dim=0)                # predicted h_t after reverting each prefix
        ok = (val + cum < -ctx.kappa)
        bpred = int(torch.cumprod(ok.to(torch.long), dim=0).sum().item())
        if bpred >= 1:
            sizes.add(bpred)
        sizes = sorted(x for x in sizes if 1 <= x <= n)
        cands = [cur.clone() for _ in sizes]
        for d, sz in zip(cands, sizes):
            d[order[:sz]] = 0.0
        res = _eval_deltas(ctx, cands)
        best_d, best_sz = None, 0
        for sz, d, r in zip(sizes, cands, res):
            if _accept(ctx, r) and sz > best_sz:
                best_sz, best_d = sz, d
        if best_d is None:
            break
        cur = best_d
    return cur


def _l0_exchange(ctx: Context, delta: torch.Tensor, t: int, km: float, q: float) -> torch.Tensor:
    """D: one-for-many support exchange. Add ONE strong unused channel j (its toward-flip step drops h_t
    by ~q·km·|g_j|) and revert as many cheap active channels as that extra slack pays for, so |S| strictly
    drops. The added channel may lie OUTSIDE the current support, so this escapes the subset trap that
    bounds pure backward pruning. Keeps the accept-safe candidate with the lowest |S|; repeats."""
    clean_flat = ctx.clean.view(-1)
    mindrop = max(2, int(K.L0_EXCHANGE_MIN_DROP))
    cur = delta
    while not _out_of_time(ctx):
        val, move_dir, score = _l0_pair(ctx, cur, t)
        if val >= 0.0:
            break
        changed_mask = cur != 0
        changed = changed_mask.nonzero(as_tuple=True)[0]
        if changed.numel() == 0:
            break
        c = _revert_cost(move_dir, score, cur, changed, km, q)
        c_order = torch.argsort(c)
        rem_order = changed[c_order]
        cum = torch.cumsum(c[c_order], dim=0)
        gabs = score.clone()                                  # strongest unused, movable channels to add
        gabs[changed_mask] = -1.0
        gabs[~movable(clean_flat, move_dir)] = -1.0
        n_add = min(int(K.L0_EXCHANGE_ADDS), int((gabs > 0).sum().item()))
        if n_add < 1:
            break
        add_idx = torch.topk(gabs, n_add).indices.tolist()
        slack = -ctx.kappa - val                              # head-room of the locked-target margin (>=0)

        def _nmax(j):                                         # cheapest removals the add's slack can pay for
            budget = slack + q * km * float(score[j].item())
            return int((cum <= budget).sum().item())

        cands = []
        j0 = add_idx[0]                                       # strongest add: a removal-count ladder
        top = _nmax(j0)
        if top >= mindrop:
            sizes, s = set(), mindrop
            while s < top:
                sizes.add(s)
                s = max(s + 1, int(s * 2))
            sizes.add(top)
            for sz in sorted(sizes):
                d = cur.clone()
                d[rem_order[:sz]] = 0.0
                d[j0] = move_dir[j0] * km
                cands.append(d)
        for j in add_idx[1:]:                                 # other adds: one candidate each (diversity)
            nrem = _nmax(j)
            if nrem < mindrop:
                continue
            d = cur.clone()
            d[rem_order[:nrem]] = 0.0
            d[j] = move_dir[j] * km
            cands.append(d)
        if not cands:
            break
        res = _eval_deltas(ctx, cands)
        best_d, best_nz = None, int(changed.numel())
        for d, r in zip(cands, res):
            if _accept(ctx, r) and r["nz"] < best_nz:
                best_nz, best_d = r["nz"], d
        if best_d is None:
            break
        cur = best_d
    return cur


def _l0_loo(ctx: Context, delta: torch.Tensor, t: int) -> torch.Tensor:
    """E: exact batched leave-one-out cleanup (only when |S| is small enough to be worth exact probes).
    Eval every single-channel revert, collect the ones that stay accept-safe, then take the LARGEST
    accept-safe cumulative group of them (most-slack-first ladder) — catches curvature / target switches
    the first-order ranking misses."""
    changed = (delta != 0).nonzero(as_tuple=True)[0]
    n = int(changed.numel())
    if n == 0 or n > K.L0_LOO_MAX:
        return delta
    cands = [delta.clone() for _ in range(n)]
    idx_list = changed.tolist()
    for d, i in zip(cands, idx_list):
        d[i] = 0.0
    res = _eval_deltas(ctx, cands)                            # exact leave-one-out
    rem = [(r["margin"], i) for i, r in zip(idx_list, res) if _accept(ctx, r)]
    if not rem:
        return delta
    rem.sort(key=lambda z: z[0])                              # most slack (most negative margin) first
    order = torch.tensor([i for _, i in rem], device=delta.device)
    m = len(rem)
    sizes, s = set(), 1
    while s < m:
        sizes.add(s)
        s *= 2
    sizes.add(m)
    sizes = sorted(sizes)
    cands2 = [delta.clone() for _ in sizes]
    for d, sz in zip(cands2, sizes):
        d[order[:sz]] = 0.0
    res2 = _eval_deltas(ctx, cands2)
    best_d, best_sz = None, 0
    for sz, d, r in zip(sizes, cands2, res2):
        if _accept(ctx, r) and sz > best_sz:
            best_sz, best_d = sz, d
    return best_d if best_d is not None else delta


def compress_l0(ctx: Context) -> str | None:
    """Phase 4 / RMSE optimizer: exact-byte L0 continuation (see block comment above).

    Warm-starts from the Bank's best flip and keeps optimizing the SUPPORT — reinforce (A) -> adaptive
    shrink (B) -> slack-aware group prune (C) -> one-for-many exchange (D) -> exact leave-one-out (E) —
    under a target re-locked to the current flip each round, banking every sparser envelope-safe candidate,
    until a full round makes no net progress or the budget runs out. ρ (the shrink budget) tightens after a
    successful shrink and backs off after a failure (FMN-style adaptive cardinality radius)."""
    delta = _warm_delta(ctx)
    if delta is None:
        return _status(ctx)
    km, q = float(ctx.k_min), ctx.q
    rho = K.L0_RHO0
    while not _out_of_time(ctx):
        k0 = int((delta != 0).sum().item())
        x = apply_delta_bytes(ctx.clean_u8, delta, ctx.shape)       # re-lock target at the current flip
        t = top_wrong_classes(logits_of(ctx.model, x), ctx.target_index, 1)[0]
        delta = _l0_reinforce(ctx, delta, t, km)                    # A
        d, ok = _l0_shrink(ctx, delta, t, km, q, rho)               # B
        if ok:
            delta = d
            rho = max(K.L0_RHO_MIN, rho - K.L0_RHO_STEP)            # success -> shrink harder next time
        else:
            rho = min(K.L0_RHO_MAX, rho + K.L0_RHO_STEP)            # failure -> back off
        delta = _l0_group_prune(ctx, delta, t, q)                   # C
        delta = _l0_exchange(ctx, delta, t, km, q)                  # D
        delta = _l0_loo(ctx, delta, t)                              # E
        if int((delta != 0).sum().item()) >= k0:                    # no net |S| progress this round
            break
    return _status(ctx)


def _run_phase(ctx: Context, fn, end_time: float) -> None:
    """Run a finder with ctx.time_left temporarily capped at `end_time` (never past the real deadline)."""
    full = ctx.time_left
    ctx.time_left = lambda: min(full(), end_time - time.time())
    try:
        fn(ctx)
    finally:
        ctx.time_left = full


def find_hybrid(ctx: Context) -> str | None:
    """Recommended hybrid (standalone approach): exact-byte APGD-DLR + filtered-gradient DCT-APGD +
    targeted-DLR repair + post-success byte pruning.

    Phase 1 (~HYBRID_APGD_FRAC of the budget): full APGD-DLR searches the whole q=1 cube.
    Phase 2 (~HYBRID_DCT_FRAC): low-frequency DCT-APGD searches smooth coordinated directions.
    Phase 3 (~HYBRID_REPAIR_FRAC): short targeted-DLR repair when a clear runner-up exists.
    Phase 4 (remainder): exact-byte L0 continuation (compress_l0) shrinks |S| of whatever flip was banked.
    Any phase that finds an envelope-safe flip short-circuits straight to compression, which runs against
    the real deadline (the per-phase cap is lifted) — after the first flip, nearly all time goes to L0."""
    full = ctx.time_left
    total = max(0.0, full())
    t0 = time.time()
    acc = 0.0
    for fn, frac in ((find_apgd_dlr, K.HYBRID_APGD_FRAC),
                     (find_dct_apgd, K.HYBRID_DCT_FRAC),
                     (_targeted_repair, K.HYBRID_REPAIR_FRAC)):
        if ctx.bank.has_safe or _out_of_time(ctx):
            break
        acc += frac
        _run_phase(ctx, fn, t0 + total * acc)

    if ctx.bank.has_flip:                                     # Phase 4: L0 compression to the real deadline
        compress_l0(ctx)
    return _status(ctx)


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
    """Find a sparse ±1/255 flip with one of the three q=1 approaches. Returns the sparsest
    envelope-safe candidate (or, when PERTURB_ALLOW_UNSAFE_FLIP=1, any flip), else the clean image."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(K.MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))  # fixed unit step (typically 1)
    q = k_min * K.Q

    if reserve_seconds is None:
        reserve_seconds = K.RESERVE_SECONDS
    deadline = t_start + max(0.05, float(timeout_seconds) - float(reserve_seconds))

    def time_left() -> float:
        return deadline - time.time()

    # Byte-space: snap clean to its uint8 grid; every edit is an integer BYTE step on this, so
    # candidates are exactly on the k/255 grid and the PNG round-trip is identity.
    clean_u8 = torch.round(clean.view(-1) * 255.0)
    envelope = K.TF32_ENVELOPE and device.type == "cuda"
    kappa = K.KAPPA_RESID if envelope else K.MARGIN_BUFFER

    # One gradient evaluation: clean margin m0 + boundary gradient g0, and a t_step to gate loops.
    g_t0 = time.time()
    m0, g0 = margin_and_grad(model, clean, target_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = max(1e-4, time.time() - g_t0)

    ctx = Context(
        model=model, device=device, clean=clean, clean_u8=clean_u8, shape=clean.shape,
        target_index=target_index, k_min=k_min, q=q, floor=floor, cap=cap, kappa=kappa,
        skip_roundtrip=K.SKIP_ROUNDTRIP, tf32_on=K.TF32_ON, envelope=envelope,
        allow_unsafe=K.ALLOW_UNSAFE_FLIP, deadline=deadline, t_step=t_step, time_left=time_left,
        bank=Bank(), m0=m0, g0=g0.view(-1),
    )

    # --- ORCHESTRATOR SWITCH: uncomment exactly ONE of the three approaches. ---
    # find_apgd_dlr(ctx)     # 1. exact-byte APGD-DLR over the full ternary cube
    # find_dct_apgd(ctx)     # 2. low-frequency filtered-gradient DCT APGD
    find_hybrid(ctx)         # 3. recommended hybrid: APGD-DLR -> DCT-APGD -> targeted repair -> RMSE prune

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
