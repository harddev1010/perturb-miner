"""perturb.py — Cardinality-Continuation Ternary Search (integrated find + sparsify).

ONE engine that solves feasibility and L0-minimization together: every iteration asks "can I flip the
true class using at most K changed channels?", growing K to find the first verified flip and shrinking
it (geometric + bisection) to minimize the support — sparsity is optimized from the first iteration,
not bolted on as post-hoc pruning.

Building blocks (all on the exact uint8 byte grid, so every candidate is validator-faithful):
  * Phase-A seeding — cheap one-backward proposals: dense sign-gradient, a multi-scale support ladder
    with near-tie randomized orders, across a loss PORTFOLIO (hard/soft/DLR/CE) and the top wrong
    classes (targeted pair losses). Batched into one forward.
  * Fixed-K projected ternary search — APGD on a latent u, projected onto T_K = {S : S_i∈{-1,0,+1},
    |S|_0 ≤ K}; the projection simultaneously ADDS / REMOVES / REVERSES actions, and the gradient is
    RE-LINEARIZED at the current discrete state each step (the nonlinear escape the static clean
    gradient cannot provide). Every projected state is a legal candidate and is verified.
  * Cardinality continuation — grow K (×FEAS_GROW) until the first flip, then shrink (×CONT_ALPHA,
    bisecting against the largest failed budget). On stagnation, re-seed a fresh basin (Phase A).

TRANSFER SAFETY (unchanged from the package): the cuDNN-TF32 ambient regime (PERTURB_TF32_ON) plus a
worst-case TF32 envelope, and a residual-cushion kappa (PERTURB_KAPPA_RESID under the envelope, else
PERTURB_MINER_MARGIN_BUFFER) — a flip counts as "safe" only when its worst-case margin <= -kappa.

BUDGET: every loop gates on out_of_budget(ctx) (which reserves a backward + an eval chunk), and
batch_eval stops launching chunks before the deadline, so the search never runs past its time and the
Bank always holds a verified, in-band, quality candidate. Tuning is via PERTURB_* env vars in
constants.py — no redeploy needed.
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
    estimate_k,
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


# ==========================================================================================
# Candidate construction — gradients propose; the byte grid keeps everything legal.
# A "field" is a per-channel latent = move_dir · score: sign(field) is the toward-flip action and
# |field| = saliency |g|. A fixed-K projection of any field is an exact, legal ternary candidate.
# ==========================================================================================
def _field(ctx: Context, x: torch.Tensor, kind: str, targets: list[int]):
    """One attack loss at x -> (value, latent field), where latent = move_dir · |g| (toward-flip)."""
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
    """Π_{T_K}: the K legal coords of largest |latent|, each stepped ±k_min along sign(latent).
    This single op chooses which coords stay active, which become zero, which are added, and the sign."""
    mag, valid = _legal(ctx, latent)
    budget = min(max(1, budget), valid)
    d = torch.zeros_like(ctx.clean_u8)
    if budget < 1:
        return d
    idx = torch.topk(mag, budget).indices
    d[idx] = latent.sign()[idx] * float(ctx.k_min)
    return d


def _project_sampled(ctx: Context, latent: torch.Tensor, budget: int) -> torch.Tensor:
    """Near-tie variant: draw K coords from the top (TIE_MULT·K) by benefit ∝ softmax(|latent|/temp).
    Diversifies when many saliencies tie and the exact top-K ranking is unreliable, staying gradient-led."""
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
    """Linearized crossing size K̂ = min{K : q·Σ_{i<=K}|latent|_(i) >= value + kappa} (legal coords)."""
    mag, valid = _legal(ctx, latent)
    if valid == 0:
        return 0
    order = torch.argsort(mag, descending=True)
    return estimate_k(value + max(ctx.kappa, 0.0), mag[order][:valid], ctx.q)


def _support_ladder(ctx: Context, latent: torch.Tensor, value: float) -> list[torch.Tensor]:
    """A multi-scale support sweep around K̂: exact top-K plus randomized near-tie orders at each scale,
    and the fully dense legal support. Returns byte deltas (one candidate each)."""
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
# Evaluation + bank folding (the validator-faithful, envelope-aware, deadline-aware path).
# ==========================================================================================
def _eval(ctx: Context, deltas: list[torch.Tensor]) -> list[dict]:
    """Verify a batch of byte deltas (envelope worst-case margin, SSIM/PSNR, L∞ band), attach each
    delta to its result, and fold into the Bank. batch_eval may return a short prefix near the deadline."""
    deltas = [d for d in deltas if d is not None]
    if not deltas:
        return []
    cands = [apply_delta_bytes(ctx.clean_u8, d, ctx.shape) for d in deltas]
    res = batch_eval(ctx, cands)
    for r, d in zip(res, deltas):
        r["delta"] = d
    ctx.bank.consider(res)
    return res


def _incumbent(ctx: Context) -> dict | None:
    """The returnable incumbent: the smallest envelope-safe flip, or (when unsafe flips are allowed)
    the smallest margin<0 flip. None until a returnable flip exists."""
    if ctx.bank.best_safe is not None:
        return ctx.bank.best_safe
    return ctx.bank.best_flip if ctx.allow_unsafe else None


# ==========================================================================================
# Target pool + Phase-A seeding.
# ==========================================================================================
def _target_pool(ctx: Context, x: torch.Tensor):
    """Top wrong classes, each with one pair-loss field, prioritized by linearized crossing size K̂
    (smaller = easier to reach). Returns (pool entries, target-class list)."""
    logits = logits_of(ctx.model, x)
    targets = top_wrong_classes(logits, ctx.target_index, K.TOPM)
    pool: list[dict] = []
    for c in targets:
        if out_of_budget(ctx):
            break
        value, move_dir, score = loss_grad(ctx.model, x, ctx.target_index, f"pair:{c}")
        latent = move_dir * score
        if _legal(ctx, latent)[1] == 0:
            continue
        pool.append({"c": c, "value": value, "latent": latent, "kc": _kc_of(ctx, latent, value)})
    pool.sort(key=lambda e: e["kc"])
    return pool, targets


def _phase_a(ctx: Context, targets: list[int], pool: list[dict]) -> tuple[list[dict], int]:
    """Cheap one-backward seeds across the loss portfolio + smallest-K̂ targets, batched into the
    evaluator. Returns (results, smallest crossing-size estimate seen) — the latter seeds continuation."""
    deltas: list[torch.Tensor] = []
    min_kc = None
    for kind in K.LOSSES:
        if out_of_budget(ctx):
            break
        value, latent = _field(ctx, ctx.clean, kind, targets)
        deltas += _support_ladder(ctx, latent, value)
        kc = _kc_of(ctx, latent, value)
        min_kc = kc if min_kc is None else min(min_kc, kc)
    for e in pool[:K.POOL_SEEDS]:
        if out_of_budget(ctx):
            break
        deltas += _support_ladder(ctx, e["latent"], e["value"])
        min_kc = e["kc"] if min_kc is None else min(min_kc, e["kc"])
    res = _eval(ctx, deltas)
    return res, max(1, min_kc or 1)


def _collect_parents(res: list[dict]) -> list[torch.Tensor]:
    """Diverse near-flip seeds for the fixed-K searches: quality flips first (sparsest), then the
    lowest-margin near-misses. Their byte deltas become continuation starting points."""
    flips = sorted((r for r in res if r.get("quality")), key=lambda r: (r["nz"], r["margin"]))
    near = sorted((r for r in res if not r.get("quality")), key=lambda r: r["margin"])
    return [r["delta"] for r in (flips + near)[:K.PARENTS] if r.get("delta") is not None]


# ==========================================================================================
# Fixed-K projected ternary search (re-linearized APGD on a latent, projected onto T_K).
# ==========================================================================================
def _fixed_k(ctx: Context, seed_delta: torch.Tensor, budget: int, targets: list[int], li: int) -> None:
    """Search T_K for a flip starting from seed_delta: each step projects the latent to the budget-K
    legal support, VERIFIES that real candidate, then RE-LINEARIZES (recomputes the gradient at the
    new discrete state) and takes a momentum step. The projection adds/removes/reverses actions, so a
    bad seed coordinate can be dropped or flipped — not just padded. Banks every quality flip it sees."""
    km = float(ctx.k_min)
    u = (seed_delta / km).clone()           # latent seeded from the parent (ternary in {-1,0,+1})
    prev_u = u.clone()
    alpha = K.APGD_ALPHA0
    best_m = float("inf")
    stale = 0
    for _ in range(K.FIXED_ITERS):
        if out_of_budget(ctx):
            return
        d = _project_topk(ctx, u, budget)
        res = _eval(ctx, [d])
        if not res:
            return
        m = res[0]["margin"]
        if m < best_m - 1e-6:
            best_m, stale = m, 0
        else:
            stale += 1
            if stale >= K.APGD_PATIENCE:
                alpha = max(K.APGD_MIN_ALPHA, alpha * 0.5)
                stale = 0
        # Re-linearize at the current discrete state; normalize so the top saliency reaches |1| and can
        # overtake an existing ternary action (enabling replacement/reversal, not just addition).
        x = apply_delta_bytes(ctx.clean_u8, d, ctx.shape)
        _, latent = _field(ctx, x, K.LOSSES[li % len(K.LOSSES)], targets)
        li += 1
        latent = latent / (latent.abs().max() + 1e-12)
        new_u = (u + alpha * latent + K.APGD_MOMENTUM * (u - prev_u)).clamp(-K.U_CLAMP, K.U_CLAMP)
        prev_u, u = u, new_u


# ==========================================================================================
# Orchestrator — grow K to find, shrink K to minimize, re-seed on stagnation.
# ==========================================================================================
def search(ctx: Context) -> None:
    """Cardinality-continuation driver. Seeds Phase A, then loops fixed-K searches whose budget is grown
    until the first flip and shrunk afterwards (geometric, bisected against the largest failed budget),
    re-seeding a fresh basin (Phase A) when a budget window converges or the search stagnates. The Bank
    holds the smallest verified safe flip throughout."""
    n = ctx.clean_u8.numel()
    pool, targets = _target_pool(ctx, ctx.clean)
    if not targets:
        targets = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.TOPM)

    res, min_kc = _phase_a(ctx, targets, pool)
    parents = _collect_parents(res)

    feas_budget = max(1, min_kc)   # current budget while no flip exists (grows on failure)
    k_fail = 0                     # largest budget known to fail for the current incumbent/basin
    converge_fail = 0             # consecutive basins that produced no improvement
    li = it = 0

    while not out_of_budget(ctx):
        it += 1
        inc = _incumbent(ctx)

        if inc is None:
            budget = max(1, min(feas_budget, n))
        else:
            k_succ = inc["nz"]
            if k_succ <= 1 or k_fail + 1 >= k_succ:
                # The current support is minimal within the searched window. Try a different basin a
                # few times (a structurally different support may be smaller) before giving up.
                if converge_fail >= K.CONVERGE_PATIENCE:
                    break
                converge_fail += 1
                pool, targets = _target_pool(ctx, ctx.clean)
                res, _ = _phase_a(ctx, targets, pool)
                parents = (_collect_parents(res) + parents)[:K.PARENTS]
                k_fail = 0
                continue
            # Shrink geometrically, but never below the largest failed budget + 1 (bisection upper-bounds
            # the win so a single hard failure does not strand the search at an unreachable K).
            budget = max(k_fail + 1, min(k_succ - 1, int(round(k_succ * K.CONT_ALPHA))))

        before_nz = inc["nz"] if inc else 1 << 30
        parent = parents[it % len(parents)] if parents else torch.zeros_like(ctx.clean_u8)
        _fixed_k(ctx, parent, budget, targets, li)
        li += K.FIXED_ITERS

        after = _incumbent(ctx)
        after_nz = after["nz"] if after else 1 << 30
        if after_nz < before_nz:                        # a (smaller) flip appeared at this budget
            converge_fail = 0
            if after.get("delta") is not None:
                parents.insert(0, after["delta"])
                del parents[K.PARENTS:]
            if inc is None:
                feas_budget = max(1, after_nz)           # switch from growing to shrinking
        elif inc is None:                                # still no flip -> grow the feasibility budget
            if feas_budget >= n:
                converge_fail += 1
                if converge_fail >= K.CONVERGE_PATIENCE:
                    break
            feas_budget = min(n, int(round(feas_budget * K.FEAS_GROW)) + 1)
        else:                                            # had a flip; this smaller budget failed
            k_fail = max(k_fail, budget)

        # Periodically re-seed from clean for a structurally different basin (a fresh support family).
        if it % K.RESTART_EVERY == 0 and not out_of_budget(ctx):
            pool, targets = _target_pool(ctx, ctx.clean)
            res, kc = _phase_a(ctx, targets, pool)
            parents = (_collect_parents(res) + parents)[:K.PARENTS]
            if inc is None:
                feas_budget = max(1, min(feas_budget, kc))


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
    """Find the sparsest envelope-safe ±k_min/255 flip via integrated cardinality-continuation search
    (or any margin<0 flip when PERTURB_ALLOW_UNSAFE_FLIP=1); return the clean image if none is found
    within the deadline."""
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

    search(ctx)

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
