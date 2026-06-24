"""perturb.py — Problem 1 (feasibility): Multi-Target Ternary Projected Gradient Search.

ONE solver, `find_feasible`: find ANY byte-valid ternary perturbation S ∈ {-1,0,+1}^N (with
0 ≤ A_i+S_i ≤ 255) that makes the validator flip the true class, then return immediately. Cost (L0 /
RMSE) is irrelevant for feasibility, so there is no compression stage — the first envelope-safe,
in-band, quality flip wins.

White-box strategy (the model is open and gradients are available):
  * gradients PROPOSE actions and supports;
  * an exact ternary projection keeps every candidate legal (round → clamp → integer byte step);
  * REAL discrete candidates are batch-evaluated on the validator-faithful path (envelope + SSIM/PSNR);
  * gradients are recomputed at the current discrete state when the local model goes stale;
  * randomized ternary mutation rescues the search when gradients stall.

Stages (see the analysis):
  1. Calibration / reserve   — handled by the deadline (RESERVE_SECONDS) and the t_step budget gate.
  2. Dense one-step          — the ε=1-byte sign-gradient flip for the soft-margin and DLR losses.
  3. Top-k support sweep     — prefix supports around the linearized crossing size K_c, per target.
  4. Ternary projected GD    — APGD on a latent u∈[-1,1]; evaluated delta = round(u)·k_min.
  5. Macro coordinate descent— block-coordinate children (add/remove/reverse the top-B actions).
  6. Gradient-free rescue    — randomized ternary mutation around the best when gradients stall.
  7. Exact success handling  — return on the first envelope-safe flip (the Bank holds it).

Every edit is an exact integer ±k_min byte step on the uint8 grid, so L∞ stays pinned at q and the
PNG round-trip is identity (see neurons/perturb/utils.py and constants.py). Tuning is via PERTURB_FEAS_*
env vars in constants.py — no redeploy needed.
"""

from __future__ import annotations

import logging
import math
import random
import time

import torch
import torch.nn.functional as F

from . import constants as K
from .utils import (
    Bank,
    Context,
    apply_delta_bytes,
    batch_eval,
    build_sparse_order,
    estimate_k,
    logits_of,
    loss_grad,
    margin_and_grad,
    movable,
    out_of_budget,
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
    return out_of_budget(ctx)


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


def _mutate_ternary_latent(base: torch.Tensor, dev, p: float = 0.05) -> torch.Tensor:
    """Random ternary mutation of a latent u: reset a small random subset to {-1,0,+1}."""
    n = base.numel()
    u = base.clone()
    m = torch.rand(n, device=dev) < p
    u[m] = torch.randint(0, 3, (int(m.sum().item()),), device=dev).float() - 1.0
    return u


# ==========================================================================================
# Candidate construction (gradients propose; the byte grid keeps everything legal)
# ==========================================================================================
def _dense_action(ctx: Context, move_dir: torch.Tensor) -> torch.Tensor:
    """The ε=1-byte sign-gradient candidate: every movable channel stepped one byte toward the flip.

    This is the exact best first-order candidate when sparsity does not matter, and it costs only the
    one backward pass that produced `move_dir` — the highest-value first probe for feasibility."""
    clean_flat = ctx.clean.view(-1)
    mask = movable(clean_flat, move_dir)
    d = torch.zeros_like(ctx.clean_u8)
    d[mask] = move_dir[mask] * float(ctx.k_min)
    return d


def _supports_around_k(ctx: Context, move_dir: torch.Tensor, score: torch.Tensor,
                       value: float, mults) -> list[torch.Tensor]:
    """Top-k support sweep along descending benefit b_i = |g_i| (movable channels only).

    The linearized minimum number of unit actions to cross the y-vs-c boundary is
    K = min{k : q·Σ_{r≤k} b_(r) ≥ value + κ}. We emit prefix supports at several multiples of K
    (a medium support often succeeds when the fully dense candidate destructively interferes) plus the
    full positive-benefit support."""
    clean_flat = ctx.clean.view(-1)
    km = float(ctx.k_min)
    masked, order, valid = build_sparse_order(score, move_dir, clean_flat)
    if valid == 0:
        return []
    sorted_score = masked[order][:valid]
    kc = estimate_k(value + ctx.kappa, sorted_score, ctx.q)
    ks = {int(round(f * kc)) for f in mults}
    ks.add(valid)                                            # full positive-benefit support
    deltas: list[torch.Tensor] = []
    for k in sorted(k for k in ks if k >= 1):
        k = min(k, valid)
        idx = order[:k]
        d = torch.zeros_like(ctx.clean_u8)
        d[idx] = move_dir[idx] * km
        deltas.append(d)
    return deltas


def _block_children(ctx: Context, u: torch.Tensor, move_dir: torch.Tensor,
                    score: torch.Tensor, blocks) -> list[torch.Tensor]:
    """Macro coordinate descent (Stage 5): from a discrete parent S=round(u), switch the top-B coordinates
    to their toward-flip action move_dir_i, for a geometric ladder of block sizes B.

    Because move_dir is the best legal action regardless of the current S_i, a switch may ADD an action,
    REMOVE one (S_i≠0 → toward-clean is never proposed here, but a reverse is), or REVERSE +1↔-1 — so an
    add-only search cannot get trapped once the gradient direction changes. Ranked by predicted gain
    q_i = score_i for coordinates not already at move_dir_i and not box-clipped."""
    clean_flat = ctx.clean.view(-1)
    s = u.round().clamp(-1.0, 1.0)
    gain = score.clone()
    gain[~movable(clean_flat, move_dir)] = -1.0             # box-clipped moves buy nothing
    gain[s == move_dir] = -1.0                              # already at the toward-flip action
    order = torch.argsort(gain, descending=True)
    pos = int((gain > 0).sum().item())
    children: list[torch.Tensor] = []
    for b in blocks:
        b = min(int(b), pos)
        if b < 1:
            continue
        d = s.clone()
        idx = order[:b]
        d[idx] = move_dir[idx]
        children.append(d)
    return children


# ==========================================================================================
# find_feasible — the Problem 1 solver
# ==========================================================================================
def _seed_grad(ctx: Context, x: torch.Tensor, kind: str, targets: list[int]):
    """Gradient of one attack loss at x: (value, toward-flip move_dir, saliency |g|)."""
    if kind == "soft":
        return loss_grad(ctx.model, x, ctx.target_index, "soft", top_wrong=targets, tau=K.FEAS_TAU)
    return loss_grad(ctx.model, x, ctx.target_index, kind)


def find_feasible(ctx: Context) -> str | None:
    """Multi-Target Ternary Projected Gradient Search — find any envelope-safe q=1 flip, then stop.

    Seeds with the cheap one-backward candidates (dense sign-gradient + top-k support sweeps for the
    untargeted soft/DLR losses and the top runner-up classes), then runs an adaptive ternary-projected
    population search: each iteration evaluates the projected population, tracks the best REAL margin,
    halves the latent step and refreshes around the best on a stall, and otherwise takes a projected
    gradient step (+ sign-only + block-coordinate children) on the lowest-margin parents, recomputing the
    gradient at each discrete state. Returns 'safe' the instant an envelope-safe flip is banked; else the
    best margin<0 if any; else logs the best margin reached as an infeasibility signal."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    targets = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.FEAS_TOPM)

    # --- Stages 2-3: fast one-backward seeds (dense + top-k sweeps) -------------------------
    U = [torch.zeros(n, device=dev)]                                       # zero start
    for kind in K.FEAS_LOSSES:                                             # untargeted soft-margin + DLR
        _, move_dir, score = _seed_grad(ctx, ctx.clean, kind, targets)
        _eval_deltas(ctx, [_dense_action(ctx, move_dir)])
        if ctx.bank.has_safe:
            return "safe"
        _eval_deltas(ctx, _supports_around_k(ctx, move_dir, score, ctx.m0, K.FEAS_K_MULTS))
        if ctx.bank.has_safe:
            return "safe"
        U.append(move_dir.clone())

    for c in targets:                                                      # multi-target pair candidates
        if _out_of_time(ctx):
            break
        value, move_dir, score = loss_grad(ctx.model, ctx.clean, ctx.target_index, f"pair:{c}")
        _eval_deltas(ctx, _supports_around_k(ctx, move_dir, score, value, K.FEAS_K_MULTS))
        if ctx.bank.has_safe:
            return "safe"
        U.append(move_dir.clone())

    for p in K.FEAS_SPARSE_STARTS:                                         # sparse random starts
        u = torch.zeros(n, device=dev)
        m = torch.rand(n, device=dev) < p
        u[m] = torch.randint(0, 2, (int(m.sum().item()),), device=dev).float() * 2.0 - 1.0
        U.append(u)
    for _ in range(K.FEAS_DENSE_STARTS):                                   # dense random starts
        U.append(torch.randint(0, 2, (n,), device=dev).float() * 2.0 - 1.0)

    # --- Stages 4-6: adaptive ternary-projected search -------------------------------------
    best_margin = float("inf")
    best_u = None
    alpha = K.FEAS_ALPHA0
    stale = 0
    li = 0
    while not _out_of_time(ctx):
        deltas = [u.round().clamp(-1.0, 1.0) * km for u in U]              # exact ternary projection
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
            if stale >= K.FEAS_PATIENCE:                                   # APGD stall -> shrink + refresh
                alpha = max(K.FEAS_MIN_ALPHA, alpha * 0.5)
                stale = 0
                if best_u is not None:
                    U = [best_u.clone()] + [_mutate_ternary_latent(best_u, dev) for _ in range(len(U) - 1)]
                continue

        order = sorted(range(len(res)), key=lambda i: res[i]["margin"])[:K.FEAS_TOPK]
        kind = K.FEAS_LOSSES[li % len(K.FEAS_LOSSES)]                      # rotate the attack loss
        li += 1
        new_u: list[torch.Tensor] = []
        if best_u is not None:
            new_u.append(best_u.clone())
        for idx in order:
            if _out_of_time(ctx):
                break
            _, move_dir, score = _seed_grad(ctx, res[idx]["cand"], kind, targets)  # grad at this state
            new_u.append((U[idx] + alpha * move_dir).clamp(-1.0, 1.0))     # Stage 4: projected GD step
            new_u.append(move_dir.clone())                                 # sign-only
            new_u += _block_children(ctx, U[idx], move_dir, score, K.FEAS_BLOCKS)  # Stage 5
        for _ in range(K.FEAS_MUT):                                        # Stage 6: gradient-free rescue
            new_u.append(_mutate_ternary_latent(best_u if best_u is not None else U[bi], dev))
        U = new_u

    if not ctx.bank.has_flip:
        logger.info(f"[feasible] no flip; best_margin={best_margin:.4f} "
                    f"({'likely q=1 infeasible' if best_margin > 2.0 else 'near-miss'})")
    return _status(ctx)


# ==========================================================================================
# find_feasible_upgraded — portfolio + diverse beam + adaptive blocks (Problem 1, upgraded)
# ------------------------------------------------------------------------------------------
# A strict superset of find_feasible. Same contract (return on the first envelope-safe flip, cost
# irrelevant), but every knob the analysis flagged as high-impact is in play: a LOSS PORTFOLIO with
# reward-per-second allocation, a target pool prioritized by the linearized crossing size K̂_c (rebuilt
# when the best wrong class changes), a DIVERSE de-duplicated beam, many structured restarts, per-parent
# ADAPTIVE block sizes with replacement/reversal moves, gradient-accuracy + stagnation monitoring,
# gradient ENSEMBLES, support-size NEIGHBORHOODS, near-tie sampling, and SPATIALLY structured proposals.
# Note on item 15 (nondifferentiable preprocessing): the validator path here is differentiable and, for
# grid-aligned ±k_min edits, the PNG round-trip is identity (skip_roundtrip), so no surrogate is needed.
# ==========================================================================================
def _loss_field(ctx: Context, x: torch.Tensor, kind: str, targets: list[int]):
    """One loss's (value, toward-flip move_dir sign, saliency |g|). Wraps the soft loss's extra args."""
    if kind == "soft":
        return loss_grad(ctx.model, x, ctx.target_index, "soft", top_wrong=targets, tau=K.FEASUP_TAU)
    return loss_grad(ctx.model, x, ctx.target_index, kind)


def _ensemble_field(ctx: Context, x: torch.Tensor, kinds, targets: list[int]):
    """Gradient ensemble g_ens = Σ_ℓ move_dir_ℓ·score_ℓ / (‖g_ℓ‖₁+ε); consensus sign + |g_ens| saliency.
    Coordinates where independently-trained losses agree tend to be the most reliable actions (item 8)."""
    acc = torch.zeros(ctx.clean_u8.numel(), device=ctx.clean_u8.device)
    for kind in kinds:
        if _out_of_time(ctx):
            break
        _, md, sc = _loss_field(ctx, x, kind, targets)
        acc += (md * sc) / (sc.sum() + 1e-12)                  # toward-flip signed field, L1-normalized
    return acc.sign(), acc.abs()


def _delta_from_idx(ctx: Context, base: torch.Tensor, idx: torch.Tensor, move_dir: torch.Tensor):
    """Set the selected flat channels of `base` (a byte delta) to their toward-flip ±k_min action."""
    d = base.clone()
    d[idx] = move_dir[idx] * float(ctx.k_min)
    return d


def _tie_idx(masked: torch.Tensor, order: torch.Tensor, valid: int, k: int,
             temp: float, mult: int) -> torch.Tensor:
    """Near-tie sampling (item 10): draw k coords from the top (mult·k) by benefit, with probability
    ∝ exp(b_i/temp). When many benefits tie the exact top-k ranking is unreliable, so this diversifies
    while staying strongly gradient-guided."""
    pool = order[:min(valid, max(k, mult * k))]
    w = torch.softmax(masked[pool] / max(temp, 1e-6), dim=0)
    pick = torch.multinomial(w, min(k, pool.numel()), replacement=False)
    return pool[pick]


def _support_neighborhood(ctx: Context, move_dir, score, value, mults, tie_variants):
    """Support-size neighborhood (items 3/9/10): prefix supports at several multiples of K̂ plus the
    fully dense legal sign candidate, with a few near-tie-sampled variants per size. Returns byte deltas."""
    clean_flat = ctx.clean.view(-1)
    km = float(ctx.k_min)
    base = torch.zeros_like(ctx.clean_u8)
    masked, order, valid = build_sparse_order(score, move_dir, clean_flat)
    if valid == 0:
        return []
    kc = estimate_k(value + ctx.kappa, masked[order][:valid], ctx.q)
    deltas = [_delta_from_idx(ctx, base, movable(clean_flat, move_dir).nonzero(as_tuple=True)[0], move_dir)]
    seen_k = set()
    for f in mults:
        k = min(max(int(round(f * kc)), 1), valid)
        if k in seen_k:
            continue
        seen_k.add(k)
        deltas.append(_delta_from_idx(ctx, base, order[:k], move_dir))
        for _ in range(tie_variants):
            deltas.append(_delta_from_idx(ctx, base, _tie_idx(masked, order, valid, k,
                                                              K.FEASUP_TIE_TEMP, K.FEASUP_TIE_MULT), move_dir))
    return deltas


def _q_rank(ctx: Context, s_unit: torch.Tensor, move_dir: torch.Tensor, score: torch.Tensor):
    """Replacement/reversal gain q_i = -g_i(a*_i - S_i) = score_i·(1 - move_dir_i·s_i), masking box-clipped
    moves (item 6). add (0→±1) scores score_i; reversal (∓1→±1) scores 2·score_i; already-best scores 0."""
    q = score * (1.0 - move_dir * s_unit)
    q[~movable(ctx.clean.view(-1), move_dir)] = -1.0
    return q


def _block_moves(ctx: Context, delta: torch.Tensor, move_dir, score, blocks) -> list[torch.Tensor]:
    """Macro coordinate-descent children (items 5/6): switch the top-B coords (by replacement/reversal
    gain q) to their toward-flip action, for the given block-size ladder. Operates in byte-delta space
    (distinct from find_feasible's latent-space _block_children)."""
    km = float(ctx.k_min)
    s_unit = delta / km
    q = _q_rank(ctx, s_unit, move_dir, score)
    order = torch.argsort(q, descending=True)
    pos = int((q > 0).sum().item())
    out: list[torch.Tensor] = []
    for b in blocks:
        b = min(int(b), pos)
        if b < 1:
            continue
        d = delta.clone()
        d[order[:b]] = move_dir[order[:b]] * km
        out.append(d)
    return out


def _spatial_children(ctx: Context, move_dir, score) -> list[torch.Tensor]:
    """Spatially structured proposals (item 11): top saliency PATCHES (all channels) and per-RGB-channel
    dense candidates. Network features are spatially correlated, so a coherent region can beat scattered
    top-gradient channels."""
    C, H, W = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2])
    km = float(ctx.k_min)
    clean_flat = ctx.clean.view(-1)
    can = movable(clean_flat, move_dir)
    sal = score.clone()
    sal[~can] = 0.0
    base = torch.zeros_like(ctx.clean_u8)
    deltas: list[torch.Tensor] = []

    p = max(1, int(K.FEASUP_PATCH))
    s2 = sal.view(C, H, W).sum(dim=0, keepdim=True).unsqueeze(0)           # [1,1,H,W] summed over channels
    pooled = F.avg_pool2d(s2, p, stride=p, ceil_mode=True)[0, 0]           # [ceil(H/p), ceil(W/p)] saliency
    ph, pw = pooled.shape                                                  # ceil_mode => grid covers all of H×W
    flat = pooled.reshape(-1)
    nptch = flat.numel()
    for frac in (0.05, 0.15, 0.40):                                       # cover a few patch budgets
        t = max(1, min(int(round(frac * nptch)), nptch))
        top = torch.topk(flat, t).indices
        mask = torch.zeros(ph * pw, device=flat.device, dtype=torch.bool)
        mask[top] = True
        mask2d = mask.view(ph, pw)
        full = mask2d.repeat_interleave(p, 0)[:H].repeat_interleave(p, 1)[:, :W]  # upsample, trim to H×W
        chan_mask = full.reshape(1, H, W).expand(C, H, W).reshape(-1) & can
        idx = chan_mask.nonzero(as_tuple=True)[0]
        if idx.numel():
            deltas.append(_delta_from_idx(ctx, base, idx, move_dir))

    chans = can.view(C, H, W)                                             # per-RGB-channel dense
    for c in range(C):
        idx = chans[c].reshape(-1).nonzero(as_tuple=True)[0] + c * H * W
        if idx.numel():
            deltas.append(_delta_from_idx(ctx, base, idx, move_dir))
    return deltas


def _rescue_children(ctx: Context, beam: list[dict], dev) -> list[torch.Tensor]:
    """Gradient-free rescue (item 14): mutate around SEVERAL near-flips — flip a random fraction of active
    actions, reverse a random subset, or combine the supports of two diverse parents."""
    km = float(ctx.k_min)
    out: list[torch.Tensor] = []
    parents = beam[:min(4, len(beam))]
    for st in parents:
        d0 = st["delta"]
        for frac in K.FEASUP_RESCUE_FRACS:
            d = d0.clone()
            m = torch.rand(d.numel(), device=dev) < frac
            vals = (torch.randint(0, 3, (int(m.sum().item()),), device=dev).float() - 1.0) * km
            d[m] = vals
            out.append(d)
    if len(parents) >= 2:                                                 # combine two diverse supports
        a, b = parents[0]["delta"], parents[-1]["delta"]
        take = torch.rand(a.numel(), device=dev) < 0.5
        out.append(torch.where(take, a, b))
    return out


def _build_target_pool(ctx: Context, x: torch.Tensor, dev) -> list[dict]:
    """Target pool prioritized by linearized crossing size K̂_c (items 2/9). One pair-loss backward per
    target; entries carry the cached field so seeding/expansion can reuse them. Smaller K̂_c first."""
    logits = logits_of(ctx.model, x)
    cands = top_wrong_classes(logits, ctx.target_index, K.FEASUP_TOPM)
    extra = [c for c in range(logits.numel())
             if c != ctx.target_index and c not in cands]
    random.shuffle(extra)
    cands += extra[:max(0, K.FEASUP_RAND_TARGETS)]                        # a few random wrong classes
    clean_flat = ctx.clean.view(-1)
    pool: list[dict] = []
    for c in cands:
        if _out_of_time(ctx):
            break
        value, md, sc = loss_grad(ctx.model, x, ctx.target_index, f"pair:{c}")
        masked, order, valid = build_sparse_order(sc, md, clean_flat)
        if valid == 0:
            continue
        kc = estimate_k(value + ctx.kappa, masked[order][:valid], ctx.q)
        pool.append({"c": c, "value": value, "move_dir": md, "score": sc, "kc": kc})
    pool.sort(key=lambda e: e["kc"])                                      # prioritize smallest crossing
    return pool


def _mask_of(delta: torch.Tensor) -> torch.Tensor:
    return delta != 0


def _too_similar(a: torch.Tensor, b: torch.Tensor, thr: float) -> bool:
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    return union > 0 and inter / union > thr


def _beam_insert(beam: list[dict], st: dict, cap: int, thr: float) -> None:
    """Insert a candidate state, rejecting near-duplicate supports of a better state and evicting
    near-duplicate worse states; keep the `cap` lowest-margin, support-diverse states (item 3)."""
    st["mask"] = _mask_of(st["delta"])
    for s in beam:
        if s["margin"] <= st["margin"] and _too_similar(s["mask"], st["mask"], thr):
            return
    beam[:] = [s for s in beam if not (st["margin"] < s["margin"] and _too_similar(st["mask"], s["mask"], thr))]
    beam.append(st)
    beam.sort(key=lambda s: s["margin"])
    del beam[cap:]


def find_feasible_upgraded(ctx: Context) -> str | None:
    """Upgraded Problem 1 solver: loss portfolio + diverse beam + adaptive blocks (see block comment).

    Seeds a diverse beam from structured restarts (dense sign, support neighborhoods, per-target supports,
    random subsets), then loops: pick a beam parent and a portfolio loss (reward-per-second weighted),
    recompute the field at that discrete state, batch a speculative set of children (support neighborhoods,
    adaptive replacement/reversal block moves, near-tie variants, spatial proposals, an occasional
    ensemble), evaluate them in one forward, fold survivors into the beam, and adapt the parent's block
    size from realized vs. predicted improvement. On stagnation it rotates loss/target/parent and runs a
    gradient-free rescue. Returns 'safe' the instant an envelope-safe flip is banked."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    kinds = list(K.FEASUP_LOSSES)

    pool = _build_target_pool(ctx, ctx.clean, dev)
    targets = [e["c"] for e in pool] or top_wrong_classes(logits_of(ctx.model, ctx.clean),
                                                           ctx.target_index, K.FEASUP_TOPM)

    # --- Structured restarts (items 4/12): one batched seed population --------------------
    seeds: list[torch.Tensor] = []
    for kind in kinds:                                                    # untargeted portfolio losses
        if _out_of_time(ctx):
            break
        value, md, sc = _loss_field(ctx, ctx.clean, kind, targets)
        seeds += _support_neighborhood(ctx, md, sc, ctx.m0, K.FEASUP_K_MULTS, K.FEASUP_TIE_VARIANTS)
    for e in pool[:min(5, len(pool))]:                                    # smallest-K̂ targeted supports
        seeds += _support_neighborhood(ctx, e["move_dir"], e["score"], e["value"],
                                       K.FEASUP_K_MULTS, K.FEASUP_TIE_VARIANTS)
    for p in K.FEASUP_SPARSE_STARTS:                                      # sparse random subsets
        d = torch.zeros_like(ctx.clean_u8)
        m = torch.rand(n, device=dev) < p
        d[m] = (torch.randint(0, 2, (int(m.sum().item()),), device=dev).float() * 2.0 - 1.0) * km
        seeds.append(d)
    for _ in range(K.FEASUP_DENSE_STARTS):
        seeds.append((torch.randint(0, 2, (n,), device=dev).float() * 2.0 - 1.0) * km)

    beam: list[dict] = []
    res = _eval_deltas(ctx, seeds)
    if ctx.bank.has_safe:
        return "safe"
    for r in res:
        _beam_insert(beam, {"delta": r["delta"], "margin": r["margin"], "block": K.FEASUP_BLOCK0,
                            "fail": 0}, K.FEASUP_BEAM, K.FEASUP_DUP_JACCARD)
    if not beam:                                                          # degenerate: seed from clean
        _beam_insert(beam, {"delta": torch.zeros_like(ctx.clean_u8), "margin": ctx.m0,
                            "block": K.FEASUP_BLOCK0, "fail": 0}, K.FEASUP_BEAM, K.FEASUP_DUP_JACCARD)

    # --- Adaptive portfolio / beam search --------------------------------------------------
    reward = {k: 0.0 for k in kinds}                                      # margin-drop per second (EMA)
    best_margin = beam[0]["margin"]
    pi = ti = it = 0                                                      # parent / target / iter cursors
    stagnant = 0
    init_best_wrong = targets[0] if targets else None

    while not _out_of_time(ctx):
        it += 1
        parent = beam[pi % len(beam)]
        pi += 1
        # loss selection weighted by reward-per-second, with an exploration floor (item 1)
        w = torch.tensor([max(reward[k], 0.0) + 0.1 for k in kinds])
        kind = kinds[int(torch.multinomial(w / w.sum(), 1).item())]
        ent = pool[ti % len(pool)] if pool else None                     # rotate targets by priority
        ti += 1

        x = apply_delta_bytes(ctx.clean_u8, parent["delta"], ctx.shape)
        t0 = time.time()
        if K.FEASUP_ENS_EVERY > 0 and it % K.FEASUP_ENS_EVERY == 0:       # ensemble field (item 8)
            move_dir, score = _ensemble_field(ctx, x, K.FEASUP_ENS_LOSSES, targets)
            value = ctx.m0
        else:
            value, move_dir, score = _loss_field(ctx, x, kind, targets)

        children: list[torch.Tensor] = []
        children += _block_moves(ctx, parent["delta"], move_dir, score,
                                 (parent["block"] // 2, parent["block"], parent["block"] * 2))
        children += _support_neighborhood(ctx, move_dir, score, value, K.FEASUP_K_MULTS,
                                           K.FEASUP_TIE_VARIANTS)
        if ent is not None:                                              # targeted support neighborhood
            children += _support_neighborhood(ctx, ent["move_dir"], ent["score"], ent["value"],
                                              K.FEASUP_K_MULTS, K.FEASUP_TIE_VARIANTS)
        if K.FEASUP_SPATIAL:
            children += _spatial_children(ctx, move_dir, score)

        res = _eval_deltas(ctx, children)
        if ctx.bank.has_safe:
            return "safe"
        if not res:
            break
        bi = min(range(len(res)), key=lambda i: res[i]["margin"])
        child_best = res[bi]["margin"]
        dt = max(time.time() - t0, 1e-3)
        reward[kind] = 0.7 * reward[kind] + 0.3 * max(0.0, parent["margin"] - child_best) / dt

        for r in res:
            _beam_insert(beam, {"delta": r["delta"], "margin": r["margin"], "block": parent["block"],
                                "fail": 0}, K.FEASUP_BEAM, K.FEASUP_DUP_JACCARD)

        # adaptive block + gradient-accuracy monitor (items 5/7)
        if child_best < parent["margin"] - 1e-6:
            parent["block"] = min(K.FEASUP_BLOCK_MAX, parent["block"] * 2)
            parent["fail"] = 0
        else:
            parent["block"] = max(K.FEASUP_BLOCK_MIN, parent["block"] // 2)
            parent["fail"] += 1

        if child_best < best_margin - 1e-6:
            best_margin = child_best
            stagnant = 0
        else:
            stagnant += 1

        # stagnation rotation + rescue (items 13/14) and dynamic target-pool rebuild (item 2)
        if stagnant >= K.FEASUP_STAGNATION or (K.FEASUP_POOL_REFRESH > 0 and it % K.FEASUP_POOL_REFRESH == 0):
            xb = apply_delta_bytes(ctx.clean_u8, beam[0]["delta"], ctx.shape)
            cur_best_wrong = top_wrong_classes(logits_of(ctx.model, xb), ctx.target_index, 1)[0]
            if cur_best_wrong != init_best_wrong or stagnant >= K.FEASUP_STAGNATION:
                pool = _build_target_pool(ctx, xb, dev) or pool
                targets = [e["c"] for e in pool] or targets
                init_best_wrong = cur_best_wrong
        if stagnant >= K.FEASUP_STAGNATION:
            _eval_deltas(ctx, _rescue_children(ctx, beam, dev))
            if ctx.bank.has_safe:
                return "safe"
            pi += 1                                                      # jump to a different parent
            stagnant = 0

    if not ctx.bank.has_flip:
        logger.info(f"[feasible_upgraded] no flip; best_margin={best_margin:.4f} "
                    f"({'likely q=1 infeasible' if best_margin > 2.0 else 'near-miss'})")
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
    """Find ANY envelope-safe ±k_min/255 flip (Problem 1 / feasibility) and return it; else the clean
    image. With PERTURB_ALLOW_UNSAFE_FLIP=1, returns any margin<0 flip (the literal spec)."""
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
    # headroom for serialization + verification (#4); the search deadline shrinks accordingly.
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

    # Approach 1 (feasibility): the selected engine finds the first flip and serves as the repair engine.
    engine = find_feasible_upgraded if K.SOLVER == "upgraded" else find_feasible
    engine(ctx)

    # Approach 2 (L0 minimization): if enabled, minimize |S| while the Bank keeps the verified incumbent.
    if K.APPROACH2 and ctx.bank.has_flip:
        from . import approach2
        approach2.compress(ctx, repair_fn=engine)

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
