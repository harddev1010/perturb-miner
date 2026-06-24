"""approach2.py — Ternary Anytime Sparse Optimizer (Problem 2 / L0 minimization).

Runs AFTER a feasibility flip exists (produced by the PERTURB_SOLVER engine in perturb.py) and minimizes
the number of changed channels |S|_0 while the Bank continuously retains the smallest exactly-graded flip
as the verified incumbent. Anytime + deadline-aware: every phase banks only validator-faithful (envelope +
SSIM/PSNR) candidates, so an interrupt at any point still returns the best incumbent so far.

Phases (each independently toggleable via PERTURB_A2_* — see constants.py):
  E.1 reinforce   — same-cost margin reinforcement (re-sign + weak<->strong swap) to create deletion slack.
  B   prune       — hierarchical (delta-debugging) group deletion of low-importance coordinates.
  E.2 exchange    — compressing exchanges: add 1 strong unused channel, drop >=2 weak ones (net shrink).
  C   apgd        — fixed-K ternary APGD: exact Euclidean projection onto T_K = {S : |S|_0<=K}, loss
                    portfolio, warm/random restarts, adaptive step; driven by a coarse->fine K schedule.
  D   repair      — repair near-flips into smaller valid flips; optionally re-run Approach 1 for new basins.
  σ0  sigma_zero  — ternary sigma-zero-style soft-L0 sparsification (complementary; off by default).
  R   rescue      — gradient-free randomized mutation around the incumbent.

The incumbent / valid archive is the shared Bank (lexicographic order |S|, L∞, RMSE). The same Approach 1
engine selected by PERTURB_SOLVER is reused as the repair / independent-basin engine (passed in as a
callback), so Approach 2 composes with either the original or the upgraded Approach 1.
"""

from __future__ import annotations

import logging
import math
import random

import torch

from . import constants as K
from .utils import (
    apply_delta_bytes,
    batch_eval,
    build_sparse_order,
    estimate_k,
    eval_budget,
    logits_of,
    loss_grad,
    movable,
    out_of_budget,
    top_wrong_classes,
)

logger = logging.getLogger(__name__)


# ==========================================================================================
# Shared helpers (self-contained so this module never imports perturb.py — no import cycle)
# ==========================================================================================
def _out_of_time(ctx) -> bool:
    return out_of_budget(ctx)


def _eval(ctx, deltas: list[torch.Tensor]) -> list[dict]:
    """Grade byte deltas on the validator-faithful path; fold into the Bank; attach the delta."""
    if not deltas:
        return []
    cands = [apply_delta_bytes(ctx.clean_u8, d, ctx.shape) for d in deltas]
    res = batch_eval(ctx, cands)
    ctx.bank.consider(res)
    for r, d in zip(res, deltas):
        r["delta"] = d
    return res


def _accept(ctx, r: dict) -> bool:
    """A valid working base: a quality (in-band + SSIM/PSNR) flip that is envelope-safe — or, with
    PERTURB_ALLOW_UNSAFE_FLIP=1, any quality flip. Requiring quality stops shrink stages from drifting
    back toward the clean image (margin-safe but out of band)."""
    if r.get("safe") and r.get("quality"):
        return True
    return bool(ctx.allow_unsafe and r.get("flipped") and r.get("quality"))


def _incumbent_delta(ctx) -> torch.Tensor | None:
    """Integer byte delta (flat) of the Bank's best flip — safe preferred — or None if no flip yet."""
    best = ctx.bank.best_safe if ctx.bank.best_safe is not None else ctx.bank.best_flip
    if best is None:
        return None
    d = best.get("delta")
    if d is not None:
        return d.clone()
    return torch.round(best["cand"].view(-1) * 255.0) - ctx.clean_u8


def _incumbent_k(ctx) -> int:
    best = ctx.bank.best_safe if ctx.bank.best_safe is not None else ctx.bank.best_flip
    return int(best["nz"]) if best is not None else 0


def _lock_target(ctx, delta: torch.Tensor) -> int:
    """Best wrong class at the current candidate — a stable target to rank the support against."""
    x = apply_delta_bytes(ctx.clean_u8, delta, ctx.shape)
    return top_wrong_classes(logits_of(ctx.model, x), ctx.target_index, 1)[0]


def _pair(ctx, delta: torch.Tensor, t: int):
    """Locked-target margin h_t = z_y - z_t, its toward-flip move dir, and |∂h_t/∂x| at clean+delta."""
    x = apply_delta_bytes(ctx.clean_u8, delta, ctx.shape)
    return loss_grad(ctx.model, x, ctx.target_index, f"pair:{t}")


def _revert_cost(move_dir, score, delta, idx, km, q) -> torch.Tensor:
    """First-order Δh_t from reverting each channel in idx one byte toward clean. Negative => reverting
    also helps the flip (do it first); positive => it costs margin slack."""
    return move_dir[idx] * score[idx] * torch.sign(delta[idx]) * km * q


# ==========================================================================================
# Phase E.1 — reinforce (same-cost margin gain)
# ==========================================================================================
def _reinforce(ctx, delta: torch.Tensor, t: int) -> torch.Tensor:
    """At FIXED |S|, re-sign active channels toward the flip and swap the weakest active channels for the
    strongest unused ones; keep the equal-k candidate with the most negative margin. Manufactures the
    slack that lets later stages delete."""
    km = float(ctx.k_min)
    clean_flat = ctx.clean.view(-1)
    _, move_dir, score = _pair(ctx, delta, t)
    active = delta != 0
    k = int(active.sum().item())
    cands = [delta]
    d = torch.zeros_like(delta)
    d[active] = move_dir[active] * km                         # (1) re-sign active toward the flip
    cands.append(d)
    act_idx = active.nonzero(as_tuple=True)[0]
    if act_idx.numel() > 0:                                   # (2) swap weak active <-> strong unused
        weak = act_idx[torch.argsort(score[act_idx])]
        gabs = score.clone()
        gabs[active] = -1.0
        gabs[~movable(clean_flat, move_dir)] = -1.0
        n_un = int((gabs > 0).sum().item())
        for frac in K.A2_SWAP_FRACS:
            cnt = min(int(frac * act_idx.numel()) + 1, int(weak.numel()), n_un)
            if cnt < 1:
                continue
            strong = torch.topk(gabs, cnt).indices
            d = delta.clone()
            d[weak[:cnt]] = 0.0
            d[strong] = move_dir[strong] * km
            cands.append(d)
    res = _eval(ctx, cands)
    pick = _pick(ctx, res, max_k=k, require_accept=False)
    return pick["delta"] if pick is not None else delta


def _pick(ctx, res: list[dict], max_k: int, require_accept: bool) -> dict | None:
    """Lowest-margin candidate that still flips, with |S|<=max_k and (optionally) accept-safe."""
    best = None
    for r in res:
        if not r.get("flipped") or r["nz"] > max_k:
            continue
        if require_accept and not _accept(ctx, r):
            continue
        if best is None or r["margin"] < best["margin"]:
            best = r
    return best


# ==========================================================================================
# Phase B — hierarchical group deletion (delta-debugging style)
# ==========================================================================================
def _group_prune(ctx, delta: torch.Tensor, t: int) -> torch.Tensor:
    """Rank active channels by revert cost (ascending), batch-eval a geometric ladder of removal counts
    (+ the linear-predicted safe prefix) in ONE forward, and take the LARGEST removal that stays accept-
    safe. If even the smallest group fails, the ladder's fine sizes act as the delta-debugging split."""
    km = float(ctx.k_min)
    q = ctx.q
    base = float(K.A2_PRUNE_LADDER) if K.A2_PRUNE_LADDER > 1 else 2.0
    cur = delta
    while not _out_of_time(ctx):
        changed = (cur != 0).nonzero(as_tuple=True)[0]
        if changed.numel() == 0:
            break
        val, move_dir, score = _pair(ctx, cur, t)
        if val >= 0.0:                                        # locked target no longer winning -> stop
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
        res = _eval(ctx, cands)
        best_d, best_sz = None, 0
        for sz, d, r in zip(sizes, cands, res):
            if _accept(ctx, r) and sz > best_sz:
                best_sz, best_d = sz, d
        if best_d is None:
            break
        cur = best_d
    return cur


# ==========================================================================================
# Phase E.2 — compressing exchange (one-for-many)
# ==========================================================================================
def _exchange(ctx, delta: torch.Tensor, t: int) -> torch.Tensor:
    """Add ONE strong unused channel j (drops h_t by ~q·km·|g_j|) and revert as many cheap active channels
    as that extra slack pays for, so |S| strictly drops. The added channel may lie OUTSIDE the current
    support, escaping the subset trap that bounds pure backward pruning."""
    km = float(ctx.k_min)
    q = ctx.q
    clean_flat = ctx.clean.view(-1)
    mindrop = max(2, int(K.A2_EXCHANGE_MIN_DROP))
    cur = delta
    while not _out_of_time(ctx):
        val, move_dir, score = _pair(ctx, cur, t)
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
        gabs = score.clone()
        gabs[changed_mask] = -1.0
        gabs[~movable(clean_flat, move_dir)] = -1.0
        n_add = min(int(K.A2_EXCHANGE_ADDS), int((gabs > 0).sum().item()))
        if n_add < 1:
            break
        add_idx = torch.topk(gabs, n_add).indices.tolist()
        slack = -ctx.kappa - val

        def _nmax(j):
            budget = slack + q * km * float(score[j].item())
            return int((cum <= budget).sum().item())

        cands = []
        j0 = add_idx[0]
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
        for j in add_idx[1:]:
            nrem = _nmax(j)
            if nrem < mindrop:
                continue
            d = cur.clone()
            d[rem_order[:nrem]] = 0.0
            d[j] = move_dir[j] * km
            cands.append(d)
        if not cands:
            break
        res = _eval(ctx, cands)
        best_d, best_nz = None, int(changed.numel())
        for d, r in zip(cands, res):
            if _accept(ctx, r) and r["nz"] < best_nz:
                best_nz, best_d = r["nz"], d
        if best_d is None:
            break
        cur = best_d
    return cur


# ==========================================================================================
# exact leave-one-out cleanup (fine pass)
# ==========================================================================================
def _loo(ctx, delta: torch.Tensor) -> torch.Tensor:
    """Eval every single-channel revert, collect the accept-safe ones, then take the LARGEST accept-safe
    cumulative group (most-slack-first ladder) — catches curvature / target switches the first-order
    ranking misses. Only when |S| is small enough to be worth exact probes."""
    changed = (delta != 0).nonzero(as_tuple=True)[0]
    n = int(changed.numel())
    # Skip when the support is too big to probe exactly in the remaining budget (#3): building n single-
    # channel reverts only to have batch_eval truncate them is wasted work.
    if n == 0 or n > min(K.A2_LOO_MAX, eval_budget(ctx)):
        return delta
    cands = [delta.clone() for _ in range(n)]
    idx_list = changed.tolist()
    for d, i in zip(cands, idx_list):
        d[i] = 0.0
    res = _eval(ctx, cands)
    rem = [(r["margin"], i) for i, r in zip(idx_list, res) if _accept(ctx, r)]
    if not rem:
        return delta
    rem.sort(key=lambda z: z[0])
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
    res2 = _eval(ctx, cands2)
    best_d, best_sz = None, 0
    for sz, d, r in zip(sizes, cands2, res2):
        if _accept(ctx, r) and sz > best_sz:
            best_sz, best_d = sz, d
    return best_d if best_d is not None else delta


# ==========================================================================================
# Phase C — fixed-K ternary APGD (exact T_K projection)
# ==========================================================================================
def _loss_field(ctx, x, kind, targets):
    if kind == "soft":
        return loss_grad(ctx.model, x, ctx.target_index, "soft", top_wrong=targets, tau=K.A2_TAU)
    return loss_grad(ctx.model, x, ctx.target_index, kind)


def _project_TK(ctx, U: torch.Tensor, k: int) -> torch.Tensor:
    """Exact Euclidean projection of a continuous latent U (unit space) onto T_K = {S : S_i∈{-1,0,+1},
    movable, |S|_0<=K}. d*_i = sign(U_i) (if legal); q_i = U_i² - (U_i-d*_i)² = 2|U_i|-1; keep the K
    largest positive q_i set to d*_i, the rest zero."""
    clean_flat = ctx.clean.view(-1)
    d_star = torch.sign(U)
    d_star[~movable(clean_flat, d_star)] = 0.0
    q = 2.0 * U.abs() - 1.0
    q[d_star == 0] = float("-inf")
    npos = int((q > 0).sum().item())
    kk = min(int(k), npos)
    S = torch.zeros_like(U)
    if kk > 0:
        idx = torch.topk(q, kk).indices
        S[idx] = d_star[idx]
    return S


def _apgd_inits(ctx, k: int, targets) -> list[torch.Tensor]:
    """Restart seeds (unit space, already |S|_0<=k): incumbent projected to k, a random subset of it, and
    a fresh top-k target-gradient support per portfolio loss."""
    km = float(ctx.k_min)
    clean_flat = ctx.clean.view(-1)
    out: list[torch.Tensor] = []
    inc = _incumbent_delta(ctx)
    if inc is not None:
        s = inc / km
        act = (s != 0).nonzero(as_tuple=True)[0]
        if act.numel() > k:                                   # keep k active coords (project down)
            keep = act[torch.randperm(act.numel(), device=s.device)[:k]]
            s2 = torch.zeros_like(s)
            s2[keep] = s[keep]
            out.append(s2)
            out.append(_project_TK(ctx, s, k))                # |U|-based projection of the incumbent
        else:
            out.append(s)
    for kind in K.A2_LOSSES:
        if _out_of_time(ctx):
            break
        _, move_dir, score = _loss_field(ctx, ctx.clean, kind, targets)
        masked, order, valid = build_sparse_order(score, move_dir, clean_flat)
        if valid == 0:
            continue
        kk = min(k, valid)
        s = torch.zeros_like(move_dir)
        s[order[:kk]] = move_dir[order[:kk]]
        out.append(s)
    return out or [torch.zeros(ctx.clean_u8.numel(), device=ctx.clean_u8.device)]


def _fixed_k_apgd(ctx, k: int, targets) -> bool:
    """Try to find a valid flip with |S|_0<=k via T_K-projected APGD over a loss portfolio + restarts.
    Returns True if the incumbent now has |S|<=k (i.e. progress at this budget)."""
    if k < 1:
        return False
    km = float(ctx.k_min)
    losses = list(K.A2_LOSSES)
    inits = _apgd_inits(ctx, k, targets)
    states = [{"S": s, "prev": s.clone(), "alpha": K.A2_APGD_ALPHA0, "kind": losses[i % len(losses)],
               "best": float("inf"), "stale": 0} for i, s in enumerate(inits)]
    for _ in range(K.A2_APGD_ITERS):
        if _out_of_time(ctx) or not states:
            break
        deltas = [s["S"] * km for s in states]
        res = _eval(ctx, deltas)
        if ctx.bank.best_safe is not None and ctx.bank.best_safe["nz"] <= k:
            return True
        for s, r in zip(states, res):
            if r["margin"] < s["best"] - 1e-6:
                s["best"], s["stale"] = r["margin"], 0
            else:
                s["stale"] += 1
                if s["stale"] >= K.A2_APGD_PATIENCE:
                    s["alpha"] = max(K.A2_APGD_MIN_ALPHA, s["alpha"] * 0.5)
                    s["stale"] = 0
        for s, r in zip(states, res):
            if _out_of_time(ctx):
                break
            _, move_dir, score = _loss_field(ctx, r["cand"], s["kind"], targets)
            sal = score / (score.max() + 1e-12)
            U = s["S"] + s["alpha"] * move_dir * sal + K.A2_APGD_MOMENTUM * (s["S"] - s["prev"])
            s["prev"] = s["S"]
            s["S"] = _project_TK(ctx, U, k)
    return ctx.bank.best_safe is not None and ctx.bank.best_safe["nz"] <= k


def _coarse_to_fine(ctx, targets) -> None:
    """Coarse-to-fine support schedule (section 5): K=ceil(0.75·U); on success repeat from the new U,
    on failure bisect toward U; then fine reductions U-1, U-2, U-4, U-8."""
    u = _incumbent_k(ctx)
    failed = u + 1
    while not _out_of_time(ctx) and u > 1:
        k = max(1, int(math.ceil(K.A2_K_COARSE * u)))
        if k >= u:
            break
        ok = _fixed_k_apgd(ctx, k, targets)
        u_now = _incumbent_k(ctx)
        if ok and u_now < u:
            u, failed = u_now, u_now + 1
        else:
            failed = min(failed, k)
            k = int(math.ceil((u + failed) / 2.0))
            if k >= u:
                break
    u = _incumbent_k(ctx)
    for step in K.A2_FINE_STEPS:                              # fine reductions
        if _out_of_time(ctx):
            break
        _fixed_k_apgd(ctx, max(1, u - int(step)), targets)
        u = _incumbent_k(ctx)


# ==========================================================================================
# Phase D — repair near-flips (+ optional independent re-run of Approach 1)
# ==========================================================================================
def _repair(ctx, repair_fn) -> None:
    """Build a near-flip below the incumbent by reverting a couple of cheap active channels, then add a
    small ladder of strong unused channels to re-flip at lower |S| (repair-prune cycle in miniature).
    Optionally re-run the Approach 1 engine to discover a fundamentally different, smaller flip basin."""
    km = float(ctx.k_min)
    q = ctx.q
    clean_flat = ctx.clean.view(-1)
    inc = _incumbent_delta(ctx)
    if inc is not None:
        kcur = int((inc != 0).sum().item())
        t = _lock_target(ctx, inc)
        _, move_dir, score = _pair(ctx, inc, t)
        active = (inc != 0).nonzero(as_tuple=True)[0]
        if active.numel() > 0:
            cost = _revert_cost(move_dir, score, inc, active, km, q)
            cheap = active[torch.argsort(cost)]
            drop = min(max(2, K.A2_REPAIR_ADDS // 2), int(cheap.numel()))
            near = inc.clone()
            near[cheap[:drop]] = 0.0                          # near-flip with |S| = kcur - drop
            gabs = score.clone()
            gabs[inc != 0] = -1.0
            gabs[~movable(clean_flat, move_dir)] = -1.0
            n_add = min(int(K.A2_REPAIR_ADDS), int((gabs > 0).sum().item()))
            if n_add >= 1:
                add = torch.topk(gabs, n_add).indices
                cands = []
                a = 1
                while a <= n_add and (kcur - drop + a) < kcur:  # keep net shrink
                    d = near.clone()
                    d[add[:a]] = move_dir[add[:a]] * km
                    cands.append(d)
                    a = max(a + 1, a * 2)
                _eval(ctx, cands)
    if K.A2_INDEP_RERUN and repair_fn is not None and not _out_of_time(ctx):
        repair_fn(ctx)                                        # different basin may be much sparser


# ==========================================================================================
# Section 8 — ternary sigma-zero-style soft-L0 sparsification (complementary)
# ==========================================================================================
def _sigma_zero(ctx, delta: torch.Tensor) -> None:
    """Maintain a per-channel keep value v∈[0,1]; each step move it by (saliency - λ): channels whose
    toward-flip benefit beats the sparsity price λ stay, the rest decay to zero. λ rises when the current
    mask is valid (push more zeros) and falls when invalid (recover feasibility). STE-free: every probed
    mask is materialized as a real ternary delta and graded by the Bank."""
    km = float(ctx.k_min)
    if delta is None:
        return
    v = (delta != 0).float()
    lam = float(K.A2_SIGMA0_LAMBDA0)
    cur = delta
    for _ in range(K.A2_SIGMA0_ITERS):
        if _out_of_time(ctx):
            break
        t = _lock_target(ctx, cur)
        _, move_dir, score = _pair(ctx, cur, t)
        sal = score / (score.max() + 1e-12)
        v = (v + 0.5 * (sal - lam)).clamp(0.0, 1.0)
        keep = (v > 0.5) & movable(ctx.clean.view(-1), move_dir)
        d = torch.zeros_like(delta)
        d[keep] = move_dir[keep] * km
        res = _eval(ctx, [d])
        if res and _accept(ctx, res[0]):
            cur = d
            lam *= 1.2                                        # valid -> demand more sparsity
        else:
            lam *= 0.8                                        # invalid -> relax


# ==========================================================================================
# Section 9 — randomized rescue
# ==========================================================================================
def _rescue(ctx) -> None:
    """Gradient-free mutation around the incumbent: replace a random fraction of the support, reverse a
    random subset, and merge the support with a random high-gradient block. Accepted purely on real
    margin via the Bank."""
    km = float(ctx.k_min)
    dev = ctx.clean_u8.device
    inc = _incumbent_delta(ctx)
    if inc is None:
        return
    cands = []
    for frac in K.A2_RESCUE_FRACS:
        d = inc.clone()
        m = torch.rand(d.numel(), device=dev) < frac
        cnt = int(m.sum().item())
        if cnt:
            d[m] = (torch.randint(0, 3, (cnt,), device=dev).float() - 1.0) * km
        cands.append(d)
    _eval(ctx, cands)


# ==========================================================================================
# Orchestrator
# ==========================================================================================
def compress(ctx, repair_fn=None) -> None:
    """Ternary Anytime Sparse Optimizer (see module docstring). Requires an existing flip in the Bank.
    Loops the enabled phases against the deadline; the Bank keeps the smallest verified flip. Stops when a
    full round makes no |S| progress or the budget runs out."""
    if not ctx.bank.has_flip:
        return
    targets = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.A2_TOPM)
    while not _out_of_time(ctx):
        k0 = _incumbent_k(ctx)
        delta = _incumbent_delta(ctx)
        if delta is None:
            break
        t = _lock_target(ctx, delta)
        if K.A2_REINFORCE:
            delta = _reinforce(ctx, delta, t)
        if K.A2_PRUNE:
            delta = _group_prune(ctx, delta, t)
        if K.A2_EXCHANGE:
            delta = _exchange(ctx, delta, t)
        if K.A2_APGD:
            _coarse_to_fine(ctx, targets)
        if K.A2_SIGMA0:
            _sigma_zero(ctx, _incumbent_delta(ctx))
        if K.A2_REPAIR:
            _repair(ctx, repair_fn)
        if K.A2_LOO:
            d = _incumbent_delta(ctx)
            if d is not None:
                _loo(ctx, d)
        if K.A2_RESCUE:
            _rescue(ctx)
        if _incumbent_k(ctx) >= k0:                           # no net progress this round
            break
