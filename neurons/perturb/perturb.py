"""perturb.py — the five flip-finding algorithms + orchestrators + the public entry point.

Design: few backward passes, many batched forward checks. Each algorithm proposes candidate
byte-perturbations, batch-evaluates them, and folds the survivors into a shared Bank that tracks
the sparsest envelope-safe flip. Faithful to the spec, adapted to byte-space + the transfer-safety
gate (see neurons/perturb/utils.py and constants.py).

SWITCHES (no redeploy needed):
  * Orchestrator: change the one function name at the call site marked below in perturb().
  * Pipeline membership: comment a line out of PIPELINE to exclude that algorithm.
  * Per-algorithm tuning + the accept gate: env vars in constants.py.
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
    apply_delta_bytes,
    batch_eval,
    build_sparse_order,
    estimate_k,
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


def _k_ladder(k_need: int, valid_count: int) -> list[int]:
    """The spec's candidate-k ladder: a geometric sweep around the linear estimate plus a few
    density anchors, clamped to [1, valid_count] and de-duplicated."""
    raw = [
        k_need // 8, k_need // 4, k_need // 2, k_need, 2 * k_need, 4 * k_need,
        int(0.001 * valid_count), int(0.005 * valid_count),
        int(0.01 * valid_count), int(0.05 * valid_count), valid_count,
    ]
    return sorted({max(1, min(int(k), valid_count)) for k in raw if k and k > 0})


# ==========================================================================================
# 1. Batched multi-loss quantized FGSM
# ==========================================================================================
def batched_multi_loss_qfgsm(ctx: Context) -> str | None:
    """One backward per loss (hard CW margin / cross-entropy / soft top-M), each ranked by |g|,
    then a whole ladder of one-byte prefix candidates batch-evaluated at once."""
    logits = logits_of(ctx.model, ctx.clean)
    top_wrong = top_wrong_classes(logits, ctx.target_index, K.QFGSM_TOPM)
    clean_flat = ctx.clean.view(-1)

    for kind in ("hard", "ce", "soft"):
        if _out_of_time(ctx):
            break
        _, move_dir, score = loss_grad(
            ctx.model, ctx.clean, ctx.target_index, kind, top_wrong=top_wrong, tau=K.QFGSM_TAU
        )
        msk_score, order, valid = build_sparse_order(score, move_dir, clean_flat)
        if valid == 0:
            continue
        k_need = estimate_k(ctx.m0 + max(ctx.kappa, 0.0), msk_score[order], ctx.q)
        cands = [apply_byte(ctx.clean_u8, move_dir, order[:k], ctx.k_min, ctx.shape)
                 for k in _k_ladder(k_need, valid)]
        _consider(ctx, cands)
        if ctx.bank.has_safe:
            break
    return _status(ctx)


# ==========================================================================================
# 2. Quantized PGD inside the one-byte cube
# ==========================================================================================
def quantized_pgd_one_byte(ctx: Context) -> str | None:
    """R latent states a∈[-1,1] (byte units, sparse random start); each step quantizes round(a) to
    {-1,0,+1} bytes, batch-evaluates, then CE-ascends a += α·sign(∇CE) and re-projects into the cube."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    states = []
    for _ in range(K.PGD_R):
        a = torch.rand(n, device=dev) * 2.0 - 1.0
        a[torch.rand(n, device=dev) >= K.PGD_START_PROB] = 0.0  # sparse start
        states.append(a)

    for _ in range(K.PGD_T):
        if _out_of_time(ctx):
            break
        cands = [apply_delta_bytes(ctx.clean_u8, a.round().clamp(-1.0, 1.0) * ctx.k_min, ctx.shape)
                 for a in states]
        _consider(ctx, cands)
        if ctx.bank.has_safe:
            break
        for r, a in enumerate(states):
            _, move_dir, _ = loss_grad(ctx.model, cands[r], ctx.target_index, "ce")  # +sign(∇CE)
            a = (a + K.PGD_ALPHA * move_dir).clamp(-1.0, 1.0)
            a[a.abs() < K.PGD_ZERO_THRESH] = 0.0
            states[r] = a
    return _status(ctx)


# ==========================================================================================
# 3. Quantized DeepFool / FAB-style top-M boundary search
# ==========================================================================================
def quantized_boundary_topM(ctx: Context) -> str | None:
    """Attack several nearby class boundaries z_true - z_t at once (not just top-1 wrong). Candidates
    are always built from the clean image (byte invariant preserved); only the gradient is re-anchored
    each round to the best real-margin candidate."""
    clean_flat = ctx.clean.view(-1)
    anchor = ctx.clean
    for _ in range(K.BOUNDARY_ROUNDS):
        if _out_of_time(ctx):
            break
        top_wrong = top_wrong_classes(logits_of(ctx.model, anchor), ctx.target_index, K.BOUNDARY_M)
        cands: list[torch.Tensor] = []
        for t in top_wrong:
            if _out_of_time(ctx):
                break
            value, move_dir, score = loss_grad(ctx.model, anchor, ctx.target_index, f"pair:{t}")
            msk_score, order, valid = build_sparse_order(score, move_dir, clean_flat)
            if valid == 0:
                continue
            k_need = estimate_k(value + max(ctx.kappa, 0.0), msk_score[order], ctx.q)
            cands += [apply_byte(ctx.clean_u8, move_dir, order[:k], ctx.k_min, ctx.shape)
                      for k in _k_ladder(k_need, valid)]
        res = _consider(ctx, cands)
        if ctx.bank.has_safe:
            break
        if res:  # re-anchor to the lowest real margin (not the predicted best)
            anchor = min(res, key=lambda r: r["margin"])["cand"]
    return _status(ctx)


# ==========================================================================================
# 4. Quantized SparseFool / JSMA-style greedy saliency
# ==========================================================================================
def quantized_saliency_greedy(ctx: Context) -> str | None:
    """Soft-margin saliency, grown in cumulative chunks (batch-tested), then a second greedy pass
    re-linearized at the best candidate with a sharper temperature."""
    clean_flat = ctx.clean.view(-1)
    top_wrong = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.QFGSM_TOPM)
    _, move_dir, score = loss_grad(
        ctx.model, ctx.clean, ctx.target_index, "soft", top_wrong=top_wrong, tau=K.SALIENCY_TAU
    )
    _, order, valid = build_sparse_order(score, move_dir, clean_flat)
    if valid == 0:
        return _status(ctx)

    cands, selected = [], 0
    for chunk in K.SALIENCY_CHUNKS:
        selected = min(valid, selected + int(chunk))
        cands.append(apply_byte(ctx.clean_u8, move_dir, order[:selected], ctx.k_min, ctx.shape))
        if selected >= valid:
            break
    res = _consider(ctx, cands)
    if ctx.bank.has_safe or _out_of_time(ctx) or not res:
        return _status(ctx)

    # Second pass: refresh saliency at the best-margin candidate, sharper tau.
    best = min(res, key=lambda r: r["margin"])
    _, move_dir2, score2 = loss_grad(
        ctx.model, best["cand"], ctx.target_index, "soft", top_wrong=top_wrong, tau=K.SALIENCY_TAU * 0.5
    )
    _, order2, valid2 = build_sparse_order(score2, move_dir2, clean_flat)
    if valid2 > 0:
        cands2 = [apply_byte(ctx.clean_u8, move_dir2, order2[:min(int(k), valid2)], ctx.k_min, ctx.shape)
                  for k in K.SALIENCY_PASS2_KS]
        _consider(ctx, cands2)
    return _status(ctx)


# ==========================================================================================
# 5. Gradient-seeded Square / block random search
# ==========================================================================================
def gradient_seeded_square_search(ctx: Context) -> str | None:
    """Last resort when gradients mislead: from a zero perturbation seeded by sign(∇CE), repeatedly
    toggle random ±1-byte square blocks (gradient-signed most of the time, random otherwise), keeping
    mutations that reduce the real margin. Every channel stays within one byte, so L∞ holds at q."""
    c, h, w = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2])
    dev = ctx.clean_u8.device
    _, grad_dir, _ = loss_grad(ctx.model, ctx.clean, ctx.target_index, "ce")  # +sign(∇CE)
    grad_dir2d = grad_dir.view(c, h, w)
    sizes = [s for s in K.SQUARE_SIZES if 1 <= s <= min(h, w)] or [1]

    base = torch.zeros(c, h, w, device=dev)  # current accepted delta in {-1,0,+1}
    best_margin = ctx.m0
    rnd = 0
    while not _out_of_time(ctx):
        cands, trials = [], []
        for b in range(K.SQUARE_BATCH):
            dt = base.clone()
            size = sizes[(rnd * K.SQUARE_BATCH + b) % len(sizes)]
            top = int(torch.randint(0, max(1, h - size + 1), (1,)).item())
            left = int(torch.randint(0, max(1, w - size + 1), (1,)).item())
            ch = int(torch.randint(0, c, (1,)).item())
            if torch.rand(1).item() < K.SQUARE_GRAD_PROB:
                blk = grad_dir2d[ch, top:top + size, left:left + size].sign()
            else:
                blk = torch.randint(0, 2, (size, size), device=dev).float() * 2.0 - 1.0
            dt[ch, top:top + size, left:left + size] = blk
            cands.append(apply_delta_bytes(ctx.clean_u8, dt.view(-1) * ctx.k_min, ctx.shape))
            trials.append(dt)
        res = _consider(ctx, cands)
        if ctx.bank.has_safe:
            break
        if res:
            jbest = min(range(len(res)), key=lambda j: res[j]["margin"])
            if res[jbest]["margin"] < best_margin:
                best_margin = res[jbest]["margin"]
                base = trials[jbest]
        rnd += 1
    return _status(ctx)


# ==========================================================================================
# Pipeline + orchestrators
# ==========================================================================================
# Single source of truth for which algorithms run, in spec order.
# Comment a line out to exclude that algorithm from EVERY orchestrator.
PIPELINE = [
    # batched_multi_loss_qfgsm,
    # quantized_pgd_one_byte,
    quantized_boundary_topM,
    # quantized_saliency_greedy,
    # gradient_seeded_square_search,
]


def find_flip_first_hit(ctx: Context) -> str | None:
    """Strict spec: return the instant any algorithm produces a flip."""
    for algo in PIPELINE:
        if _out_of_time(ctx):
            break
        if algo(ctx) in ("safe", "flip"):
            return _status(ctx)
    return _status(ctx)


def find_flip_first_safe(ctx: Context) -> str | None:
    """Run in order, stop launching new algorithms once a safe flip exists; the Bank still keeps the
    sparsest safe candidate seen across the algorithms that did run."""
    for algo in PIPELINE:
        if _out_of_time(ctx):
            break
        algo(ctx)
        if ctx.bank.has_safe:
            break
    return _status(ctx)


def find_flip_run_all(ctx: Context) -> str | None:
    """Run every enabled algorithm to the budget, then return the globally sparsest safe flip."""
    for algo in PIPELINE:
        if _out_of_time(ctx):
            break
        algo(ctx)
    return _status(ctx)


def _prefix_deltas(ctx: Context, move_dir, score, clean_flat, ratios, base=None) -> list[torch.Tensor]:
    """Byte deltas for each prefix ratio (top-|g| feasible channels), built fresh (base=None) or by
    overwriting an anchor's existing delta (base given). ratio=1.0 yields the dense candidate."""
    _, order, valid = build_sparse_order(score, move_dir, clean_flat)
    if valid == 0:
        return []
    km = float(ctx.k_min)
    out = []
    for r in ratios:
        k = max(1, min(int(round(r * valid)), valid))
        d = base.clone() if base is not None else torch.zeros_like(ctx.clean_u8)
        idx = order[:k]
        d[idx] = move_dir[idx] * km
        out.append(d)
    return out


def _eval_deltas(ctx: Context, deltas: list[torch.Tensor]) -> list[dict]:
    """Eval a list of byte deltas; fold into the bank; return results with their delta attached."""
    if not deltas:
        return []
    cands = [apply_delta_bytes(ctx.clean_u8, d, ctx.shape) for d in deltas]
    res = _consider(ctx, cands)
    for r, d in zip(res, deltas):
        r["delta"] = d
    return res


def find_beam_byte_pgd(ctx: Context) -> str | None:
    """Multi-target Beam Byte-PGD (standalone orchestrator; ignores PIPELINE).

    Flip-FIRST for high-margin images. Stage 1: from the clean image take several gradients — hard
    top-1 (z_t-z_y), soft top-M, and targeted top-2..top-(T-1) pair losses. Stage 2: build a bank of
    q=1 sign patterns at BEAM_INIT_RATIOS (incl. the full dense 1.0) from each gradient, batch-eval,
    and keep a BEAM of the BEAM_SIZE lowest REAL-margin candidates (not just flips — walk toward the
    boundary). Stage 3 (to the budget): for each beam anchor re-linearize a soft top-M gradient (sharper
    tau) and emit 'replace' (update the anchor's bytes) + 'fresh' patterns at BEAM_ROUND_RATIOS; re-eval
    and refresh the beam. If a round buys no margin, inject random byte repairs around the best anchor
    for diversity. Returns 'safe' the instant margin<=-kappa, else the best margin<0 at timeout. Every
    candidate is an exact ±k_min-byte edit, so L∞ stays at q. No minimal-prefix search up front."""
    clean_flat = ctx.clean.view(-1)
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)

    # Stage 1: gradients from the clean image (hard top-1, soft top-M, targeted top-2..top-(T-1)).
    top = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index,
                            max(K.BEAM_TOPM, K.BEAM_TARGET_CLASSES))
    grads = []
    _, md, sc = loss_grad(ctx.model, ctx.clean, ctx.target_index, f"pair:{top[0]}")           # hard top-1
    grads.append((md, sc))
    _, md, sc = loss_grad(ctx.model, ctx.clean, ctx.target_index, "soft",
                          top_wrong=top[:K.BEAM_TOPM], tau=K.BEAM_TAU)                         # soft top-M
    grads.append((md, sc))
    for t in top[1:max(1, K.BEAM_TARGET_CLASSES)]:                                             # targeted top-2..
        if _out_of_time(ctx):
            break
        _, md, sc = loss_grad(ctx.model, ctx.clean, ctx.target_index, f"pair:{t}")
        grads.append((md, sc))

    # Stage 2: initial bank + beam.
    deltas: list[torch.Tensor] = []
    for md, sc in grads:
        deltas += _prefix_deltas(ctx, md, sc, clean_flat, K.BEAM_INIT_RATIOS)
    res = _eval_deltas(ctx, deltas)
    if ctx.bank.has_safe:
        return "safe"
    beam = sorted(res, key=lambda r: r["margin"])[:K.BEAM_SIZE]

    # Stage 3: beam re-linearization to the budget.
    rounds = 0
    while beam and not _out_of_time(ctx):
        if K.BEAM_ROUNDS > 0 and rounds >= K.BEAM_ROUNDS:
            break
        rounds += 1
        new_deltas: list[torch.Tensor] = []
        for anchor in beam:
            if _out_of_time(ctx):
                break
            a_top = top_wrong_classes(logits_of(ctx.model, anchor["cand"]), ctx.target_index, K.BEAM_ROUND_TOPM)
            _, md, sc = loss_grad(ctx.model, anchor["cand"], ctx.target_index, "soft",
                                  top_wrong=a_top, tau=K.BEAM_ROUND_TAU)
            new_deltas += _prefix_deltas(ctx, md, sc, clean_flat, K.BEAM_ROUND_RATIOS, base=anchor["delta"])
            new_deltas += _prefix_deltas(ctx, md, sc, clean_flat, K.BEAM_ROUND_RATIOS)
        if not new_deltas:
            break
        res2 = _eval_deltas(ctx, new_deltas)
        if ctx.bank.has_safe:
            return "safe"
        prev_best = beam[0]["margin"]
        beam = sorted(beam + res2, key=lambda r: r["margin"])[:K.BEAM_SIZE]

        # Stalled beam -> random byte repairs around the best anchor (diversity, spends the budget).
        if beam[0]["margin"] >= prev_best - 1e-6 and not _out_of_time(ctx):
            base = beam[0]["delta"]
            rep = []
            for p in K.BEAM_REPAIR_PROBS:
                d = base.clone()
                m = torch.rand(n, device=dev) < p
                d[m] = (torch.randint(0, 3, (int(m.sum().item()),), device=dev).float() - 1.0) * km  # {-1,0,+1}
                rep.append(d)
            res3 = _eval_deltas(ctx, rep)
            if ctx.bank.has_safe:
                return "safe"
            beam = sorted(beam + res3, key=lambda r: r["margin"])[:K.BEAM_SIZE]
    return _status(ctx)                       # 'flip' if any margin<0 seen, else None


def _population_children(ctx: Context, anchor_a, move_dir, score, clean_flat) -> list[torch.Tensor]:
    """Build the next latent population around the best anchor: a dense q=1 candidate, semi-dense
    top-k prefixes (incl. the full dense ratio=1.0), two PGD continuations, and a few drop mutations.
    Every entry is a latent a∈[-1,1]; quantized to {-1,0,+1} bytes at eval time."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    pop = [move_dir.clone()]                                   # 1) dense q=1 in the gradient direction
    _, order, valid = build_sparse_order(score, move_dir, clean_flat)
    if valid > 0:                                             # 2) semi-dense top-k prefixes
        for ratio in K.POP_PREFIX_RATIOS:
            k = max(1, min(int(round(ratio * valid)), valid))
            a = torch.zeros(n, device=dev)
            idx = order[:k]
            a[idx] = move_dir[idx]
            pop.append(a)
    for alpha in K.POP_PGD_ALPHAS:                            # 3) PGD continuation from the anchor
        pop.append((anchor_a + alpha * move_dir).clamp(-1.0, 1.0))
    for drop_prob in K.POP_DROP_PROBS:                        # 4) small drop mutations around the anchor
        a = anchor_a.clone()
        drop = (a.round().abs() > 0) & (torch.rand(n, device=dev) < drop_prob)
        a[drop] = 0.0
        pop.append(a)
    return pop


def find_population_pgd(ctx: Context) -> str | None:
    """Batched Multi-Target Quantized PGD (standalone orchestrator; ignores PIPELINE).

    Flip-FIRST for high-margin images (m0 ~ 3-7): keep a POPULATION of q=1 latent perturbations
    a∈[-1,1] (delta = round(a) ∈ {-1,0,+1}), batch-evaluate them all, keep the lowest real margin, and
    re-linearize the SOFT top-M gradient at that best candidate. The next population mixes a dense q=1
    direction, semi-dense top-k prefixes (incl. full dense), PGD continuations, and drop mutations — so
    each round both tests flipping POWER (dense) and starts trimming |S| (prefixes). Returns 'safe' the
    instant margin<=-kappa, else the best margin<0 when the budget runs out. Candidates are exact byte
    edits, so L∞ stays at q. No early minimal-prefix search: find any flip first, sparsify later."""
    clean_flat = ctx.clean.view(-1)
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device

    # Initial population: zero start + sparse random ±1 starts.
    population = [torch.zeros(n, device=dev)]
    for p in K.POP_INIT_DENSITIES:
        a = torch.zeros(n, device=dev)
        mask = torch.rand(n, device=dev) < p
        a[mask] = torch.randint(0, 2, (int(mask.sum().item()),), device=dev).float() * 2.0 - 1.0
        population.append(a)

    best_margin = float("inf")
    best_a = None
    rounds = 0
    while not _out_of_time(ctx):
        if K.POP_MAX_ROUNDS > 0 and rounds >= K.POP_MAX_ROUNDS:
            break
        rounds += 1
        cands = [apply_delta_bytes(ctx.clean_u8, a.round().clamp(-1.0, 1.0) * ctx.k_min, ctx.shape)
                 for a in population]
        res = _consider(ctx, cands)
        if ctx.bank.has_safe:                                 # margin <= -kappa -> done
            return "safe"
        if not res:
            break
        bi = min(range(len(res)), key=lambda i: res[i]["margin"])
        if res[bi]["margin"] < best_margin:
            best_margin, best_a = res[bi]["margin"], population[bi].clone()
        anchor_a = best_a if best_a is not None else population[bi]
        anchor = apply_delta_bytes(ctx.clean_u8, anchor_a.round().clamp(-1.0, 1.0) * ctx.k_min, ctx.shape)

        # Re-linearize the soft top-M gradient at the best real-margin candidate (sharper tau near 0).
        top_wrong = top_wrong_classes(logits_of(ctx.model, anchor), ctx.target_index, K.POP_TOPM)
        tau = K.POP_TAU_NEAR if abs(best_margin) < K.POP_NEAR_THRESH else K.POP_TAU
        _, move_dir, score = loss_grad(ctx.model, anchor, ctx.target_index, "soft", top_wrong=top_wrong, tau=tau)
        population = _population_children(ctx, anchor_a, move_dir, score, clean_flat)
    return _status(ctx)                                       # 'flip' if any margin<0 seen, else None


# ------------------------------------------------------------------------------------------
# find_ensemble_byte_pgd helpers: ensemble directions + structured mutation + efficiency.
# ------------------------------------------------------------------------------------------
def _normalize(v: torch.Tensor) -> torch.Tensor:
    nrm = float(v.norm().item())
    return v / nrm if nrm > 0 else v


def _signmix(move_dir_a, score_a, move_dir_b, score_b):
    """Sign-mix two gradients: normalize each descent vector (move_dir·|g|), sum, take sign + |.|.
    No extra backward — reuses gradients already computed."""
    mix = _normalize(move_dir_a * score_a) + _normalize(move_dir_b * score_b)
    return mix.sign(), mix.abs()


def _opposite_deltas(ctx: Context, move_dir, score, clean_flat, ks) -> list[torch.Tensor]:
    """Opposite-sign probe on the strongest top-k channels — a small hedge against locally wrong
    gradient signs. Limited to a few k (never a dense opposite update, which is wasteful)."""
    _, order, valid = build_sparse_order(score, move_dir, clean_flat)
    if valid == 0:
        return []
    km = float(ctx.k_min)
    out = []
    for k in ks:
        k = min(int(k), valid)
        if k < 1:
            continue
        d = torch.zeros_like(ctx.clean_u8)
        idx = order[:k]
        d[idx] = -move_dir[idx] * km          # opposite of the gradient move
        out.append(d)
    return out


def _structured_mutations(ctx: Context, base, move_dir, score, clean_flat) -> list[torch.Tensor]:
    """Structured byte mutations around the best candidate (used only when near the boundary):
    drop active channels, flip active signs, add next-ranked inactive channels, and replace the
    weakest-saliency active channels with the strongest inactive ones."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    _, order, valid = build_sparse_order(score, move_dir, clean_flat)
    active = base.abs() > 0
    inactive_ranked = order[~active[order]]                  # feasible inactive channels, by saliency
    out = []
    for p in K.ENS_MUTATE_PROBS:                             # drop a fraction of active channels
        d = base.clone()
        d[active & (torch.rand(n, device=dev) < p)] = 0.0
        out.append(d)
    for p in K.ENS_MUTATE_PROBS:                             # flip a fraction of active signs
        d = base.clone()
        flip = active & (torch.rand(n, device=dev) < p)
        d[flip] = -d[flip]
        out.append(d)
    for a in K.ENS_MUTATE_ADD:                               # add next-ranked inactive channels
        d = base.clone()
        idx = inactive_ranked[:min(int(a), inactive_ranked.numel())]
        d[idx] = move_dir[idx] * km
        out.append(d)
    if active.any():                                         # replace weakest active with strongest inactive
        active_idx = active.nonzero(as_tuple=True)[0]
        weak = active_idx[torch.argsort(score[active_idx])]  # weakest saliency first
        for a in K.ENS_MUTATE_ADD:
            cnt = min(int(a), weak.numel(), inactive_ranked.numel())
            if cnt < 1:
                continue
            d = base.clone()
            d[weak[:cnt]] = 0.0
            d[inactive_ranked[:cnt]] = move_dir[inactive_ranked[:cnt]] * km
            out.append(d)
    return out


def _eval_with_eff(ctx: Context, deltas, score) -> list[dict]:
    """Eval deltas and attach margin-drop efficiency = (m0 - margin) / (q·Σ|g_i| over changed channels).
    score is the |g| that produced the deltas (None for opposite/mutation probes -> eff = inf)."""
    res = _eval_deltas(ctx, deltas)
    for r in res:
        if score is None:
            r["eff"] = float("inf")
            continue
        changed = r["delta"] != 0
        pred = ctx.q * float(score[changed].sum().item())
        r["eff"] = (ctx.m0 - r["margin"]) / max(pred, 1e-9) if pred > 0 else float("inf")
    return res


def _ens_pick_beam(pool: list[dict], size: int) -> list[dict]:
    """Beam = lowest real margin, but prefer EFFICIENT anchors (real drop tracked the linear prediction);
    fall back to raw margin order if nothing is efficient. Abandons directions the gradient lied about."""
    efficient = [r for r in pool if r.get("eff", float("inf")) >= K.ENS_EFF_FRAC]
    chosen = efficient if efficient else pool
    return sorted(chosen, key=lambda r: r["margin"])[:size]


def find_ensemble_byte_pgd(ctx: Context) -> str | None:
    """beam Byte-PGD backbone + ensemble directions + structured discrete mutation (standalone).

    The merged/strongest version. Round 0 (clean image, 3 gradients): soft top-M, targeted top-1, and a
    DLR-normalized margin gradient; from each it banks q=1 prefixes (incl. dense), plus a SIGN-MIX of the
    soft+targeted directions and limited OPPOSITE-sign probes on the strongest channels. Keeps a beam of
    the ENS_BEAM_SIZE lowest real-margin EFFICIENT candidates. Later rounds (best anchor only, 2 gradients
    + sign-mix) re-linearize and emit 'replace'/'fresh' prefixes; once |best margin| < ENS_MUTATE_THRESH
    it also runs STRUCTURED mutations (drop/flip/add/replace active channels). Efficiency pruning abandons
    a direction when the real margin barely follows the linear prediction. Returns 'safe' on margin<=-kappa,
    else the best margin<0 at timeout. Exact ±k_min-byte edits, so L∞ stays at q."""
    clean_flat = ctx.clean.view(-1)

    # Round 0: three gradients at the clean image (soft top-M, targeted top-1, DLR).
    top = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.ENS_TOPM)
    _, md_soft, sc_soft = loss_grad(ctx.model, ctx.clean, ctx.target_index, "soft", top_wrong=top, tau=K.ENS_TAU)
    _, md_tgt, sc_tgt = loss_grad(ctx.model, ctx.clean, ctx.target_index, f"pair:{top[0]}")
    _, md_dlr, sc_dlr = loss_grad(ctx.model, ctx.clean, ctx.target_index, "dlr")

    pool: list[dict] = []
    for md, sc in ((md_soft, sc_soft), (md_tgt, sc_tgt), (md_dlr, sc_dlr)):
        pool += _eval_with_eff(ctx, _prefix_deltas(ctx, md, sc, clean_flat, K.ENS_INIT_RATIOS), sc)
        if ctx.bank.has_safe:
            return "safe"
    mm, ms = _signmix(md_soft, sc_soft, md_tgt, sc_tgt)                       # sign-mix soft+targeted
    pool += _eval_with_eff(ctx, _prefix_deltas(ctx, mm, ms, clean_flat, K.ENS_INIT_RATIOS), ms)
    if ctx.bank.has_safe:
        return "safe"
    pool += _eval_with_eff(ctx, _opposite_deltas(ctx, md_soft, sc_soft, clean_flat, K.ENS_OPP_KS), None)
    if ctx.bank.has_safe:
        return "safe"
    beam = _ens_pick_beam(pool, K.ENS_BEAM_SIZE)

    # Later rounds: best anchor only (bounded gradient budget), to the time budget.
    rounds = 0
    while beam and not _out_of_time(ctx):
        if K.ENS_ROUNDS > 0 and rounds >= K.ENS_ROUNDS:
            break
        rounds += 1
        best = beam[0]
        a_top = top_wrong_classes(logits_of(ctx.model, best["cand"]), ctx.target_index, K.ENS_TOPM)
        _, md_s, sc_s = loss_grad(ctx.model, best["cand"], ctx.target_index, "soft", top_wrong=a_top, tau=K.ENS_TAU_NEAR)
        _, md_t, sc_t = loss_grad(ctx.model, best["cand"], ctx.target_index, f"pair:{a_top[0]}")
        new: list[dict] = []
        for md, sc in ((md_s, sc_s), (md_t, sc_t)):
            new += _eval_with_eff(ctx, _prefix_deltas(ctx, md, sc, clean_flat, K.ENS_ROUND_RATIOS, base=best["delta"]), sc)
            new += _eval_with_eff(ctx, _prefix_deltas(ctx, md, sc, clean_flat, K.ENS_ROUND_RATIOS), sc)
            if ctx.bank.has_safe:
                return "safe"
        mm, ms = _signmix(md_s, sc_s, md_t, sc_t)
        new += _eval_with_eff(ctx, _prefix_deltas(ctx, mm, ms, clean_flat, K.ENS_ROUND_RATIOS, base=best["delta"]), ms)
        if ctx.bank.has_safe:
            return "safe"
        if abs(best["margin"]) < K.ENS_MUTATE_THRESH and not _out_of_time(ctx):  # structured mutation near the boundary
            new += _eval_with_eff(ctx, _structured_mutations(ctx, best["delta"], md_s, sc_s, clean_flat), None)
            if ctx.bank.has_safe:
                return "safe"
        beam = _ens_pick_beam(beam + new, K.ENS_BEAM_SIZE)
    return _status(ctx)                                       # 'flip' if any margin<0 seen, else None


# ==========================================================================================
# find_hydra: Q1-Hydra finder — primitive-space search (tiles / low-freq / color / beam).
# ==========================================================================================
def _tile_boxes(h: int, w: int, size: int) -> list[tuple[int, int, int, int]]:
    """Cover the HxW grid with (r0, r1, c0, c1) boxes of side `size`."""
    boxes = []
    for r in range(0, h, size):
        for c in range(0, w, size):
            boxes.append((r, min(r + size, h), c, min(c + size, w)))
    return boxes


def _box_mask(ctx: Context, size: int, dev) -> torch.Tensor:
    """Random spatial box (all channels) as a flat bool mask."""
    c, h, w = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2])
    size = max(1, min(size, h, w))
    r = int(torch.randint(0, h - size + 1, (1,)).item())
    cc = int(torch.randint(0, w - size + 1, (1,)).item())
    m = torch.zeros(c, h, w, dtype=torch.bool, device=dev)
    m[:, r:r + size, cc:cc + size] = True
    return m.view(-1)


def _lowfreq_patterns(h: int, w: int, dev) -> list[torch.Tensor]:
    """Spatial sign maps in {-1,+1} of shape [H,W]: gradients, center/border, checkerboards, cosines."""
    rr = torch.arange(h, device=dev).view(h, 1).float()
    cc = torch.arange(w, device=dev).view(1, w).float()
    pats = [torch.sign(cc - (w - 1) / 2), torch.sign(rr - (h - 1) / 2)]          # horizontal / vertical
    rad = torch.sqrt((cc - (w - 1) / 2) ** 2 + (rr - (h - 1) / 2) ** 2)
    pats.append(torch.sign(rad - rad.mean()))                                    # center vs border
    for period in K.HYDRA_LF_PERIODS:                                            # checkerboards
        pats.append(torch.sign(torch.sin(math.pi * rr / period) * torch.sin(math.pi * cc / period)))
    for u, v in ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)):                        # DCT-like cosine signs
        pats.append(torch.sign(torch.cos(math.pi * u * rr / h) * torch.cos(math.pi * v * cc / w)))
    # Broadcast each to a contiguous [H,W] and map 0 -> +1.
    out = []
    for p in pats:
        p = p.expand(h, w).contiguous()
        out.append(torch.where(p == 0, torch.ones_like(p), p))
    return out


def _hydra_tile_deltas(ctx: Context, move_dir, score) -> list[torch.Tensor]:
    """Coherent spatial-tile candidates: rank tiles by gradient mass, move the top-N tiles as a union."""
    c, h, w = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2])
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    sc2 = score.view(c, h, w)
    out = []
    for size in K.HYDRA_TILE_SIZES:
        size = max(1, min(int(size), h, w))
        boxes = _tile_boxes(h, w, size)
        tscore = torch.tensor([float(sc2[:, r0:r1, c0:c1].sum().item()) for (r0, r1, c0, c1) in boxes])
        order = torch.argsort(tscore, descending=True).tolist()
        for num in K.HYDRA_TILE_COUNTS:
            num = min(int(num), len(boxes))
            if num < 1:
                continue
            mask = torch.zeros(c, h, w, dtype=torch.bool, device=dev)
            for i in order[:num]:
                r0, r1, c0, c1 = boxes[i]
                mask[:, r0:r1, c0:c1] = True
            m = mask.view(-1)
            d = torch.zeros_like(ctx.clean_u8)
            d[m] = move_dir[m] * km
            out.append(d)
    return out


def _hydra_lowfreq_deltas(ctx: Context, move_dir) -> list[torch.Tensor]:
    """Pure and gradient-modulated low-frequency ±1 patterns (dense, broadcast over channels)."""
    c, h, w = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2])
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    md2 = move_dir.view(c, h, w)
    out = []
    for p in _lowfreq_patterns(h, w, dev):
        pb = p.view(1, h, w).expand(c, h, w)
        out.append((pb.reshape(-1) * km).clone())                    # pure low-frequency
        out.append((torch.sign(md2 + pb).reshape(-1)) * km)          # gradient-modulated
    return out


def _hydra_color_deltas(ctx: Context) -> list[torch.Tensor]:
    """Per-channel and whole-image ±1 byte biases (nearly free coverage)."""
    c, h, w = int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2])
    km = float(ctx.k_min)
    out = []
    for ch in range(c):
        for s in (-1.0, 1.0):
            d = torch.zeros_like(ctx.clean_u8).view(c, h, w)
            d[ch] = s * km
            out.append(d.reshape(-1))
    for s in (-1.0, 1.0):
        out.append(torch.full_like(ctx.clean_u8, s * km))
    return out


def _hydra_beam(ctx: Context, seeds, move_dir, lf_deltas, color_deltas) -> str | None:
    """Primitive-space beam: mutate byte deltas by add/remove/flip tile, add low-freq, add color, or
    crossover — all clipped to the ternary cube. Keep the lowest-margin survivors; loop to the budget."""
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    sizes = [int(s) for s in K.HYDRA_TILE_SIZES if int(s) <= min(int(ctx.shape[1]), int(ctx.shape[2]))]
    sizes = sizes or [min(int(ctx.shape[1]), int(ctx.shape[2]))]
    beam = [s.clone() for s in seeds]
    rounds = 0
    while beam and not _out_of_time(ctx):
        if K.HYDRA_BEAM_ROUNDS > 0 and rounds >= K.HYDRA_BEAM_ROUNDS:
            break
        rounds += 1
        trials = []
        for d in beam:
            for _ in range(K.HYDRA_MUT_PER):
                t = d.clone()
                op = int(torch.randint(0, 6, (1,)).item())
                size = sizes[int(torch.randint(0, len(sizes), (1,)).item())]
                if op == 0:                                  # add tile (gradient sign)
                    m = _box_mask(ctx, size, dev)
                    t[m] = move_dir[m] * km
                elif op == 1:                                # remove tile
                    t[_box_mask(ctx, size, dev)] = 0.0
                elif op == 2:                                # flip tile sign
                    m = _box_mask(ctx, size, dev)
                    t[m] = -t[m]
                elif op == 3 and lf_deltas:                  # add low-frequency primitive
                    t = (t + lf_deltas[int(torch.randint(0, len(lf_deltas), (1,)).item())]).clamp(-km, km)
                elif op == 4 and color_deltas:               # add color primitive
                    t = (t + color_deltas[int(torch.randint(0, len(color_deltas), (1,)).item())]).clamp(-km, km)
                else:                                        # crossover with another beam member
                    other = beam[int(torch.randint(0, len(beam), (1,)).item())]
                    m = _box_mask(ctx, size, dev)
                    t[m] = other[m]
                trials.append(t)
        if not trials:
            break
        res = _eval_deltas(ctx, trials)
        if ctx.bank.has_safe:
            return "safe"
        beam = [r["delta"] for r in sorted(res, key=lambda r: r["margin"])[:K.HYDRA_BEAM_SIZE]]
    return _status(ctx)


def find_hydra(ctx: Context) -> str | None:
    """Q1-Hydra finder (standalone orchestrator; ignores PIPELINE).

    With q pinned to 1, strength can't grow — only the STRUCTURE of the ±1 pattern can. Hydra banks
    candidate families that cover different flip structures: global gradient prefixes, coherent spatial
    TILE unions, low-FREQUENCY global patterns, and per-CHANNEL color biases — from a soft top-M and a
    CE gradient. Then a primitive-space beam search mutates tiles/low-freq/color/crossover under real
    margin feedback to the budget. Returns 'safe' on margin<=-kappa, else best margin<0. Exact ±k_min
    byte edits throughout, so L∞ stays at q."""
    clean_flat = ctx.clean.view(-1)
    top = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index, K.HYDRA_TOPM)
    _, md_soft, sc_soft = loss_grad(ctx.model, ctx.clean, ctx.target_index, "soft", top_wrong=top, tau=K.HYDRA_TAU)
    _, md_ce, sc_ce = loss_grad(ctx.model, ctx.clean, ctx.target_index, "ce")

    lf = _hydra_lowfreq_deltas(ctx, md_soft)
    color = _hydra_color_deltas(ctx)
    deltas: list[torch.Tensor] = []
    for md, sc in ((md_soft, sc_soft), (md_ce, sc_ce)):
        deltas += _prefix_deltas(ctx, md, sc, clean_flat, K.HYDRA_PREFIX_RATIOS)
        deltas += _hydra_tile_deltas(ctx, md, sc)
    deltas += lf + color

    res = _eval_deltas(ctx, deltas)
    if ctx.bank.has_safe:
        return "safe"
    if not res:
        return _status(ctx)
    seeds = [r["delta"] for r in sorted(res, key=lambda r: r["margin"])[:K.HYDRA_BEAM_SIZE]]
    return _hydra_beam(ctx, seeds, md_soft, lf, color)


# ==========================================================================================
# find_apgd_dlr: Quantized APGD-DLR with batched restarts.
# ==========================================================================================
def _mutate_ternary_latent(base: torch.Tensor, dev) -> torch.Tensor:
    """Random ternary mutation of a latent u: reset a small random subset to {-1,0,+1}."""
    n = base.numel()
    u = base.clone()
    m = torch.rand(n, device=dev) < 0.05
    u[m] = torch.randint(0, 3, (int(m.sum().item()),), device=dev).float() - 1.0
    return u


def find_apgd_dlr(ctx: Context) -> str | None:
    """Quantized APGD-DLR with batched restarts (standalone orchestrator; ignores PIPELINE).

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
# find_apgd_targeted: targeted APGD-L∞ (DLR-T) + momentum + restarts.
# ==========================================================================================
def find_apgd_targeted(ctx: Context) -> str | None:
    """Targeted Auto-PGD on the L∞ byte cube (standalone orchestrator; ignores PIPELINE).

    The recommended live finder for confident/high-margin images. Runs, in one batch, a targeted-DLR
    trajectory against each of the top-APGDT_TARGETS runner-up classes (+ one untargeted DLR), each with
    momentum, adaptive step, and restart-from-best. Targeting concentrates the diffuse gradient and DLR
    avoids the softmax saturation that masks the gradient at p_true~0.9+. Latent u∈[-1,1], delta=round(u);
    exact ±k_min byte edits. Runs to the budget; only returns early on a safe flip — never bails."""
    n = ctx.clean_u8.numel()
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    top = top_wrong_classes(logits_of(ctx.model, ctx.clean), ctx.target_index,
                            max(K.APGDT_TOPM, K.APGDT_TARGETS))
    kinds = [f"dlrt:{t}" for t in top[:max(1, K.APGDT_TARGETS)]]
    if K.APGDT_UNTARGETED:
        kinds.append("dlr")
    states = [{"u": torch.zeros(n, device=dev), "prev": torch.zeros(n, device=dev),
               "alpha": K.APGDT_ALPHA0, "best": float("inf"), "stale": 0,
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
                if s["stale"] >= K.APGDT_PATIENCE:           # halve step + restart from this state's best
                    s["alpha"] = max(K.APGDT_MIN_ALPHA, s["alpha"] * 0.5)
                    s["stale"], s["u"] = 0, s["ubest"].clone()
        for s, r in zip(states, res):
            if _out_of_time(ctx):
                break
            _, move_dir, _ = loss_grad(ctx.model, r["cand"], ctx.target_index, s["kind"])
            z = (s["u"] + s["alpha"] * move_dir).clamp(-1.0, 1.0)        # APGD step (sign direction)
            u_new = (z + K.APGDT_MOMENTUM * (s["u"] - s["prev"])).clamp(-1.0, 1.0)  # momentum
            s["prev"], s["u"] = s["u"], u_new
    return _status(ctx)


# ==========================================================================================
# find_fmn / find_alma: minimum-norm-style support finders (continuous step -> byte snap).
# ==========================================================================================
def _project_l1(v: torch.Tensor, z: float) -> torch.Tensor:
    """Euclidean projection of v onto the L1 ball {||w||_1 <= z} (Duchi et al.). Promotes sparsity."""
    if z <= 0:
        return torch.zeros_like(v)
    absv = v.abs()
    if float(absv.sum().item()) <= z:
        return v
    su, _ = torch.sort(absv, descending=True)
    css = torch.cumsum(su, dim=0) - z
    rng = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    rho = int((su - css / rng > 0).sum().item())
    theta = css[rho - 1] / rho
    return v.sign() * (absv - theta).clamp(min=0.0)


def _snap_support(ctx: Context, delta_cont: torch.Tensor, clean_flat: torch.Tensor, ratios) -> str | None:
    """Snap a continuous perturbation's support to ±k_min byte prefixes (top-|delta| channels, sign(delta)),
    evaluate, and fold into the bank. Returns 'safe' if a candidate clears -kappa."""
    cands = _prefix_deltas(ctx, torch.sign(delta_cont), delta_cont.abs(), clean_flat, ratios)
    _eval_deltas(ctx, cands)
    return "safe" if ctx.bank.has_safe else None


def find_fmn(ctx: Context) -> str | None:
    """Q1-FMN-L1 support finder (standalone orchestrator; ignores PIPELINE).

    Fast Minimum-Norm flavor used as a SUPPORT finder: take continuous margin-reducing steps under an
    adaptive L1 radius (grow when not adversarial, shrink when adversarial — "breathe" around the
    boundary), L1-project to stay sparse, and each step snap the support to ±k_min byte prefixes and
    verify. Outer loop re-initializes to the budget. The continuous delta is never submitted."""
    x0f = ctx.clean.view(-1)
    n = x0f.numel()
    dev = ctx.clean_u8.device
    eps0 = K.FMN_EPS0_FRAC * n * ctx.q
    while not _out_of_time(ctx):
        delta = torch.zeros(n, device=dev)
        eps_l1 = eps0
        for _ in range(K.FMN_STEPS):
            if _out_of_time(ctx):
                break
            x = (x0f + delta).clamp(0.0, 1.0).view(ctx.shape)
            m, move_dir, _ = loss_grad(ctx.model, x, ctx.target_index, "hard")
            eps_l1 = eps_l1 * K.FMN_SHRINK if m < 0 else eps_l1 * K.FMN_GROW   # FMN radius "breathing"
            delta = _project_l1(delta + K.FMN_ALPHA * move_dir, eps_l1)        # step toward flip, L1-project
            delta = (x0f + delta).clamp(0.0, 1.0) - x0f                        # box feasibility
            if _snap_support(ctx, delta, x0f, K.FMN_RATIOS) == "safe":
                return "safe"
    return _status(ctx)


def find_alma(ctx: Context) -> str | None:
    """Q1-ALMA-lite (standalone orchestrator; ignores PIPELINE).

    Augmented-Lagrangian flavor: minimize an L1-ish distance while a multiplier (lam) + penalty (rho) on
    the not-adversarial constraint c=max(m+kappa,0) grows until the margin is crossed — smoothly, so it
    is less jumpy than hard thresholding (good for high-entropy multi-competitor cases). Each step snaps
    the support to byte prefixes and verifies. Outer loop re-initializes to the budget."""
    x0f = ctx.clean.view(-1)
    n = x0f.numel()
    dev = ctx.clean_u8.device
    while not _out_of_time(ctx):
        delta = torch.zeros(n, device=dev)
        lam, rho, stalled = K.ALMA_LAM0, K.ALMA_RHO0, 0
        best_delta, best_m = delta.clone(), float("inf")
        for _ in range(K.ALMA_STEPS):
            if _out_of_time(ctx):
                break
            x = (x0f + delta).clamp(0.0, 1.0).view(ctx.shape)
            m, move_dir, _ = loss_grad(ctx.model, x, ctx.target_index, "hard")
            c = max(m + ctx.kappa, 0.0)
            if m < best_m:
                best_m, best_delta = m, delta.clone()
            penalty = lam + rho * c
            # descend: reduce margin (penalty-weighted) + shrink the L1 norm toward 0
            delta = delta + K.ALMA_ALPHA * penalty * move_dir - K.ALMA_ALPHA * K.ALMA_L1W * torch.sign(delta)
            delta = (x0f + delta).clamp(0.0, 1.0) - x0f
            if c > 0:
                lam = lam + rho * c          # multiplier update
            else:
                stalled += 1
            if stalled >= 2:
                rho, stalled = rho * K.ALMA_RHO_GROW, 0
            if _snap_support(ctx, delta, x0f, K.ALMA_RATIOS) == "safe":
                return "safe"
        if _snap_support(ctx, best_delta, x0f, K.ALMA_RATIOS) == "safe":
            return "safe"
    return _status(ctx)


# ==========================================================================================
# find_frank_wolfe: Q1-Block Frank-Wolfe + Sparse-RS rescue.
# ==========================================================================================
def find_frank_wolfe(ctx: Context) -> str | None:
    """Q1-Block Frank-Wolfe + Sparse-RS rescue (standalone orchestrator; ignores PIPELINE).

    Block top-K Frank-Wolfe: at the current byte delta, the FW oracle picks the K steepest feasible
    coordinates (K from a schedule) to move ±k_min; each round also emits Sparse-RS support mutations
    (add-ranked / remove / swap / flip) and structured variants (union-with-best, drop/replace
    weakest-active). Re-anchors to the best real-margin byte delta and loops to the budget."""
    clean_flat = ctx.clean.view(-1)
    dev = ctx.clean_u8.device
    km = float(ctx.k_min)
    delta = torch.zeros_like(ctx.clean_u8)
    best, best_m, ki = delta.clone(), float("inf"), 0

    while not _out_of_time(ctx):
        x = ((ctx.clean_u8 + delta).clamp(0.0, 255.0) / 255.0).view(ctx.shape)
        m, move_dir, score = loss_grad(ctx.model, x, ctx.target_index, "hard")
        _, order, valid = build_sparse_order(score, move_dir, clean_flat)
        if valid == 0:
            break
        kk = min(int(K.FW_K_SCHEDULE[ki % len(K.FW_K_SCHEDULE)]), valid)
        ki += 1
        idx = order[:kk]
        active = delta != 0
        inactive_ranked = order[~active[order]]

        cands = []
        d = delta.clone(); d[idx] = move_dir[idx] * km; cands.append(d)             # FW: add steepest block
        cands.append(torch.where(best != 0, best, delta))                          # union with best
        if bool(active.any()):
            act = active.nonzero(as_tuple=True)[0]
            weak = act[torch.argsort(score[act])][:max(1, int(K.FW_DROP_RATIO * act.numel()))]
            d2 = delta.clone(); d2[weak] = 0.0; cands.append(d2)                    # drop weakest-active
            d3 = delta.clone(); d3[weak] = 0.0                                      # replace weak with strong
            repl = idx[:weak.numel()]
            d3[repl] = move_dir[repl] * km
            cands.append(d3)
        for _ in range(K.FW_RS_MUT):                                               # Sparse-RS support mutations
            base = best if bool((best != 0).any()) else delta
            d = base.clone()
            op = int(torch.randint(0, 4, (1,)).item())
            if op == 0 and inactive_ranked.numel() > 0:                            # add top-ranked inactive
                add = inactive_ranked[:K.FW_RS_ADD]; d[add] = move_dir[add] * km
            elif op == 1 and bool((d != 0).any()):                                 # remove random active
                act = (d != 0).nonzero(as_tuple=True)[0]
                d[act[torch.randperm(act.numel(), device=dev)[:K.FW_RS_SWAP]]] = 0.0
            elif op == 2 and bool((d != 0).any()):                                 # flip random active signs
                act = (d != 0).nonzero(as_tuple=True)[0]
                sw = act[torch.randperm(act.numel(), device=dev)[:K.FW_RS_SWAP]]; d[sw] = -d[sw]
            elif inactive_ranked.numel() > 0:                                      # add random inactive
                add = inactive_ranked[torch.randperm(inactive_ranked.numel(), device=dev)[:K.FW_RS_ADD]]
                d[add] = move_dir[add] * km
            cands.append(d)

        res = _eval_deltas(ctx, cands)
        if ctx.bank.has_safe:
            return "safe"
        if not res:
            break
        j = min(range(len(res)), key=lambda i: res[i]["margin"])
        if res[j]["margin"] < best_m:
            best_m, best = res[j]["margin"], res[j]["delta"].clone()
        delta = res[j]["delta"].clone()
    return _status(ctx)


# ==========================================================================================
# optim_rmse_*: POST-FLIP RMSE refinement (minimize |S|).
# ------------------------------------------------------------------------------------------
# These are NOT finders. They warm-start from the Bank's best flip (safe preferred) and shrink it.
# With every changed channel pinned at exactly ±k_min bytes, RMSE = q·sqrt(|S|/n), so minimizing
# RMSE == minimizing |S|. Each refiner folds its sparser survivors back into the Bank (ordered by
# |S|, then linf, then rmse), so the global best automatically tracks the sparsest. Acceptance is
# always the validator-faithful batched eval (envelope + SSIM/PSNR + κ); gradients only RANK/PREDICT.
# Run one or several, in order, AFTER a finder at the orchestrator switch in perturb().
# ==========================================================================================
def _warm_delta(ctx: Context) -> torch.Tensor | None:
    """Integer byte delta (flat) of the Bank's best flip — safe preferred — or None if no flip yet."""
    best = ctx.bank.best_safe if ctx.bank.best_safe is not None else ctx.bank.best_flip
    if best is None:
        return None
    cand_u8 = torch.round(best["cand"].view(-1) * 255.0)
    return cand_u8 - ctx.clean_u8


def _warm_delta_cont(ctx: Context) -> torch.Tensor | None:
    """The same warm-start as a continuous [0,1]-space delta (byte·1/255), or None."""
    db = _warm_delta(ctx)
    return None if db is None else db * K.Q


def _accept(ctx: Context, r: dict) -> bool:
    """Is candidate r a valid working base to keep refining? Envelope-safe normally; any quality flip
    when PERTURB_ALLOW_UNSAFE_FLIP=1. (The Bank still independently records the global best.)"""
    if r.get("safe"):
        return True
    return bool(ctx.allow_unsafe and r.get("flipped") and r.get("quality"))


def _margin_grad_bytes(ctx: Context, delta_bytes: torch.Tensor):
    """Hard CW margin and its signed [0,1]-space gradient (flat) at clean + delta_bytes."""
    x = apply_delta_bytes(ctx.clean_u8, delta_bytes, ctx.shape)
    m, g = margin_and_grad(ctx.model, x, ctx.target_index)
    return m, g.view(-1)


def _margin_grad_cont(ctx: Context, delta_cont: torch.Tensor):
    """Hard CW margin and its signed [0,1]-space gradient (flat) at clip(clean + delta_cont)."""
    x = (ctx.clean.view(-1) + delta_cont).clamp(0.0, 1.0).view(ctx.shape)
    m, g = margin_and_grad(ctx.model, x, ctx.target_index)
    return m, g.view(-1)


def optim_rmse_prune(ctx: Context, delta: torch.Tensor | None = None) -> str | None:
    """Stage A — gradient-ranked byte rollback with adaptive batch splitting (RMSE refiner).

    Warm-start from the Bank's best flip (or the passed byte delta). Each round takes ONE backward for the
    hard-margin gradient g at the current adversarial point and predicts the margin cost of removing each
    changed channel (rolling it one byte back to clean): c_i = g_i·(−q·sign(δ_i)). Channels are ranked by
    c_i ascending — counter-productive (c_i<0) and cheap (small c_i) first. A geometric ladder of removal
    counts (plus the linear-predicted safe prefix) is batch-evaluated in ONE forward, and the LARGEST
    removal that stays accept-safe is taken — fewer removals are monotonically safer, so this is exactly
    the spec's gradient-guided delta-debugging / binary split, vectorized. Recompute g and repeat until no
    single channel can be removed (local minimum) or the budget runs out."""
    if delta is None:
        delta = _warm_delta(ctx)
    if delta is None:
        return _status(ctx)
    q = ctx.q
    base = float(K.OPTIM_PRUNE_LADDER) if K.OPTIM_PRUNE_LADDER > 1 else 2.0
    rounds = 0
    while not _out_of_time(ctx):
        if K.OPTIM_PRUNE_MAX_ROUNDS > 0 and rounds >= K.OPTIM_PRUNE_MAX_ROUNDS:
            break
        rounds += 1
        changed = (delta != 0).nonzero(as_tuple=True)[0]
        if changed.numel() == 0:
            break
        m, g = _margin_grad_bytes(ctx, delta)
        if m >= 0.0:                                          # not adversarial here -> nothing safe to prune
            break
        sgn = torch.sign(delta[changed])
        c = g[changed] * (-q * sgn)                           # predicted margin change when removing each
        sort_idx = torch.argsort(c)                           # ascending: most-removable first
        order = changed[sort_idx]
        n = int(order.numel())
        sizes = set()                                         # geometric ladder of removal counts
        s = 1
        while s < n:
            sizes.add(s)
            s = max(s + 1, int(s * base))
        sizes.add(n)
        cum = torch.cumsum(c[sort_idx], dim=0)                # linear-predicted safe prefix
        ok = (m + cum < -ctx.kappa)
        bpred = int(torch.cumprod(ok.to(torch.long), dim=0).sum().item())  # leading-true run length
        if bpred >= 1:
            sizes.add(bpred)
        sizes = sorted(x for x in sizes if 1 <= x <= n)
        cands = []
        for sz in sizes:
            d = delta.clone()
            d[order[:sz]] = 0.0
            cands.append(d)
        res = _eval_deltas(ctx, cands)
        best_d, best_sz = None, 0
        for sz, d, r in zip(sizes, cands, res):
            if _accept(ctx, r) and sz > best_sz:
                best_sz, best_d = sz, d
        if best_d is None:                                    # converged: not even one removal stays safe
            break
        delta = best_d
    return _status(ctx)


def optim_rmse_exchange(ctx: Context) -> str | None:
    """Stage B — one-for-many coordinate exchange (RMSE refiner; breaks the pruning local minimum).

    When deletion stalls every remaining channel looks individually necessary. Add ONE strong unused
    channel j (best one-byte move −sign(g_j), predicted margin gain −q·|g_j|) and remove the cheapest
    ≥OPTIM_EXCHANGE_MIN_DROP existing channels its slack can pay for, so |S| strictly drops. Per round
    (one backward): rank changed channels by removal cost c_i ascending; for each of the top-ADDS unused
    movable channels build a candidate that adds it and removes as many cheap channels as the margin
    budget (−κ − m) + q·|g_j| allows. Batch-eval; keep the accept-safe candidate with the lowest |S|;
    repeat until no exchange helps or the budget runs out."""
    delta = _warm_delta(ctx)
    if delta is None:
        return _status(ctx)
    q, km = ctx.q, float(ctx.k_min)
    clean_flat = ctx.clean.view(-1)
    mindrop = max(2, int(K.OPTIM_EXCHANGE_MIN_DROP))
    rounds = 0
    while not _out_of_time(ctx):
        if K.OPTIM_EXCHANGE_MAX_ROUNDS > 0 and rounds >= K.OPTIM_EXCHANGE_MAX_ROUNDS:
            break
        rounds += 1
        m, g = _margin_grad_bytes(ctx, delta)
        if m >= 0.0:
            break
        changed_mask = delta != 0
        changed = changed_mask.nonzero(as_tuple=True)[0]
        nchanged = int(changed.numel())
        if nchanged == 0:
            break
        sgn = torch.sign(delta[changed])                      # cheapest-to-remove existing channels first
        c = g[changed] * (-q * sgn)
        c_order = torch.argsort(c)
        rem_order = changed[c_order]
        cum = torch.cumsum(c[c_order], dim=0)
        gabs = g.abs().clone()                                # strongest unused, movable channels to add
        move_dir = -torch.sign(g)
        can_move = movable(clean_flat, move_dir)
        gabs[changed_mask] = -1.0
        gabs[~can_move] = -1.0
        n_add = min(int(K.OPTIM_EXCHANGE_ADDS), int((gabs > 0).sum().item()))
        if n_add < 1:
            break
        add_idx = torch.topk(gabs, n_add).indices.tolist()
        slack = -ctx.kappa - m                                # margin head-room of the current flip (>=0)

        def _nmax(j):                                         # cheapest removals the add's slack can pay for
            budget = slack + q * float(g[j].abs().item())
            return int((cum <= budget).sum().item())

        cands = []
        j0 = add_idx[0]                                       # strongest add: a removal-count ladder
        top = _nmax(j0)                                       # (the prediction is sharpest near the boundary,
        if top >= mindrop:                                    #  so back off via a ladder, not one big jump)
            sizes, s = set(), mindrop
            while s < top:
                sizes.add(s)
                s = max(s + 1, int(s * 2))
            sizes.add(top)
            for sz in sorted(sizes):
                d = delta.clone()
                d[rem_order[:sz]] = 0.0
                d[j0] = move_dir[j0] * km
                cands.append(d)
        for j in add_idx[1:]:                                 # other adds: one candidate each (diversity)
            nrem = _nmax(j)
            if nrem < mindrop:
                continue
            d = delta.clone()
            d[rem_order[:nrem]] = 0.0
            d[j] = move_dir[j] * km
            cands.append(d)
        if not cands:
            break
        res = _eval_deltas(ctx, cands)
        best_d, best_nz = None, nchanged
        for d, r in zip(cands, res):
            if _accept(ctx, r) and r["nz"] < best_nz:
                best_nz, best_d = r["nz"], d
        if best_d is None:
            break
        delta = best_d
    return _status(ctx)


def optim_rmse_fmn_l2(ctx: Context) -> str | None:
    """Warm-start FMN-L2 refiner -> snap -> Stage-A prune.

    Seed δ from the Bank flip and let the L2 radius ε "breathe" around the boundary: shrink ε when the
    continuous point is adversarial, grow it when the flip is lost, taking a normalized margin-descent
    step inside the ball each iteration. Every step snaps the support to ±k_min byte prefixes and verifies,
    so the Bank only ever banks exact byte flips. Finishes with optim_rmse_prune on whatever it banked
    (the spec's "reshape continuously, then exact rollback")."""
    dc = _warm_delta_cont(ctx)
    if dc is None:
        return _status(ctx)
    x0 = ctx.clean.view(-1)
    q = ctx.q
    delta = dc.clone()
    eps = float(delta.norm().item())
    gamma = K.OPTIM_FMN_L2_GAMMA
    alpha = K.OPTIM_FMN_L2_ALPHA * q
    for _ in range(K.OPTIM_FMN_L2_STEPS):
        if _out_of_time(ctx):
            break
        m, g = _margin_grad_cont(ctx, delta)
        if m < 0.0:                                           # adversarial -> shrink toward the boundary
            eps = min(eps * (1.0 - gamma), float(delta.norm().item()))
        else:                                                 # lost the flip -> grow
            eps = eps * (1.0 + gamma)
        gnorm = float(g.norm().item())
        if gnorm > 0.0:
            delta = delta - alpha * (g / gnorm)               # normalized margin descent
        nrm = float(delta.norm().item())                      # project onto the L2 ball of radius eps
        if nrm > eps > 0.0:
            delta = delta * (eps / nrm)
        delta = (x0 + delta).clamp(0.0, 1.0) - x0
        _snap_support(ctx, delta, x0, K.OPTIM_FMN_L2_RATIOS)
    return optim_rmse_prune(ctx)


def optim_rmse_fab_l2(ctx: Context) -> str | None:
    """Warm-start FAB-L2 refiner -> snap -> Stage-A prune.

    FAB-style: linearize the decision boundary at the current point and step toward the boundary point
    NEAREST the clean image — a convex blend of the projection of the current point and the projection of
    x0 onto the linearized boundary — with a small overshoot η back into the adversarial region so the
    snapped byte candidate keeps flipping. Snap + verify each step; finish with optim_rmse_prune."""
    dc = _warm_delta_cont(ctx)
    if dc is None:
        return _status(ctx)
    x0 = ctx.clean.view(-1)
    delta = dc.clone()
    eta = K.OPTIM_FAB_L2_ETA
    amax = K.OPTIM_FAB_L2_ALPHA_MAX
    for _ in range(K.OPTIM_FAB_L2_STEPS):
        if _out_of_time(ctx):
            break
        m, g = _margin_grad_cont(ctx, delta)
        gn2 = float((g * g).sum().item())
        if gn2 <= 0.0:
            break
        d_curr = -(m / gn2) * g                               # current point -> its boundary projection
        diff0 = -delta                                        # x0 - x  (current x = x0 + delta)
        l0 = m + float((g * diff0).sum().item())              # linearized margin at x0
        d_clean = diff0 - (l0 / gn2) * g                      # current point -> x0's boundary projection
        nc = float(d_curr.norm().item())
        n0 = float(d_clean.norm().item())
        w = min(amax, nc / (nc + n0 + 1e-12))                 # bias toward the clean-side projection
        delta = delta + (1.0 - w) * d_curr + w * d_clean      # FAB step toward the nearest boundary
        delta = delta - eta * (abs(m) / gn2) * g              # overshoot back into the adversarial region
        delta = (x0 + delta).clamp(0.0, 1.0) - x0
        _snap_support(ctx, delta, x0, K.OPTIM_FAB_L2_RATIOS)
    return optim_rmse_prune(ctx)


def optim_rmse_sigma_zero(ctx: Context) -> str | None:
    """σ-zero refiner: descend a differentiable L0 surrogate to find a minimal SUPPORT.

    ||δ||_0 ≈ Σ_i δ_i²/(δ_i²+σ) is smooth for σ>0, so plain gradient descent can push channels toward off.
    Loss L = max(margin, 0) + (1/n)·surrogate: while not flipped the margin term dominates (cross the
    boundary); once flipped only the surrogate remains (thin the support). The gradient is ∞-normalized;
    weak channels are hard-zeroed below a relative threshold τ that RISES when adversarial (chase sparsity)
    and FALLS when not (let channels back in). Each step snaps the surviving support to ±k_min bytes and
    verifies. Warm-starts from the Bank flip; pair with optim_rmse_prune to polish."""
    dc = _warm_delta_cont(ctx)
    if dc is None:
        return _status(ctx)
    x0 = ctx.clean.view(-1)
    n = int(x0.numel())
    q = ctx.q
    sigma = K.OPTIM_SIGMA_SIGMA
    tau = K.OPTIM_SIGMA_TAU0
    t = K.OPTIM_SIGMA_T
    delta = dc.clone()
    nsteps = max(1, K.OPTIM_SIGMA_STEPS)
    for i in range(nsteps):
        if _out_of_time(ctx):
            break
        m, g = _margin_grad_cont(ctx, delta)
        gL = g.clone() if m > 0.0 else torch.zeros_like(g)    # margin term only while not yet flipped
        d2 = delta * delta
        gL = gL + ((2.0 * delta * sigma) / ((d2 + sigma) ** 2)) / float(n)  # + surrogate term
        ginf = float(gL.abs().max().item())
        eta = K.OPTIM_SIGMA_ETA0 * q * 0.5 * (1.0 + math.cos(math.pi * i / nsteps))  # cosine anneal
        if ginf > 0.0:
            delta = delta - eta * (gL / ginf)
        delta = (x0 + delta).clamp(0.0, 1.0) - x0
        dmax = float(delta.abs().max().item())                # hard-zero below the relative threshold τ
        if dmax > 0.0:
            delta = torch.where(delta.abs() < tau * dmax, torch.zeros_like(delta), delta)
        step_frac = eta / max(q, 1e-12)
        tau = tau + t * step_frac if m < 0.0 else max(0.0, tau - t * step_frac)
        _snap_support(ctx, delta, x0, K.OPTIM_SIGMA_RATIOS)
    return _status(ctx)


def optim_rmse_fmn_l0(ctx: Context) -> str | None:
    """Warm-start FMN-L0 refiner: the budget ε IS the integer support count, shrunk toward the boundary.

    Seed δ and ε from the Bank flip (ε = current |S|). Each step take a normalized margin-descent step,
    then project to L0 by KEEPING the ε channels of largest |δ| and zeroing the rest; shrink ε when the
    kept point is adversarial (chase fewer channels), grow it back when the flip is lost. Snap the kept
    support to ±k_min bytes and verify each step. In the byte regime ε is exactly |S|, so this drives RMSE
    down directly. Warm-starts from the Bank flip; pair with optim_rmse_prune to polish."""
    dc = _warm_delta_cont(ctx)
    if dc is None:
        return _status(ctx)
    x0 = ctx.clean.view(-1)
    n = int(x0.numel())
    q = ctx.q
    alpha = K.OPTIM_FMN_L0_ALPHA * q
    gamma = K.OPTIM_FMN_L0_GAMMA
    delta = dc.clone()
    eps = max(1, int((delta != 0).sum().item()))
    best_eps = eps
    for _ in range(K.OPTIM_FMN_L0_STEPS):
        if _out_of_time(ctx):
            break
        m, g = _margin_grad_cont(ctx, delta)
        if m < 0.0:                                           # adversarial -> shrink the support budget
            best_eps = min(best_eps, int((delta != 0).sum().item()))
            eps = max(1, min(int(math.floor(eps * (1.0 - gamma))), best_eps))
        else:                                                 # lost the flip -> grow it back
            eps = min(n, max(eps + 1, int(math.ceil(eps * (1.0 + gamma)))))
        gnorm = float(g.norm().item())
        if gnorm > 0.0:
            delta = delta - alpha * (g / gnorm)               # normalized margin descent
        if eps < n:                                           # L0 projection: keep the ε largest-|δ| channels
            keep_idx = torch.topk(delta.abs(), eps).indices
            keep = torch.zeros_like(delta, dtype=torch.bool)
            keep[keep_idx] = True
            delta = delta * keep
        delta = (x0 + delta).clamp(0.0, 1.0) - x0
        _snap_support(ctx, delta, x0, K.OPTIM_FMN_L0_RATIOS)
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
    """Find a sparse ±1/255 flip with the batched flip-first pipeline. Returns the sparsest
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
    hard_deadline = t_start + max(0.05, float(timeout_seconds) - float(reserve_seconds))
    deadline = min(hard_deadline, t_start + K.FIND_FLIP_BUDGET)

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

    # --- ORCHESTRATOR SWITCH: change the function name on the next line to swap strategy. ---
    # find_flip_first_safe(ctx)
    # find_flip_first_hit(ctx)
    # find_flip_run_all(ctx)
    # find_population_pgd(ctx)         # Batched Multi-Target Quantized PGD (population)
    # find_beam_byte_pgd(ctx)        # Multi-target Beam Byte-PGD
    # find_ensemble_byte_pgd(ctx)    # beam + ensemble dirs (DLR / sign-mix / opposite) + structured mutation
    # find_hydra(ctx)                # Q1-Hydra: primitive-space (prefix / tiles / low-freq / color) + beam
    # find_apgd_dlr(ctx)             # Quantized APGD-DLR with batched restarts (ternary cube)
    find_apgd_targeted(ctx)        # targeted APGD-L∞ (DLR-T) + momentum + restarts (live finder)
    # find_fmn(ctx)                  # Q1-FMN-L1 support finder (continuous L1 breathing -> byte snap)
    # find_frank_wolfe(ctx)          # Q1-Block Frank-Wolfe + Sparse-RS rescue
    # find_alma(ctx)                 # Q1-ALMA-lite (augmented Lagrangian -> byte snap)

    deadline = hard_deadline

    # --- RMSE REFINEMENT: runs on the Bank's best flip; uncomment to layer after the finder. ---
    # Each shrinks |S| and folds sparser survivors back into the Bank. Compose by uncommenting several
    # (they run in order, each warm-starting from the Bank's current best). fmn_l2/fab_l2 auto-prune.
    optim_rmse_prune(ctx)          # Stage A: gradient-ranked byte rollback + adaptive batch (recommended)
    optim_rmse_exchange(ctx)       # Stage B: 1-for-many coordinate exchange (run after prune stalls)
    # optim_rmse_fmn_l2(ctx)         # warm-start FMN-L2 breathing -> snap -> prune
    # optim_rmse_fab_l2(ctx)         # warm-start FAB-L2 boundary projection -> snap -> prune
    # optim_rmse_sigma_zero(ctx)     # σ-zero differentiable-L0 support finder -> snap
    # optim_rmse_fmn_l0(ctx)         # warm-start FMN-L0 (integer support budget) -> snap

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
