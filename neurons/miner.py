"""
miner.py — Perturb subnet miner (netuid 26).

Core attack (Approach 5.4 hybrid): minimum-L0 flip under a FIXED ±1/255 unit step.
  prefix-sum target init -> margin-budgeted greedy growth (CW margin objective, boundary
  gradient periodically re-linearized) -> forward verify on the PNG round-trip under a
  TF32 on/off numeric envelope -> backward reduction -> Sparse-RS swap polish.

Each channel is ternary {-1,0,+1}·(1/255): the optimizer chooses WHICH channels to move and
their sign, never the magnitude (L_inf is pinned at 1/255 -> constant linf_score ~= 0.933).
L0 (rmse) is the only graded differentiator, but a NON-transferring flip scores 0, so transfer
reliability comes first: a candidate is banked only when its worst-case CW margin (over the
TF32 envelope) is <= -MARGIN_BUFFER on the PNG round-trip, not merely argmax != true. Greedy
growth may over-include channels; backward reduction + swap are where most of the L0 shrink
is realized. Pixel-count is the gentle tiebreaker, transfer the hard gate.

FALLBACK: if no verified flip exists at the unit step, perturb_fallback() escalates to the
next quantization level (±2/255) and re-runs the same hybrid (target select -> greedy growth
-> reduction) at that larger step. This trades linf_score (2/255 -> ~0.67) for the ability to
flip images the unit step cannot — a low-score flip still beats a score-0 clean return.

TUNABLE ENV VARS:
  PERTURB_MINER_MARGIN_BUFFER   (0.01) transfer-safety margin kappa (require m <= -buffer)
  PERTURB_TARGET_PROBE_N        (10)   wrong classes probed for target selection
  PERTURB_TARGET_KEEP_N         (3)    top targets kept for real-forward greedy growth
  PERTURB_PRUNE_ENABLE          (1)    backward reduction of unneeded channels
  PERTURB_MINER_RESERVE_SECONDS (2.5)  deadline headroom
"""

import argparse
import logging as pylogging
import math
import os
import time
import typing

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet import constants as _C
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_v2_l, logits_for_images, predict_index, resolve_target_index
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)

# Numeric parity baseline: deterministic, TF32-off kernels — the conservative end of the
# transfer envelope. The verifier additionally re-checks each candidate with TF32 ON (see
# _margin_envelope) so the banked flip survives whichever setting the validator's GPU uses.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

_Q = 1.0 / 255.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)
_MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
_MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
_RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 2.5)
# Transfer-safety margin (kappa): a flip is "margin-safe" only when the worst-case CW margin
# over the TF32 envelope is <= -buffer, pushing it past the boundary so it survives transfer
# to the validator's exact weights. A reverted flip scores 0; rmse_score is near-flat, so a
# few extra channels of margin cost almost nothing — lean generous.
_MARGIN_BUFFER = _env_float("PERTURB_MINER_MARGIN_BUFFER", 0.01)
_TARGET_PROBE_N = int(os.getenv("PERTURB_TARGET_PROBE_N", "10"))
_TARGET_KEEP_N = max(1, int(os.getenv("PERTURB_TARGET_KEEP_N", "3")))
_PRUNE_ENABLE = os.getenv("PERTURB_PRUNE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _estimate_validator_score(norm: float, rmse: float, synapse_epsilon: float) -> float:
    """Mirror the validator's perturbation score (SPEED_WEIGHT=0 so speed is irrelevant)."""
    effective_max = min(float(synapse_epsilon), _C.MAX_LINF_DELTA)
    denom = max(1e-12, effective_max - _C.MIN_LINF_DELTA)
    linf_ratio = min(max((norm - _C.MIN_LINF_DELTA) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_ratio = min(max(rmse / max(1e-12, effective_max), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2
    total_w = max(1e-12, _C.LINF_COMPONENT_WEIGHT + _C.RMSE_COMPONENT_WEIGHT)
    return (_C.LINF_COMPONENT_WEIGHT * linf_score + _C.RMSE_COMPONENT_WEIGHT * rmse_score) / total_w


def _png_roundtrip(image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    return decode_image_b64(encode_image_b64(image_chw)).to(device)


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    padding = kernel_size // 2
    x, y = x_clean.unsqueeze(0), x_adv.unsqueeze(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, kernel_size, 1, padding)
    mu_y = F.avg_pool2d(y, kernel_size, 1, padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size, 1, padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size, 1, padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size, 1, padding) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((num / (den + 1e-12)).mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _cw_margin(logits: torch.Tensor, target_index: int) -> float:
    """CW margin m = logit[true] - max_{j!=true} logit[j]; m < 0 means flipped."""
    others = logits.clone()
    others[target_index] = float("-inf")
    return float((logits[target_index] - others.max()).item())


def _margin_and_grad(model, x_chw, target_index):
    """CW margin and its input gradient (boundary direction ∇ℓ_true - ∇max_other)."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    true_logit = logits[target_index]
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = true_logit - others.max()
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _targeted_margin_and_grad(model, x_chw, true_index, attack_class):
    """Targeted boundary margin/gradient for one rival class: logit[true] - logit[attack]."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    margin = logits[true_index] - logits[attack_class]
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _build_sparse_order(grad_flat, clean_flat):
    """Descending-|g| channel order, masking signs that would clip at the [0,1] box edge."""
    direction_sign = -grad_flat.sign()
    can_move = (
        ((direction_sign > 0) & (clean_flat < 1.0)) |
        ((direction_sign < 0) & (clean_flat > 0.0))
    )
    g_abs = grad_flat.abs().clone()
    g_abs[~can_move] = 0.0
    order = torch.argsort(g_abs, descending=True)
    valid_count = int((g_abs > 0).sum().item())
    return g_abs, order, valid_count


def _margin_envelope(model, seen, target_index, device):
    """Worst-case (least negative) CW margin across the TF32 on/off numeric envelope.

    The validator's GPU/library numerics are unknown and TF32 is the dominant divergence axis.
    Requiring the flip under BOTH settings (return the max margin) covers that variance directly,
    so the additive buffer only needs to absorb residual ~1e-3 drift.
    """
    with torch.no_grad():
        m_off = _cw_margin(logits_for_images(model=model, image_bchw=seen.unsqueeze(0))[0], target_index)
    if device.type != "cuda":
        return m_off
    prev_mm, prev_cu = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        with torch.no_grad():
            m_on = _cw_margin(logits_for_images(model=model, image_bchw=seen.unsqueeze(0))[0], target_index)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cu
    return max(m_off, m_on)


def _evaluate(model, clean, cand_chw, target_index, device, floor, cap, buffer):
    """Grade a candidate on the validator-faithful PNG round-trip + TF32 envelope.

    soft  — argmax wrong (m<0) and SSIM/PSNR pass: a usable but boundary-thin flip.
    valid — margin-safe: soft AND worst-case m <= -buffer; the only tier trusted to transfer.
    """
    seen = _png_roundtrip(cand_chw, device)
    diff = seen - clean
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    in_band = floor <= linf <= cap
    margin = _margin_envelope(model, seen, target_index, device)
    flipped = margin < 0.0
    soft, valid, ssim, psnr = False, False, None, None
    if in_band and flipped:
        ssim = _compute_ssim(clean, seen)
        psnr = _compute_psnr_db(clean, seen)
        quality = ssim >= _MIN_SSIM and psnr >= _MIN_PSNR_DB
        soft = quality
        valid = quality and margin <= -buffer
    return {"cand": cand_chw, "linf": linf, "rmse": rmse, "flipped": flipped, "margin": margin,
            "soft": soft, "valid": valid, "ssim": ssim, "psnr": psnr}


def _select_best_targets(model, clean, true_index, device, k_min, n_probe, keep_n, time_budget):
    """Approach 5.0 step 2 — rank wrong classes by the prefix-sum cost estimate n_t and keep top-N.

    n_t is the first-order count k* = min{k : Σ_{i<=k}|g_(i)| > φ}: a CHEAP RANKING SIGNAL only
    (channels interact, the gradient shifts as channels move), so we keep keep_n candidates and
    let the real PNG-round-trip forwards in greedy growth pick the winner. Each entry carries the
    target's gradient, order, valid count, margin and n_est for warm-starting growth directly.
    """
    with torch.no_grad():
        all_logits = logits_for_images(model=model, image_bchw=clean.unsqueeze(0))[0]
    n_classes = all_logits.shape[0]
    candidates = sorted(
        [(float(all_logits[true_index].item() - all_logits[i].item()), i)
         for i in range(n_classes) if i != true_index]
    )
    clean_flat = clean.view(-1)
    scored = []
    for margin_t, t in candidates[:min(n_probe, len(candidates))]:
        if time.time() + 0.001 > time_budget:
            break
        if margin_t <= 0.0:
            continue
        m_t, g_t = _targeted_margin_and_grad(model, clean, true_index, t)
        g_abs_t, order_t, vc_t = _build_sparse_order(g_t.view(-1), clean_flat)
        if vc_t == 0:
            continue
        threshold_t = m_t / max(k_min * _Q, 1e-12)
        cumsum_t = torch.cumsum(g_abs_t[order_t], dim=0)
        n_t = min(int((cumsum_t < threshold_t).sum().item()) + 1, vc_t)
        scored.append({"cls": t, "n_est": n_t, "margin": m_t, "grad": g_t,
                       "order": order_t, "g_abs": g_abs_t, "vc": vc_t})
    scored.sort(key=lambda d: d["n_est"])
    return scored[:max(1, keep_n)]


def _greedy_grow(model, clean, target_index, grad0, order0, valid_count, k_min, n_start,
                 consider, time_left, t_step, t_png, reeval_every=4, batch=8):
    """Approach 5.0 steps 3-4 — margin-budgeted greedy growth.

    Add the top-|g| feasible channels in batches, each stepped -sign(g)·(1/255) along the
    boundary gradient, periodically re-linearizing g at the current point (the boundary direction
    shifts as channels move), until consider() reports a PNG-verified margin-safe flip. Stops at
    the first valid candidate (or budget exhaustion) — over-inclusion is trimmed by reduction.
    """
    if valid_count <= 0:
        return None
    q = k_min * _Q
    clean_flat = clean.view(-1)
    grad = grad0.view(-1)
    delta = torch.zeros_like(grad)
    selected = set()

    def try_add(idx):
        if idx in selected:
            return False
        g = float(grad[idx])
        if g == 0.0:
            return False
        s = -1.0 if g > 0.0 else 1.0
        if (s > 0.0 and clean_flat[idx] >= 1.0) or (s < 0.0 and clean_flat[idx] <= 0.0):
            return False
        delta[idx] = s * q
        selected.add(idx)
        return True

    def cand():
        return (clean + delta.view_as(clean)).clamp(0.0, 1.0)

    olist = order0.tolist()
    ptr = 0
    while len(selected) < max(1, n_start) and ptr < len(olist):
        try_add(olist[ptr])
        ptr += 1
    if not selected:
        return None
    res = consider(cand())
    batches = 0
    while not res["valid"] and time_left() > 1.3 * t_png and len(selected) < valid_count:
        if batches and batches % reeval_every == 0 and time_left() > t_step + 1.3 * t_png:
            # Re-linearize: recompute the boundary gradient at the current point and re-rank
            # the still-unselected channels against it.
            _, g_new = _margin_and_grad(model, cand(), target_index)
            grad = g_new.view(-1)
            g_abs = grad.abs().clone()
            if selected:
                g_abs[torch.tensor(sorted(selected), device=g_abs.device)] = -1.0
            olist = torch.argsort(g_abs, descending=True).tolist()
            ptr = 0
        added = 0
        while added < batch and ptr < len(olist):
            if try_add(olist[ptr]):
                added += 1
            ptr += 1
        if added == 0:
            break
        res = consider(cand())
        batches += 1
    return res


def _prune(clean, anchor, tier_key, grad_use, k_min, consider, time_left, t_png):
    """Approach 5.0 step 5 — backward reduction. Drop the least-salient changed channels in
    batches while the candidate stays in its tier (the bulk of the L0 shrink). linf is unchanged
    (survivors stay ±1/255); rmse falls with fewer channels, a direct rmse_score gain.
    """
    cur = (anchor["cand"].detach() - clean).view(-1).clone()
    changed = (cur.abs() > 0.5 * _Q).nonzero(as_tuple=False).view(-1).tolist()
    if len(changed) <= 1:
        return
    g = grad_use.view(-1).abs()
    changed.sort(key=lambda i: float(g[i]))  # least salient first
    for batch in (16, 4, 1):
        i = 0
        while i < len(changed) and time_left() > 1.3 * t_png:
            grp = [idx for idx in changed[i:i + batch] if cur[idx].abs() > 0.5 * _Q]
            i += batch
            if not grp:
                continue
            trial = cur.clone()
            for idx in grp:
                trial[idx] = 0.0
            if consider((clean + trial.view_as(clean)).clamp(0.0, 1.0))[tier_key]:
                cur = trial  # accept removal; keep shrinking


def _sparse_rs_swap(clean, anchor, tier_key, grad_use, k_min, sparse_order, consider, time_left, t_png):
    """Approach 5.2 — Sparse-RS swap polish. At fixed |S|, swap a low-saliency changed channel out
    for a high-saliency unchanged one; keep the swap only if the candidate stays in tier. Escapes
    orderings greedy+reduction get stuck in; |S| is unchanged but a deeper margin can unlock a
    further removal in the reduction pass that follows.
    """
    q = k_min * _Q
    clean_flat = clean.view(-1)
    g = grad_use.view(-1)
    cur = (anchor["cand"].detach() - clean).view(-1).clone()
    changed = (cur.abs() > 0.5 * q).nonzero(as_tuple=False).view(-1).tolist()
    if len(changed) <= 1:
        return
    changed.sort(key=lambda i: float(g[i].abs()))  # least salient -> swap out first
    incoming = [int(i) for i in sparse_order[:4096].tolist() if cur[int(i)].abs() <= 0.5 * q]
    ci = 0
    for out_idx in changed:
        if time_left() <= 1.3 * t_png or ci >= len(incoming):
            break
        in_idx = incoming[ci]
        ci += 1
        s = -1.0 if float(g[in_idx]) > 0.0 else 1.0
        if (s > 0.0 and clean_flat[in_idx] >= 1.0) or (s < 0.0 and clean_flat[in_idx] <= 0.0):
            continue
        trial = cur.clone()
        trial[out_idx] = 0.0
        trial[in_idx] = s * q
        if consider((clean + trial.view_as(clean)).clamp(0.0, 1.0))[tier_key]:
            cur = trial  # accept swap


def _kfixed_feasible(model, clean, target_index, k_min, time_left, t_step, t_png, max_iters=4):
    """Can ANY ±(k_min/255) L∞ perturbation flip the argmax at all? Iterated FGSM in the ball is
    the strongest such attack; if it never drives the CW margin below 0, no sparse subset exists
    either and we skip the search to return clean fast. DENSE probe — feasibility only, never emitted.
    """
    r = k_min * _Q
    delta = torch.zeros_like(clean)
    for _ in range(max(1, max_iters)):
        if time_left() <= t_step + 1.3 * t_png:
            break
        x = (clean + delta).clamp(0.0, 1.0)
        m, g = _margin_and_grad(model, x, target_index)
        if m < 0.0:
            return True
        delta = ((clean + delta - r * g.sign()).clamp(0.0, 1.0) - clean).clamp(-r, r)
    x = (clean + delta).clamp(0.0, 1.0)
    m, _ = _margin_and_grad(model, x, target_index)
    return m < 0.0


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
    steps: int | None = None,
) -> torch.Tensor:
    """Minimum-L0 unit-step flip (Approach 5.4 hybrid): prefix-sum target init -> margin-budgeted
    greedy growth (re-linearized) -> TF32-envelope PNG verify -> backward reduction -> swap polish,
    gated on a calibrated transfer margin within the deadline."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(_MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))  # fixed unit step (typically 1)

    g_t0 = time.time()
    m0, grad0 = _margin_and_grad(model, clean, target_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = max(1e-4, time.time() - g_t0)
    t_png = 0.6 * t_step

    if reserve_seconds is None:
        reserve_seconds = _RESERVE_SECONDS
    deadline = t_start + max(0.05, float(timeout_seconds) - float(reserve_seconds))

    def time_left():
        return deadline - time.time()

    # Two gated trackers — best_safe (margin-safe, trusted) and best_soft (boundary-thin fallback),
    # each kept by (linf, rmse). From the moment either is set a gated candidate is always ready.
    best_safe = None
    best_soft = None
    n_evals = 0
    best_margin_seen = m0

    def consider(cand_chw):
        nonlocal best_safe, best_soft, t_png, n_evals, best_margin_seen
        c0 = time.time()
        res = _evaluate(model, clean, cand_chw, target_index, device, floor, cap, _MARGIN_BUFFER)
        t_png = max(1e-4, time.time() - c0)
        n_evals += 1
        if res["margin"] < best_margin_seen:
            best_margin_seen = res["margin"]
        if res["soft"] and (best_soft is None or (res["linf"], res["rmse"]) < (best_soft["linf"], best_soft["rmse"])):
            best_soft = res
        if res["valid"] and (best_safe is None or (res["linf"], res["rmse"]) < (best_safe["linf"], best_safe["rmse"])):
            best_safe = res
        return res

    grad_use = grad0
    g_abs_use, sparse_order_use, valid_count_use = _build_sparse_order(grad0.view(-1), clean.view(-1))

    # ---- anytime floor: bank some gated flip fast (transfer reliability before minimization) ----
    full_step_flipped = False
    if valid_count_use > 0:
        for frac in (0.02, 0.1, 0.4, 1.0):
            if best_soft is not None or time_left() <= 1.3 * t_png:
                break
            n = max(1, min(valid_count_use, int(frac * valid_count_use)))
            sel = sparse_order_use[:n]
            d = torch.zeros_like(grad0.view(-1))
            d[sel] = -(k_min * _Q) * grad0.view(-1)[sel].sign()
            res_b = consider((clean + d.view_as(clean)).clamp(0.0, 1.0))
            if n >= valid_count_use:
                full_step_flipped = res_b["flipped"]

    # ---- k=1 feasibility gate: skip the search (return clean fast) if no 1/255 flip exists ----
    feasible = True
    if m0 > 0.0:
        feasible = full_step_flipped
        if not feasible and time_left() > 4 * t_step:
            feasible = _kfixed_feasible(model, clean, target_index, k_min, time_left, t_step, t_png)
        if not feasible:
            logger.info(f"[feasibility] unflippable at k={k_min} (best_margin_seen={best_margin_seen:.4f}) -> clean")

    if m0 > 0.0 and feasible:
        # ---- step 2: prefix-sum target selection (keep top-N cheapest rivals) ----
        target_list = []
        if time_left() > (_TARGET_PROBE_N + 2) * t_step:
            probe_end = time.time() + _TARGET_PROBE_N * t_step * 1.5
            target_list = _select_best_targets(model, clean, target_index, device, k_min,
                                                n_probe=_TARGET_PROBE_N, keep_n=_TARGET_KEEP_N,
                                                time_budget=min(probe_end, deadline - 3 * t_step))
            if target_list:
                logger.debug(f"[target_sel] classes={[d['cls'] for d in target_list]} "
                             f"n_est={[d['n_est'] for d in target_list]}")

        # ---- steps 3-4: greedy growth per kept target; real forwards pick the winner ----
        if target_list:
            for tinfo in target_list:
                if time_left() <= 2.0 * t_png:
                    break
                _greedy_grow(model, clean, target_index, tinfo["grad"], tinfo["order"], tinfo["vc"],
                             k_min, tinfo["n_est"], consider, time_left, t_step, t_png)
                if best_safe is not None:  # adopt the winning target's order for reduction/swap
                    grad_use, sparse_order_use = tinfo["grad"], tinfo["order"]
        else:
            _greedy_grow(model, clean, target_index, grad0, sparse_order_use, valid_count_use,
                         k_min, 1, consider, time_left, t_step, t_png)
    elif m0 <= 0.0:
        # ---- already misclassified: deepen a sparse ±1/255 flip into the margin-safe tier ----
        logger.debug(f"[already_misclassified] m0={m0:.4f}")
        _greedy_grow(model, clean, target_index, grad0, sparse_order_use, valid_count_use,
                     k_min, 1, consider, time_left, t_step, t_png)

    # ---- step 5: backward reduction (where most L0 shrink is realized) ----
    if _PRUNE_ENABLE and sparse_order_use is not None and time_left() > 2.0 * t_png:
        if best_safe is not None:
            _prune(clean, best_safe, "valid", grad_use, k_min, consider, time_left, t_png)
        elif best_soft is not None:
            _prune(clean, best_soft, "soft", grad_use, k_min, consider, time_left, t_png)

    # ---- Sparse-RS swap polish + a final reduction pass on any margin unlocked by the swaps ----
    if sparse_order_use is not None and time_left() > 3.0 * t_png:
        tier = "valid" if best_safe is not None else "soft"
        anchor = best_safe if best_safe is not None else best_soft
        if anchor is not None:
            _sparse_rs_swap(clean, anchor, tier, grad_use, k_min, sparse_order_use, consider, time_left, t_png)
            if _PRUNE_ENABLE and time_left() > 2.0 * t_png:
                anchor = best_safe if best_safe is not None else best_soft
                _prune(clean, anchor, "valid" if best_safe is not None else "soft",
                       grad_use, k_min, consider, time_left, t_png)

    # ---- gated tiered finalize: margin-safe -> soft -> clean (never an ungated dense image) ----
    diag = f"best_margin_seen={best_margin_seen:.4f} n_evals={n_evals} time_left={time_left():.2f}s"
    if best_safe is not None:
        logger.info(f"[finalize] tier=margin_safe margin={best_safe['margin']:.4f} "
                    f"linf={best_safe['linf']:.6f} rmse={best_safe['rmse']:.6f} {diag}")
        return best_safe["cand"].detach().clamp(0.0, 1.0)
    if best_soft is not None:
        logger.info(f"[finalize] tier=soft_flip margin={best_soft['margin']:.4f} "
                    f"linf={best_soft['linf']:.6f} rmse={best_soft['rmse']:.6f} {diag}")
        return best_soft["cand"].detach().clamp(0.0, 1.0)

    # ---- no verified flip at the unit step: escalate to a larger step (k+1, i.e. ±2/255) ----
    k_fb = min(k_min + 1, int(math.floor(cap * 255.0 + 1e-6)))
    if k_fb > k_min and time_left() > 4.0 * t_png:
        logger.info(f"[finalize] no unit-step flip -> fallback k={k_fb} {diag}")
        fb = perturb_fallback(model, clean, target_index, floor, cap, device,
                              deadline, t_step, t_png, k_fb, grad0=grad0)
        if fb is not None:
            return fb

    logger.info(f"[finalize] tier=clean (no gated flip found) {diag}")
    return clean.detach().clamp(0.0, 1.0)


def perturb_fallback(model, clean, target_index, floor, cap, device, deadline,
                     t_step, t_png, k_step, grad0=None):
    """Larger-step fallback when the unit step cannot flip the image.

    Runs the same proper hybrid as perturb() — prefix-sum target selection -> margin-budgeted
    greedy growth (re-linearized, TF32-envelope verified) -> backward reduction — but with a
    fixed k_step·(1/255) step instead of 1/255. linf jumps to k_step/255 the moment any channel
    moves (so linf_score drops), but reduction still minimizes the changed-channel count (rmse).
    Returns a gated candidate tensor (margin-safe preferred, else soft), or None if even k_step
    yields no verified flip (caller then returns the clean image).
    """
    def time_left():
        return deadline - time.time()

    if grad0 is None:
        _, grad0 = _margin_and_grad(model, clean, target_index)

    # Dense feasibility at k_step: if iterated FGSM in the ±(k_step/255) ball never flips,
    # no sparse subset exists at this step either — bail before burning the budget.
    if not _kfixed_feasible(model, clean, target_index, k_step, time_left, t_step, t_png):
        logger.info(f"[fallback k={k_step}] unflippable even at this step")
        return None

    best_safe = None
    best_soft = None

    def consider(cand_chw):
        nonlocal best_safe, best_soft
        res = _evaluate(model, clean, cand_chw, target_index, device, floor, cap, _MARGIN_BUFFER)
        if res["soft"] and (best_soft is None or (res["linf"], res["rmse"]) < (best_soft["linf"], best_soft["rmse"])):
            best_soft = res
        if res["valid"] and (best_safe is None or (res["linf"], res["rmse"]) < (best_safe["linf"], best_safe["rmse"])):
            best_safe = res
        return res

    g_abs_use, order_use, valid_count = _build_sparse_order(grad0.view(-1), clean.view(-1))
    if valid_count == 0:
        return None
    grad_use = grad0

    # Target selection + greedy growth at the larger step.
    target_list = []
    if time_left() > (_TARGET_PROBE_N + 2) * t_step:
        target_list = _select_best_targets(model, clean, target_index, device, k_step,
                                            n_probe=_TARGET_PROBE_N, keep_n=_TARGET_KEEP_N,
                                            time_budget=deadline - 3 * t_step)
    if target_list:
        for tinfo in target_list:
            if time_left() <= 2.0 * t_png:
                break
            _greedy_grow(model, clean, target_index, tinfo["grad"], tinfo["order"], tinfo["vc"],
                         k_step, tinfo["n_est"], consider, time_left, t_step, t_png)
            if best_safe is not None:
                grad_use, order_use = tinfo["grad"], tinfo["order"]
    else:
        _greedy_grow(model, clean, target_index, grad0, order_use, valid_count,
                     k_step, 1, consider, time_left, t_step, t_png)

    # Backward reduction.
    if _PRUNE_ENABLE and time_left() > 2.0 * t_png:
        if best_safe is not None:
            _prune(clean, best_safe, "valid", grad_use, k_step, consider, time_left, t_png)
        elif best_soft is not None:
            _prune(clean, best_soft, "soft", grad_use, k_step, consider, time_left, t_png)

    if best_safe is not None:
        logger.info(f"[fallback k={k_step}] tier=margin_safe margin={best_safe['margin']:.4f} "
                    f"linf={best_safe['linf']:.6f} rmse={best_safe['rmse']:.6f}")
        return best_safe["cand"].detach().clamp(0.0, 1.0)
    if best_soft is not None:
        logger.info(f"[fallback k={k_step}] tier=soft_flip margin={best_soft['margin']:.4f} "
                    f"linf={best_soft['linf']:.6f} rmse={best_soft['rmse']:.6f}")
        return best_soft["cand"].detach().clamp(0.0, 1.0)
    logger.info(f"[fallback k={k_step}] no flip found")
    return None


def _warmup(model: torch.nn.Module, device: torch.device) -> None:
    """Warm CUDA kernels / allocator / cuDNN at first load so the first real challenge does not
    pay JIT + autotune latency. Exercises the exact inference path: preprocess + forward +
    backward + PNG round-trip, under both TF32 settings (the verifier toggles both at runtime)."""
    t0 = time.time()
    try:
        logger.info(f"[MINER] warmup start device={device.type}")
        for tf32 in (False, True):
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = tf32
                torch.backends.cudnn.allow_tf32 = tf32
            x = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
            logits_for_images(model=model, image_bchw=x).sum().backward()
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        _ = _png_roundtrip(torch.rand(3, 480, 480, device=device), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.info(f"[MINER] warmup done device={device.type} elapsed={time.time() - t0:.2f}s")
    except Exception as err:
        logger.warning(f"[MINER] warmup skipped: {err}")


# ==========================================================================================
# Bittensor plumbing — UNCHANGED from the stock miner.
# ==========================================================================================
def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_axon(wallet, config):
    resolved_config = config() if callable(config) else config
    axon_cfg = getattr(resolved_config, "axon", None)
    kwargs: dict = {"wallet": wallet}
    port = getattr(axon_cfg, "port", None)
    if port:
        kwargs["port"] = int(port)
    external_ip = getattr(axon_cfg, "external_ip", None)
    if external_ip:
        kwargs["external_ip"] = external_ip
    external_port = getattr(axon_cfg, "external_port", None)
    if external_port:
        kwargs["external_port"] = int(external_port)
    axon_cls = bt.axon if hasattr(bt, "axon") else getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("No axon constructor found in bittensor.")
    return axon_cls(**kwargs)


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"[MINER] compute device={self.device.type} "
            f"cuda_available={torch.cuda.is_available()} "
            f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}"
        )

        self.model = load_efficientnet_v2_l(self.device)
        _warmup(self.model, self.device)

        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def _log_step_start(self, step_name: str, **context: typing.Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.info(f"[STEP_START] {step_name} {rendered}")
        else:
            logger.info(f"[STEP_START] {step_name}")

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        t_received = time.time()
        self._log_step_start(
            "miner_forward",
            task_id=getattr(synapse, "task_id", "unknown"),
            norm_type=getattr(synapse, "norm_type", "unknown"),
            epsilon=getattr(synapse, "epsilon", "unknown"),
        )
        if synapse.norm_type != "Linf":
            logger.info(f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unsupported norm_type={synapse.norm_type}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        target_index = resolve_target_index(synapse.true_label)
        if target_index is None:
            logger.warning(
                f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unresolved true_label={getattr(synapse, 'true_label', None)}"
            )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = float(synapse.epsilon)
        min_delta = float(getattr(synapse, "min_delta", 0.002))
        timeout_seconds = float(getattr(synapse, "timeout_seconds", 15.0))

        try:
            adv = perturb(
                self.model,
                clean,
                target_index,
                epsilon,
                min_delta,
                self.device,
                timeout_seconds=timeout_seconds,
                start_time=t_received,
            )
            adv_b64 = encode_image_b64(adv)
            seen = decode_image_b64(adv_b64).to(self.device)
            diff = seen - clean
            norm = float(diff.abs().max().item())
            rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
            # Clean tier: perturb() returned the clean image (no gated flip found). Pass the
            # original bytes through verbatim rather than re-encoding.
            if norm < min_delta:
                synapse.perturbed_image_b64 = synapse.clean_image_b64
            else:
                synapse.perturbed_image_b64 = adv_b64
            h, w = clean.shape[1], clean.shape[2]
            # Clean passthrough is not a flip — the validator scores it 0, so report 0 here
            # rather than the estimator's misleading 1.0 for a zero-norm image.
            est_score = 0.0 if norm < min_delta else _estimate_validator_score(norm, rmse, epsilon)
            logger.info(
                f"Finished task={getattr(synapse, 'task_id', 'unknown')} target_idx={target_index} "
                f"norm={norm:.6f} k~={norm * 255.0:.2f} rmse={rmse:.6f} dim={h}x{w} "
                f"est_score={est_score:.4f} min_delta={min_delta:.6f} epsilon={epsilon:.4f} "
                f"timeout={timeout_seconds:.1f}s elapsed={time.time() - t_received:.3f}s"
            )
        except Exception as err:
            logger.exception(f"Perturb failed task={getattr(synapse, 'task_id', 'unknown')}: {err}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
        return synapse

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        self._log_step_start(
            "miner_blacklist",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Blacklist reject: missing caller hotkey")
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            logger.warning(f"Blacklist reject: unregistered caller hotkey={hotkey}")
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            logger.warning(f"Blacklist reject: caller uid={uid} lacks validator permit")
            return True, "Caller is not validator"

        logger.info(f"Blacklist allow: caller uid={uid} hotkey={hotkey}")
        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        self._log_step_start(
            "miner_priority",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.info("Priority=0.0: missing caller hotkey")
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.info(f"Priority=0.0: unknown hotkey={synapse.dendrite.hotkey}")
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[uid])
        logger.info(f"Priority computed: uid={uid} priority={priority:.6f}")
        return priority

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        announced_ip = getattr(self.config.axon, "external_ip", None) or "auto-detect"
        announced_port = getattr(self.config.axon, "external_port", None) or self.config.axon.port
        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.network} "
            f"with netuid: {self.config.netuid} | bind_port={self.config.axon.port} "
            f"announce={announced_ip}:{announced_port}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        logger.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (FMN minimum-norm engine)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))),
    )

    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = int(getattr(config.axon, "port", getattr(config, "axon_port", 9000)))

    external_ip = (os.getenv("AXON_EXTERNAL_IP") or os.getenv("RUNPOD_PUBLIC_IP") or "").strip()
    config.axon.external_ip = external_ip or None
    external_port_raw = (
        os.getenv("AXON_EXTERNAL_PORT")
        or os.getenv(f"RUNPOD_TCP_PORT_{config.axon.port}")
        or ""
    ).strip()
    config.axon.external_port = int(external_port_raw) if external_port_raw else None

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()
