"""
miner_qu_sparsefool.py — Perturb subnet miner (netuid 26) — Quantized SparseFool engine

A full-algorithm ALTERNATIVE to the gradient-rank + binary-search miner.py. Instead of
ranking the clean-image gradient once, this miner uses decision-BOUNDARY GEOMETRY: a
DeepFool walk locates the nearest boundary and its normal, the normal is sparsified into
±1/255 channel moves, and the support is grown by a real-forward binary search. This is the
"Quantized SparseFool-style attack" — SparseFool (Modas et al., 2019) was designed
specifically for sparse adversarial perturbations via boundary geometry rather than one-shot
saliency sorting, which matches this challenge's "change as few channels as possible, each
exactly ±1/255" objective far better than dense PGD/CW.

WHY BOUNDARY GEOMETRY (vs. clean-image gradient ranking):
  The clean gradient g(x0) is tangent to the loss surface at x0, but the channels that
  actually matter are the ones tangent to the surface AT THE BOUNDARY. DeepFool steps to the
  boundary and we re-linearize there, so the sparse direction is the true boundary normal —
  it escapes the weakness of clean-image gradients on nonlinear models.

CORE FLOW (m0 > 0, true class still on top):
  bank-a-flip sweep
  → for a few outer rounds:
       DeepFool walk to boundary  →  recompute the targeted boundary normal there
       →  sparsify the normal into ±1/255 channel priorities
       →  binary-search the support count (real PNG round-trip forwards)
       →  advance x to the boundary and re-linearize next round
  → safety-grow (deepen margin to m <= -buffer)
  → cleanup/prune (drop unneeded channels; lower rmse)
  → gated tiered finalize
ALREADY-MISCLASSIFIED (m0 <= 0):
  bank-a-flip sweep → minimal ±1/255 sparse top-saliency search → safety-grow → prune

EVERY candidate — from any phase — is routed through one margin-buffer gate (_evaluate on
the lossless PNG round-trip): a candidate is a "soft" flip when the argmax is already wrong
and SSIM/PSNR pass, and "margin-safe" only when the CW margin m = logit[true] -
max_{j!=true} logit[j] <= -buffer. Only margin-safe candidates are trusted to transfer to
the validator's exact weights; soft flips are the fallback tier; the clean image is the
last resort. This makes a non-transferring (score-0) dense image impossible to return.

SCORING (validator mirror, SPEED_WEIGHT=0):
  score = 0.7 * linf_score + 0.3 * rmse_score
  linf_score = (1 - (norm - MIN)/(min(eps,0.03) - MIN))^2 ; rmse_score = (1 - rmse/eps_eff)^2
  Each changed channel = ±1/255 gives norm ~= 0.003922 -> linf_score ~= 0.933 (the optimum);
  fewer changed channels -> lower rmse -> higher rmse_score. So: flip reliably, hold linf at
  1/255, minimise channel count.

TUNABLE ENV VARS:
  PERTURB_MINER_MARGIN_BUFFER      (default 0.01) — transfer-safety margin (m <= -buffer)
  PERTURB_SF_NUM_CLASSES           (default 5)    — competing classes linearized per DeepFool step
  PERTURB_SF_DEEPFOOL_ITERS        (default 3)    — max DeepFool steps per boundary walk
  PERTURB_SF_OUTER_ROUNDS          (default 3)    — boundary re-linearization rounds
  PERTURB_SF_OVERSHOOT             (default 0.02) — DeepFool overshoot past the boundary
  PERTURB_PRUNE_ENABLE             (default 1)    — greedy cleanup of unneeded channels
  PERTURB_MINER_RESERVE_SECONDS    (default 2.5)  — deadline headroom
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

# Numeric parity with the live validator (deterministic matmul / conv kernels). The validator
# decodes lossless PNG and runs a single forward; any TF32 / autotuned-kernel divergence
# between miner and validator can silently flip a borderline margin, so we pin the exact same
# numerics the validator uses.
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
_MARGIN_BUFFER = _env_float("PERTURB_MINER_MARGIN_BUFFER", 0.01)
_SF_NUM_CLASSES = max(1, int(os.getenv("PERTURB_SF_NUM_CLASSES", "5")))
_SF_DEEPFOOL_ITERS = max(1, int(os.getenv("PERTURB_SF_DEEPFOOL_ITERS", "3")))
_SF_OUTER_ROUNDS = max(1, int(os.getenv("PERTURB_SF_OUTER_ROUNDS", "3")))
_SF_OVERSHOOT = _env_float("PERTURB_SF_OVERSHOOT", 0.02)
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


def _margin_and_grad(model, x_chw, target_index):
    """Untargeted CW margin: logit[true] - max_{j≠true} logit[j], and its gradient."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    true_logit = logits[target_index]
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = true_logit - others.max()
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _targeted_margin_and_grad(model, x_chw, true_index, attack_class):
    """Targeted margin: logit[true] - logit[attack_class], and its gradient (the boundary
    normal between the two classes)."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    margin = logits[true_index] - logits[attack_class]
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _build_sparse_order(grad_flat, clean_flat):
    """Gradient-sorted order and valid-move mask for a sparse perturbation.

    To reduce the quantity whose gradient is `grad_flat`, each channel moves by
    -sign(grad_i). A channel is movable only if it can move that way inside [0,1]; the
    unmovable channels are zeroed before the descending |grad| sort.
    """
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


def _run_sparse_binary_search(clean, g_flat, k_min, valid_count, sparse_order, g_abs_valid,
                              m_use, consider, time_left, t_png, accept_key="soft"):
    """Exponential doubling + binary search for the minimum channel count that flips.

    The accept predicate is consider(...)[accept_key] — by default "soft" (an in-band,
    quality-passing argmax flip). We search on the FLIP itself: locking the minimal flip fast
    banks score under the clock; deepening to m <= -buffer is left to safety-grow afterwards.
    """
    threshold = m_use / max(k_min * _Q, 1e-12)
    cumsum = torch.cumsum(g_abs_valid[sparse_order], dim=0)
    n_base = min(int((cumsum < threshold).sum().item()) + 1, valid_count)
    n_base = max(n_base, 1)

    def _make_sparse(n):
        d = torch.zeros_like(g_flat)
        d[sparse_order[:n]] = -(k_min * _Q) * g_flat[sparse_order[:n]].sign()
        return (clean + d.view_as(clean)).clamp(0.0, 1.0)

    lo, hi = n_base, n_base
    first_flip_n = None
    while hi <= valid_count and time_left() > 1.3 * t_png:
        if consider(_make_sparse(hi))[accept_key]:
            first_flip_n = hi
            break
        lo = hi + 1
        hi = min(hi * 2, valid_count)

    if first_flip_n is not None:
        blo, bhi = lo, first_flip_n
        while blo < bhi and time_left() > 1.3 * t_png:
            mid = (blo + bhi) // 2
            if consider(_make_sparse(mid))[accept_key]:
                bhi = mid
            else:
                blo = mid + 1

    return first_flip_n


def _deepfool_boundary(model, x, true_index, num_classes, max_iter, overshoot, time_left, t_step):
    """DeepFool walk to the nearest decision boundary (the geometry SparseFool sparsifies).

    Returns (x_boundary, chosen_class): a point on/just past the boundary, and the competing
    class whose linearized boundary was crossed (None if the walk could not start). Each
    iteration linearizes the top `num_classes` competing logits at the current point, picks
    the class with the smallest minimal-perturbation ratio |f_k| / ||w_k||_2 (DeepFool's
    nearest-boundary criterion), and takes the minimal-L2 step toward it. Restricted to the
    top classes so cost is ~num_classes backward passes per iteration; time-gated throughout.
    """
    x_i = x.detach().clone()
    r_tot = torch.zeros_like(x)
    chosen = None
    for _ in range(max_iter):
        if time_left() <= 2.5 * t_step:
            break
        xv = x_i.detach().clone().requires_grad_(True)
        logits = logits_for_images(model=model, image_bchw=xv.unsqueeze(0))[0]
        if int(logits.argmax().item()) != true_index:
            break
        order = torch.argsort(logits, descending=True).tolist()
        cands = [c for c in order if c != true_index][:num_classes]
        grad_true = torch.autograd.grad(logits[true_index], xv, retain_graph=True)[0].view(-1)
        best = None
        for k in cands:
            if time_left() <= 2.0 * t_step:
                break
            grad_k = torch.autograd.grad(logits[k], xv, retain_graph=True)[0].view(-1)
            w_k = grad_k - grad_true
            f_k = float(logits[k].item() - logits[true_index].item())
            nrm = float(w_k.norm().item()) + 1e-12
            ratio = abs(f_k) / nrm
            if best is None or ratio < best[0]:
                best = (ratio, w_k, abs(f_k), nrm, k)
        if best is None:
            break
        _, w_l, abs_f, nrm, k = best
        r_i = (abs_f / (nrm * nrm)) * w_l
        r_tot = r_tot + r_i.view_as(r_tot)
        x_i = (x + (1.0 + overshoot) * r_tot).clamp(0.0, 1.0)
        chosen = k
    return x_i.detach(), chosen


def _prune(clean, anchor, tier_key, grad_use, k_min, consider, time_left, t_png):
    """Greedy reverse of safety-grow: drop the least-salient changed channels in batches while
    the candidate stays in its tier ("valid" margin-safe, or "soft"). consider() re-banks an
    improved (same-linf, lower-rmse) candidate automatically; this only tracks the working
    delta so accepted removals compound. linf is unaffected (survivors stay ±1/255)."""
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
            res = consider((clean + trial.view_as(clean)).clamp(0.0, 1.0))
            if res[tier_key]:
                cur = trial  # accept removal; keep shrinking from here


def _evaluate(model, clean, cand_chw, target_index, device, floor, cap, buffer):
    """Evaluate a candidate on the validator-faithful PNG round-trip.

    Computes the CW margin m = logit[true] - max_{j!=true} logit[j] on the decoded image and
    returns two grades:
      soft  — argmax already wrong (m < 0) and in-band SSIM/PSNR pass: a usable but thin flip.
      valid — margin-safe: soft AND m <= -buffer, i.e. pushed past the boundary by the buffer.
    Only margin-safe candidates are trusted to transfer; soft flips are the fallback tier.
    """
    seen = _png_roundtrip(cand_chw, device)
    diff = seen - clean
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    in_band = floor <= linf <= cap
    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=seen.unsqueeze(0))[0]
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = float((logits[target_index] - others.max()).item())
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


def _kfixed_feasible(model, clean, target_index, k_min, time_left, t_step, t_png, max_iters=4):
    """Is the image flippable by ANY perturbation confined to the ±(k_min/255) L∞ ball?

    k is fixed at 1/255 and the objective is the minimum-rmse flip — never a larger step. So
    the first question is binary: can a 1/255 perturbation flip the argmax AT ALL? Iterated
    FGSM inside the ±(k_min/255) ball is the strongest such attack; if it never drives the CW
    margin below 0, the image is unflippable at the fixed k and there is no sparse subset to
    find. This probe is DENSE — used only for the yes/no feasibility answer, never returned as
    a candidate (a dense step fails the SSIM/PSNR quality gate).
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
    """Quantized SparseFool: boundary geometry → sparse ±1/255 → support search → grow/prune."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(_MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))

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

    # Two gated trackers — never an ungated image.
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

    # Default sparse order from the untargeted gradient; the boundary walk refines it.
    grad_use = grad0
    g_abs_use, sparse_order_use, valid_count_use = _build_sparse_order(grad0.view(-1), clean.view(-1))

    # =========================== BANK A FLIP ASAP (anytime floor) ======================
    # Lock in *some* gated flip before spending budget on boundary geometry, so a clean-tier
    # (score 0) return is impossible whenever a flip is reachable. The SparseFool rounds below
    # then drive the channel count (and rmse) back down.
    full_step_flipped = False  # did the full dense ±1/255 sign step flip argmax? (k=1 feasibility)
    if valid_count_use > 0:
        for frac in (0.02, 0.1, 0.4, 1.0):
            if best_soft is not None or time_left() <= 1.3 * t_png:
                break
            n = max(1, min(valid_count_use, int(frac * valid_count_use)))
            d = torch.zeros_like(grad0.view(-1))
            d[sparse_order_use[:n]] = -(k_min * _Q) * grad0.view(-1)[sparse_order_use[:n]].sign()
            res_b = consider((clean + d.view_as(clean)).clamp(0.0, 1.0))
            if n >= valid_count_use:
                full_step_flipped = res_b["flipped"]
        if best_soft is not None:
            logger.debug(f"[bank] flip banked n_evals={n_evals} margin={best_soft['margin']:.4f}")

    # =========================== k=1 FEASIBILITY GATE =================================
    # Can a 1/255 perturbation flip this image AT ALL? The full dense sign step is the strongest
    # single 1/255 attack; if it (and a short in-ball PGD confirmation) cannot flip the argmax,
    # the image is unflippable at k=1 — skip the boundary walk and return clean FAST rather than
    # spending the whole timeout. When m0>0 and infeasible, no flip is banked, so control falls
    # straight through to the clean finalize.
    feasible = True
    if m0 > 0.0:
        feasible = full_step_flipped
        if not feasible and time_left() > 4 * t_step:
            feasible = _kfixed_feasible(model, clean, target_index, k_min, time_left, t_step, t_png)
        if not feasible:
            logger.info(f"[feasibility] unflippable at k={k_min} "
                        f"(best_margin_seen={best_margin_seen:.4f}); skipping search -> clean")

    if m0 > 0.0 and feasible:
        # =================== QUANTIZED SPARSEFOOL OUTER ROUNDS =========================
        # Each round walks to the boundary, re-linearizes the targeted boundary normal there,
        # sparsifies it into ±1/255 channel priorities, and binary-searches the support. The
        # next round re-linearizes from the new boundary point — escaping the clean-gradient's
        # bias. The sparse candidate is always (clean + sparse ±1/255), so linf stays at the
        # 1/255 optimum no matter how many rounds run.
        x_cur = clean
        for rnd in range(_SF_OUTER_ROUNDS):
            if time_left() <= (_SF_NUM_CLASSES + 3) * t_step:
                break
            x_bnd, chosen = _deepfool_boundary(
                model, x_cur, target_index, _SF_NUM_CLASSES, _SF_DEEPFOOL_ITERS,
                _SF_OVERSHOOT, time_left, t_step,
            )
            if chosen is not None and time_left() > 2.0 * t_step:
                _, g_bnd = _targeted_margin_and_grad(model, x_bnd, target_index, chosen)
            else:
                # Boundary walk stalled — fall back to the untargeted margin normal here.
                _, g_bnd = _margin_and_grad(model, x_bnd, target_index)
            g_flat = g_bnd.view(-1)
            g_abs_b, order_b, vc_b = _build_sparse_order(g_flat, clean.view(-1))
            if vc_b == 0:
                break
            soft_before = best_soft
            _run_sparse_binary_search(clean, g_flat, k_min, vc_b, order_b, g_abs_b,
                                      m0, consider, time_left, t_png)
            # Adopt the boundary normal's order for safety-grow / prune if it improved things.
            if best_soft is not None and best_soft is not soft_before:
                grad_use, sparse_order_use = g_bnd, order_b
            # A sparse ±1/255 flip already sits at the linf optimum: once we hold a flip at the
            # k_min step there is nothing more linf can buy, so stop walking.
            anchor = best_safe if best_safe is not None else best_soft
            if anchor is not None and anchor["linf"] <= k_min * _Q + 1e-9 and rnd >= 1:
                break
            x_cur = x_bnd
    elif m0 <= 0.0:
        # =================== ALREADY MISCLASSIFIED (m0 <= 0) ===========================
        # The clean image is already wrong. Emit a minimal ±1/255 sparse perturbation on the
        # top-saliency channels keeping the argmax wrong with m <= -buffer, via the same gate.
        logger.debug(f"[already_misclassified] m0={m0:.4f}")
        if valid_count_use > 0:
            _run_sparse_binary_search(clean, grad0.view(-1), k_min, valid_count_use,
                                      sparse_order_use, g_abs_use, max(abs(m0), 0.05),
                                      consider, time_left, t_png)

    # =========================== SAFETY-GROW ==========================================
    # The search accepts on the flip itself, so the held flip may be a thin soft flip (m < 0
    # but > -buffer). Deepen it: append up to ~64 top-saliency ±1/255 channels until
    # m <= -buffer (ideally <= -2*buffer). linf stays at k_min/255; only rmse rises a little.
    anchor = best_safe if best_safe is not None else best_soft
    if (anchor is not None and sparse_order_use is not None
            and anchor["margin"] > -2.0 * _MARGIN_BUFFER):
        grad_flat = grad_use.view(-1)
        cur_delta = (anchor["cand"].detach() - clean).view(-1).clone()
        changed = cur_delta.abs() > (0.5 * _Q)
        n_changed = int(changed.sum().item())
        window = min(sparse_order_use.numel(), n_changed + 4096)
        to_add = []
        for idx in sparse_order_use[:window].tolist():
            if len(to_add) >= 64:
                break
            if not changed[idx]:
                to_add.append(idx)
        i, batch = 0, 16
        while i < len(to_add) and time_left() > 1.3 * t_png:
            for idx in to_add[i:i + batch]:
                cur_delta[idx] = -(k_min * _Q) * grad_flat[idx].sign()
            i += batch
            cand_g = (clean + cur_delta.view_as(clean)).clamp(0.0, 1.0)
            res_g = consider(cand_g)
            if res_g["valid"] and (best_safe is None or res_g["margin"] < best_safe["margin"]):
                best_safe = res_g
            if best_safe is not None and best_safe["margin"] <= -2.0 * _MARGIN_BUFFER:
                break

    # =========================== CLEANUP / PRUNE ======================================
    # Lowest-priority step. safety-grow only ADDS channels; some may be unneeded. With leftover
    # time, greedily drop the least-salient changed channels while the candidate stays in its
    # tier. linf unchanged; rmse drops with fewer channels — a direct rmse_score gain.
    if _PRUNE_ENABLE and sparse_order_use is not None and time_left() > 2.0 * t_png:
        if best_safe is not None:
            _prune(clean, best_safe, "valid", grad_use, k_min, consider, time_left, t_png)
        elif best_soft is not None:
            _prune(clean, best_soft, "soft", grad_use, k_min, consider, time_left, t_png)

    # =========================== GATED TIERED FINALIZE ================================
    diag = (f"best_margin_seen={best_margin_seen:.4f} n_evals={n_evals} "
            f"time_left={time_left():.2f}s deadline_hit={time_left() <= 1.3 * t_png}")
    if best_safe is not None:
        logger.info(f"[finalize] tier=margin_safe margin={best_safe['margin']:.4f} "
                    f"linf={best_safe['linf']:.6f} rmse={best_safe['rmse']:.6f} {diag}")
        return best_safe["cand"].detach().clamp(0.0, 1.0)
    if best_soft is not None:
        logger.info(f"[finalize] tier=soft_flip margin={best_soft['margin']:.4f} "
                    f"linf={best_soft['linf']:.6f} rmse={best_soft['rmse']:.6f} {diag}")
        return best_soft["cand"].detach().clamp(0.0, 1.0)
    logger.info(f"[finalize] tier=clean (no gated flip found; returning clean image) {diag}")
    return clean.detach().clamp(0.0, 1.0)


def _warmup(model: torch.nn.Module, device: torch.device) -> None:
    try:
        x = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
        logits_for_images(model=model, image_bchw=x).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
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
            if norm < min_delta:
                synapse.perturbed_image_b64 = synapse.clean_image_b64
            else:
                synapse.perturbed_image_b64 = adv_b64
            h, w = clean.shape[1], clean.shape[2]
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
    parser = argparse.ArgumentParser(description="Perturb subnet miner (Quantized SparseFool engine)")
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
