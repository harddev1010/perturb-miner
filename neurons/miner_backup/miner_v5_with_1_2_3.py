"""
miner_v5_with_1_2_3.py — All three strategies combined (highest-ROI path)

CHANGES vs v4:
  Strategy 1 — Target class selection (before Phase 1):
    Probe the N closest wrong classes, pick the one that requires the fewest channels to
    flip to (lowest estimated sparse cost), and use its gradient for Phase 1.
    Expected gain: 20–60% fewer channels when the nearest boundary ≠ max-logit boundary.

  Strategy 2 — Re-linearise at boundary (after Phase 1):
    Once Phase 1 finds a valid candidate, bisect on the line clean→adv to locate the
    decision boundary x_bnd, recompute the gradient at x_bnd, and run a second binary
    search.  The boundary gradient is tangent to the true decision surface.
    Expected gain: 10–40% additional reduction on top of Strategy 1.

  Strategy 3 — FMN-L0 alternative phase (after re-linearize, before FMN-LInf):
    Run foolbox L0FMNAttack and snap the result to ±1/255.  L0FMN explicitly minimises
    the number of changed channels via iterative projected gradient descent, providing a
    completely different trajectory from the gradient-sorted binary search.
    Expected gain: wins on images where both Phase 1 and the re-linearised search miss a
    sparser global path.

PHASE ORDER:
  [target selection] → Phase 1 binary search → [re-linearize] → Phase 2b FMN-L0
  → Phase 2 FMN-LInf → Phase 3 PGD fallback

All phases share the same consider() that tracks the globally best (linf, rmse) pair.
Any phase can improve on a prior phase's result; the best is returned at the end.
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

try:
    import foolbox as fb
except Exception:
    fb = None

from perturbnet import constants as _C
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_v2_l, logits_for_images, predict_index, resolve_target_index
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
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
_RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 3.0)
_TARGET_PROBE_N = int(os.getenv("PERTURB_TARGET_PROBE_N", "10"))
_RELINEARIZE_BISECT_STEPS = int(os.getenv("PERTURB_RELINEARIZE_BISECT_STEPS", "6"))
_FMN_L0_TIME_FRAC = float(os.getenv("PERTURB_FMN_L0_TIME_FRAC", "0.35"))


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


class _PreprocessedModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return logits_for_images(model=self.base, image_bchw=x)


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
    """Untargeted CW margin: logit[true] - max_{j≠true} logit[j]."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    true_logit = logits[target_index]
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = true_logit - others.max()
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _targeted_margin_and_grad(model, x_chw, true_index, attack_class):
    """Targeted margin: logit[true] - logit[attack_class]."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    margin = logits[true_index] - logits[attack_class]
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _build_sparse_order(grad_flat, clean_flat):
    """Gradient-sorted order and valid-move mask for sparse perturbation."""
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
                               m_use, consider, time_left, t_png):
    """Exponential doubling + binary search to find minimum channel count that flips label."""
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
        if consider(_make_sparse(hi))["valid"]:
            first_flip_n = hi
            break
        lo = hi + 1
        hi = min(hi * 2, valid_count)

    if first_flip_n is not None:
        blo, bhi = lo, first_flip_n
        while blo < bhi and time_left() > 1.3 * t_png:
            mid = (blo + bhi) // 2
            if consider(_make_sparse(mid))["valid"]:
                bhi = mid
            else:
                blo = mid + 1

    return first_flip_n


def _select_best_target(model, clean, true_index, device, k_min, n_probe, time_budget):
    """Pick the wrong class that minimises the estimated sparse channel count."""
    with torch.no_grad():
        all_logits = logits_for_images(model=model, image_bchw=clean.unsqueeze(0))[0]
    n_classes = all_logits.shape[0]
    candidates = sorted(
        [(float(all_logits[true_index].item() - all_logits[i].item()), i)
         for i in range(n_classes) if i != true_index]
    )
    best_n = float("inf")
    best_class = None
    best_grad = None
    best_order = None
    best_g_abs = None
    best_vc = 0
    clean_flat = clean.view(-1)
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
        if n_t < best_n:
            best_n = n_t
            best_class = t
            best_grad = g_t
            best_order = order_t
            best_g_abs = g_abs_t
            best_vc = vc_t
    return best_class, best_grad, best_order, best_g_abs, best_vc


def _signed_step(clean, grad, radius):
    return (clean - radius * grad.sign()).clamp(0.0, 1.0)


def _scale_dir(clean, direction, base_linf, target_linf):
    return (clean + direction * (target_linf / base_linf)).clamp(0.0, 1.0)


def _fmn_direction(model, clean, target_index, device, steps):
    """FMN-LInf minimum-norm direction (Phase 2 fallback)."""
    if fb is None:
        return None
    try:
        fmodel = fb.PyTorchModel(_PreprocessedModel(model).to(device).eval(), bounds=(0.0, 1.0))
        attack = fb.attacks.LInfFMNAttack(steps=int(steps))
        labels = torch.tensor([int(target_index)], device=device)
        raw_advs, _, _ = attack(fmodel, clean.unsqueeze(0), labels, epsilons=None)
        return raw_advs.squeeze(0) - clean
    except Exception:
        return None


def _fmn_l0_snapped(model, clean, target_index, device, steps):
    """L0FMNAttack result snapped to ±1/255 on changed channels."""
    if fb is None:
        return None
    try:
        L0Attack = getattr(fb.attacks, "L0FMNAttack", None)
        if L0Attack is None:
            return None
        fmodel = fb.PyTorchModel(_PreprocessedModel(model).to(device).eval(), bounds=(0.0, 1.0))
        attack = L0Attack(steps=int(steps))
        labels = torch.tensor([int(target_index)], device=device)
        raw_advs, _, _ = attack(fmodel, clean.unsqueeze(0), labels, epsilons=None)
        delta = raw_advs.squeeze(0) - clean
        mask = delta.abs() > 1e-9
        snapped_delta = torch.zeros_like(delta)
        snapped_delta[mask] = delta[mask].sign() * _Q
        return (clean + snapped_delta).clamp(0.0, 1.0)
    except Exception:
        return None


def _evaluate(model, clean, cand_chw, target_index, device, floor, cap):
    seen = _png_roundtrip(cand_chw, device)
    diff = seen - clean
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    in_band = floor <= linf <= cap
    flipped = predict_index(model=model, image_chw=seen) != target_index
    valid, ssim, psnr = False, None, None
    if in_band and flipped:
        ssim = _compute_ssim(clean, seen)
        psnr = _compute_psnr_db(clean, seen)
        valid = ssim >= _MIN_SSIM and psnr >= _MIN_PSNR_DB
    return {"cand": cand_chw, "linf": linf, "rmse": rmse, "flipped": flipped,
            "valid": valid, "ssim": ssim, "psnr": psnr}


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
    """Sparse minimum-RMSE L∞ flip: target class selection + re-linearize + FMN-L0 (all three)."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(_MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))
    k_max = max(k_min, int(math.floor(cap * 255.0 + 1e-6)))
    levels = list(range(k_min, k_max + 1))

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

    best = None

    def consider(cand_chw):
        nonlocal best, t_png
        c0 = time.time()
        res = _evaluate(model, clean, cand_chw, target_index, device, floor, cap)
        t_png = max(1e-4, time.time() - c0)
        if res["valid"] and (best is None or (res["linf"], res["rmse"]) < (best["linf"], best["rmse"])):
            best = res
        return res

    # =========================== STRATEGY 1: TARGET CLASS SELECTION =====================
    grad_use = grad0
    sparse_order_use = None
    g_abs_use = None
    valid_count_use = 0

    if m0 > 0.0 and time_left() > (_TARGET_PROBE_N + 2) * t_step:
        probe_budget_end = time.time() + _TARGET_PROBE_N * t_step * 1.5
        best_t, best_tgrad, best_torder, best_tg_abs, best_tvc = _select_best_target(
            model, clean, target_index, device, k_min,
            n_probe=_TARGET_PROBE_N,
            time_budget=min(probe_budget_end, deadline - 3 * t_step),
        )
        if best_tgrad is not None and best_tvc > 0:
            grad_use = best_tgrad
            sparse_order_use = best_torder
            g_abs_use = best_tg_abs
            valid_count_use = best_tvc
            logger.debug(f"[target_sel] chose class={best_t}")

    # =========================== PHASE 1: sparse top-k binary search ====================
    if m0 > 0.0:
        g_flat = grad_use.view(-1)
        clean_flat = clean.view(-1)

        if sparse_order_use is None or g_abs_use is None:
            g_abs_use, sparse_order_use, valid_count_use = _build_sparse_order(g_flat, clean_flat)

        if valid_count_use > 0:
            _run_sparse_binary_search(clean, g_flat, k_min, valid_count_use,
                                      sparse_order_use, g_abs_use, m0, consider, time_left, t_png)

    # FGSM fallback (uses original untargeted gradient for reliability)
    if best is None:
        for k in levels[:3]:
            if time_left() <= 1.3 * t_png:
                break
            if consider(_signed_step(clean, grad0, k * _Q))["valid"]:
                break

    # =========================== STRATEGY 2: RE-LINEARIZE AT BOUNDARY ==================
    _bisect_steps = _RELINEARIZE_BISECT_STEPS
    needed_2 = _bisect_steps * t_png + t_step + 4 * t_png
    if best is not None and time_left() > needed_2:
        x_adv = best["cand"].detach()
        delta_dir = x_adv - clean

        lam_lo, lam_hi = 0.0, 1.0
        for _ in range(_bisect_steps):
            if time_left() <= t_step + 3 * t_png:
                break
            lam_mid = (lam_lo + lam_hi) / 2.0
            x_mid = (clean + lam_mid * delta_dir).clamp(0.0, 1.0)
            x_mid_png = _png_roundtrip(x_mid, device)
            if predict_index(model=model, image_chw=x_mid_png) != target_index:
                lam_hi = lam_mid
            else:
                lam_lo = lam_mid

        if time_left() > t_step + 2 * t_png:
            x_bnd = (clean + lam_hi * delta_dir).clamp(0.0, 1.0)
            m_bnd, grad_bnd = _margin_and_grad(model, x_bnd, target_index)
            logger.debug(f"[relinearize] lam_hi={lam_hi:.4f} m_bnd={m_bnd:.4f}")

            if m_bnd < 1.0 and time_left() > 2 * t_png:
                g2 = grad_bnd.view(-1)
                c2 = clean.view(-1)
                g2_valid, ord2, vc2 = _build_sparse_order(g2, c2)

                if vc2 > 0:
                    m2 = max(abs(m_bnd), 0.05)
                    _run_sparse_binary_search(clean, g2, k_min, vc2, ord2, g2_valid,
                                              m2, consider, time_left, t_png)

    # =========================== STRATEGY 3: FMN-L0 =====`==============================
    won_k_min = best is not None and best["linf"] <= k_min * _Q + 1e-9
    if not won_k_min and _FMN_L0_TIME_FRAC > 0.0 and time_left() > 2 * t_step:
        l0_time = _FMN_L0_TIME_FRAC * time_left()
        l0_steps = max(20, int(l0_time / t_step))
        logger.debug(f"[fmn_l0] steps={l0_steps} budget={l0_time:.2f}s")
        cand_l0 = _fmn_l0_snapped(model, clean, target_index, device, l0_steps)
        if cand_l0 is not None and time_left() > 1.3 * t_png:
            res_l0 = consider(cand_l0)
            logger.debug(f"[fmn_l0] valid={res_l0['valid']} linf={res_l0['linf']:.6f} rmse={res_l0['rmse']:.6f}")

    # =========================== PHASE 2: FMN-LInf =====================================
    won_k_min = best is not None and best["linf"] <= k_min * _Q + 1e-9
    affordable = int(0.55 * time_left() / t_step)
    if not won_k_min and affordable >= 4:
        fmn_steps = int(steps) if steps else min(60, affordable)
        direction = _fmn_direction(model, clean, target_index, device, fmn_steps)
        if direction is not None:
            base_linf = float(direction.abs().max().item())
            if base_linf > 1e-12:
                for k in levels:
                    if time_left() <= 1.3 * t_png:
                        break
                    if best is not None and k * _Q >= best["linf"] - 1e-12:
                        break
                    if consider(_scale_dir(clean, direction, base_linf, k * _Q))["valid"]:
                        break

    # =========================== PHASE 3: PGD fallback =================================
    if best is None:
        delta = torch.zeros_like(clean)
        for k in levels:
            r = k * _Q
            alpha = max(_Q, r / 4.0)
            delta = delta.clamp(-r, r)
            local = 0
            while time_left() > t_step + 1.3 * t_png and local < 12:
                x = (clean + delta).clamp(0.0, 1.0)
                margin, grad = _margin_and_grad(model, x, target_index)
                if margin < 0.0 and consider(x)["valid"]:
                    break
                delta = ((clean + delta - alpha * grad.sign()).clamp(0.0, 1.0) - clean).clamp(-r, r)
                local += 1
            if best is not None:
                break

    if best is not None:
        return best["cand"].detach().clamp(0.0, 1.0)
    return _signed_step(clean, grad0, k_max * _Q).detach()


def _warmup(model: torch.nn.Module, device: torch.device) -> None:
    try:
        x = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
        logits_for_images(model=model, image_bchw=x).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as err:
        logger.warning(f"[MINER] warmup skipped: {err}")


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
    pylogging.basicConfig(level=level, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    pylogging.getLogger().setLevel(level)


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_efficientnet_v2_l(self.device)
        _warmup(self.model, self.device)
        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(forward_fn=self.forward, blacklist_fn=self.blacklist, priority_fn=self.priority)

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
        self._log_step_start("miner_forward", task_id=getattr(synapse, "task_id", "unknown"),
                             norm_type=getattr(synapse, "norm_type", "unknown"),
                             epsilon=getattr(synapse, "epsilon", "unknown"))
        if synapse.norm_type != "Linf":
            logger.info(f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unsupported norm_type={synapse.norm_type}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        target_index = resolve_target_index(synapse.true_label)
        if target_index is None:
            logger.warning(f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unresolved true_label={getattr(synapse, 'true_label', None)}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = float(synapse.epsilon)
        min_delta = float(getattr(synapse, "min_delta", 0.002))
        timeout_seconds = float(getattr(synapse, "timeout_seconds", 15.0))

        try:
            adv = perturb(
                self.model, clean, target_index, epsilon, min_delta, self.device,
                timeout_seconds=timeout_seconds, start_time=t_received,
            )
            synapse.perturbed_image_b64 = encode_image_b64(adv)
            seen = decode_image_b64(synapse.perturbed_image_b64).to(self.device)
            diff = seen - clean
            norm = float(diff.abs().max().item())
            rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
            h, w = clean.shape[1], clean.shape[2]
            est_score = _estimate_validator_score(norm, rmse, epsilon)
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
        self._log_step_start("miner_blacklist", task_id=getattr(synapse, "task_id", "unknown"),
                             caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None))
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
        self._log_step_start("miner_priority", task_id=getattr(synapse, "task_id", "unknown"),
                             caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None))
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
    parser = argparse.ArgumentParser(description="Perturb subnet miner (all three strategies)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint", dest="chain_endpoint", type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port", dest="axon_port", type=int,
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
        os.getenv("AXON_EXTERNAL_PORT") or os.getenv(f"RUNPOD_TCP_PORT_{config.axon.port}") or ""
    ).strip()
    config.axon.external_port = int(external_port_raw) if external_port_raw else None
    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))
    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()
