"""
miner-fmn.py — Perturb subnet miner (netuid 26) with the quantization-aware FMN-L-inf engine.

This is the STOCK neurons/miner.py with its inline small-step PGD block swapped for the strong
minimum-norm engine from flipper-fmn.py. Everything else — wallet/subtensor/axon plumbing,
blacklist/priority, run loop, config — is preserved verbatim so this drops into the real repo:

    cp miner-fmn.py  /home/dev/Perturb/neurons/miner.py      # (or run it directly from there)

Only `PerturbMiner.forward` changed: instead of clamping to +/-epsilon (which busts the 0.03 cap
and scores 0), it calls `perturb(...)` — discrete-k FMN with PNG-verified, time-budgeted search.

────────────────────────────────────────────────────────────────────────────────────────
DEPLOYMENT NOTES (read once)
────────────────────────────────────────────────────────────────────────────────────────
  * Dependencies: torch, torchvision, foolbox, pillow, numpy (same as the lab). foolbox is
    OPTIONAL here — if it isn't installed the engine still runs (probe + PGD fallback path);
    install it to enable the full FMN locate phase:  pip install foolbox
  * The engine is embedded below (one self-contained block) so there is NO import of the
    hyphenated flipper-fmn.py and NO new package to place. It imports only names the real
    perturbnet.* already exposes (image_io, model) plus torch/foolbox.
  * LIVE vs DEFENSIVE constants (the robustness rule): values the synapse TRANSMITS are read
    live every challenge (epsilon, min_delta, timeout_seconds); values it does NOT transmit
    (max_linf_delta, min_ssim, min_psnr) are local defensive defaults, env-overridable below
    so an operator can match a specific validator without code changes.
  * GPU strongly recommended: EfficientNetV2-L at 480^2 is ms/forward on GPU but seconds on
    CPU. The engine is wall-clock budgeted either way and always returns a well-formed reply.
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
    import foolbox as fb  # OPTIONAL — enables the FMN locate phase; engine degrades without it
except Exception:  # pragma: no cover - environment dependent
    fb = None

from perturbnet import constants as _C
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_v2_l, logits_for_images, predict_index, resolve_target_index
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)

# cuDNN autotuning: input H/W is constant within a run, so let cuDNN pick the fastest kernels.
torch.backends.cudnn.benchmark = True

_Q = 1.0 / 255.0  # one 8-bit quantisation step; the finest perturbation a PNG can carry


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Defensive defaults for the NON-transmitted validator gates (env-overridable, mirroring the
# subnet's own env-configurable constants). We never AIM near the cap — we minimise k — so
# even if a validator runs a tighter cap, hugging the floor stays safe.
_MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)     # gate 6 ceiling default
_MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)                 # gate 8 default
_MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)           # gate 9 default
# Headroom subtracted from the live timeout for encode + network round-trip + final verify.
_RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 1.0)


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


# ==========================================================================================
# PERTURBATION ENGINE  (verbatim from flipper-fmn.py; the only edits are: constants read from
# the locals above instead of perturbnet.constants, and a foolbox-optional guard). Public
# entry point is `perturb`; the `_*` helpers are its building blocks. See flipper-fmn.py for
# the full rationale — in brief: PNG quantisation makes the final L-inf an integer k/255, so
# we minimise k (smallest flipping level), verify every candidate on the re-decoded PNG, and
# wall-clock budget the loop. Phases: cheap probes -> FMN locate -> PGD fallback -> RMSE prune.
# ==========================================================================================
class _PreprocessedModel(torch.nn.Module):
    """Adapts the classifier so foolbox can feed it native-res [0,1] images. logits_for_images
    applies the validator's exact resize+crop+normalize, so FMN's gradients flow back to the
    very native pixels the validator measures."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N,3,H,W) in [0,1]
        return logits_for_images(model=self.base, image_bchw=x)


def _png_roundtrip(image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return the image EXACTLY as the validator will see it: encode to PNG, decode back. Bakes
    in 8-bit quantisation so any check here matches scoring reality."""
    return decode_image_b64(encode_image_b64(image_chw)).to(device)


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    """Validator-equivalent avg-pool SSIM (c1=0.01^2, c2=0.03^2)."""
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
    """Validator-equivalent PSNR (data_range=1.0; 99.0 if identical)."""
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _margin_and_grad(model, x_chw, target_index):
    """Untargeted CW-style margin and its gradient wrt the native pixels.

        margin = logit[true] - max_{j != true} logit[j]
    margin < 0  <=>  the image is adversarial. We DESCEND margin. sign(grad)==0 on cropped-out
    / dead pixels, so signed steps never waste budget there."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    true_logit = logits[target_index]
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = true_logit - others.max()
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _signed_step(clean, grad, radius):
    """Steepest-descent L-inf step of size `radius`: move every pixel against sign(grad). Dead-
    gradient pixels (sign 0) stay untouched -> perturbation as sparse as the model's sensitivity."""
    return (clean - radius * grad.sign()).clamp(0.0, 1.0)


def _scale_dir(clean, direction, base_linf, target_linf):
    """Scale a perturbation DIRECTION so its L-inf == target_linf, add to clean, clamp."""
    return (clean + direction * (target_linf / base_linf)).clamp(0.0, 1.0)


def _fmn_direction(model, clean, target_index, device, steps):
    """FMN (minimal-L-inf) perturbation DIRECTION = raw adversarial minus clean, or None if
    foolbox is unavailable / errors. Take attack()'s FIRST return (raw_advs); the SECOND with
    epsilons=None comes back as the CLEAN image."""
    if fb is None:
        return None
    try:
        fmodel = fb.PyTorchModel(_PreprocessedModel(model).to(device).eval(), bounds=(0.0, 1.0))
        attack = fb.attacks.LInfFMNAttack(steps=int(steps))
        labels = torch.tensor([int(target_index)], device=device)  # class to move AWAY from
        raw_advs, _, _ = attack(fmodel, clean.unsqueeze(0), labels, epsilons=None)
        return raw_advs.squeeze(0) - clean
    except Exception:
        return None  # foolbox hiccup -> caller falls back to probes / PGD (reliability first)


def _evaluate(model, clean, cand_chw, target_index, device, floor, cap):
    """Judge a candidate the way the validator will: PNG round-trip, then measure on the decoded
    image. `valid` means ALL hard gates pass (label flipped, floor<=Linf<=cap, SSIM, PSNR).
    SSIM/PSNR are only computed when the cheap checks already pass (they can fail at high k)."""
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
    """Sparse minimum-RMSE L∞ flip, verified in PNG space, time-budgeted to the deadline.

    Objective order: (1) flip at all, (2) smallest k=Linf*255, (3) fewest changed channels
    (= lowest RMSE), all gates passing on the re-decoded PNG. Always returns a valid image.

    Phase 1 — sparse top-k probe: perturb only the highest-|gradient| channels estimated
    to be sufficient to cross the decision boundary; much sparser than all-channel FGSM.
    Phase 2 — FMN direction: finds a continuous minimum-norm direction, scaled to each k.
    Phase 3 — PGD fallback: reliability net when Phases 1-2 miss.
    Phase 4 — three-stage elimination: prefix binary shrink → block chunk → single-pixel
    gradient-guided removal; starts from a sparse candidate so each stage is cheap."""
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

    # =========================== PHASE 1 — sparse top-k probe ==========================
    # Sort channels by |gradient| descending, skip channels that would be erased by
    # clamping, then try the smallest estimated budget and a few larger multiples.
    if m0 > 0.0:
        g_flat = grad0.view(-1)
        clean_flat = clean.view(-1)
        direction_sign = -g_flat.sign()
        can_move = (
            ((direction_sign > 0) & (clean_flat < 1.0)) |
            ((direction_sign < 0) & (clean_flat > 0.0))
        )
        g_abs_valid = g_flat.abs().clone()
        g_abs_valid[~can_move] = 0.0
        sparse_order = torch.argsort(g_abs_valid, descending=True)
        valid_count = int((g_abs_valid > 0).sum().item())

        if valid_count > 0:
            # Linear estimate: need Σ|g_i| over selected set > m0 / (k_min * Q)
            threshold = m0 / max(k_min * _Q, 1e-12)
            cumsum = torch.cumsum(g_abs_valid[sparse_order], dim=0)
            n_base = min(int((cumsum < threshold).sum().item()) + 1, valid_count)
            n_base = max(n_base, 10)

            for mult in [1.0, 1.5, 2.0, 3.0, 5.0]:
                if time_left() <= 1.3 * t_png:
                    break
                n_try = min(int(round(n_base * mult)), valid_count)
                d = torch.zeros_like(g_flat)
                sel_idx = sparse_order[:n_try]
                d[sel_idx] = -(k_min * _Q) * g_flat[sel_idx].sign()
                if consider((clean + d.view_as(clean)).clamp(0.0, 1.0))["valid"]:
                    break

    # FGSM fallback — only if sparse probe found nothing
    if best is None:
        for k in levels[:3]:
            if time_left() <= 1.3 * t_png:
                break
            if consider(_signed_step(clean, grad0, k * _Q))["valid"]:
                break

    # =========================== PHASE 2 — FMN locate + discrete k verify ==============
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

    # =========================== PHASE 3 — PGD fallback ================================
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

    # =========================== PHASE 4 — three-stage elimination =====================
    # Ported from miner_top.py: prefix binary shrink → block chunk → single-pixel.
    # Works with an explicit `sel` list so we can reconstruct any subset cheaply.
    if best is not None and time_left() > t_step + 1.3 * t_png:
        base_cand = best["cand"].clone()
        delta_flat = (base_cand - clean).view(-1)
        sel_tensor = torch.nonzero(delta_flat.abs() > _Q / 4.0, as_tuple=False).view(-1)

        if sel_tensor.numel() > 0:
            sel = sel_tensor.tolist()
            signs_stored = delta_flat.sign().long()  # ±1 per channel, 0 elsewhere

            def _rebuild(sel_list):
                base_u8 = (clean * 255.0).round().clamp(0, 255).long()
                flat_b = base_u8.view(-1).clone()
                if sel_list:
                    st = torch.tensor(sel_list, dtype=torch.long, device=device)
                    s = signs_stored[st]
                    flat_b[st] = (flat_b[st] + s).clamp(0, 255)
                return flat_b.view_as(base_u8).float() / 255.0

            def _still_valid(adv_float):
                res = _evaluate(model, clean, adv_float, target_index, device, floor, cap)
                return res["valid"]

            elim_dl = deadline - 0.12  # tiny safety margin

            # --- 4a: prefix binary shrink — smallest leading prefix that still flips ---
            lo, hi = 1, len(sel)
            best_prefix = sel[:]
            best_prefix_adv = base_cand
            while lo <= hi and time.time() < elim_dl:
                mid = (lo + hi) // 2
                trial_adv = _rebuild(sel[:mid])
                if _still_valid(trial_adv):
                    best_prefix = sel[:mid]
                    best_prefix_adv = trial_adv
                    hi = mid - 1
                else:
                    lo = mid + 1
            if len(best_prefix) < len(sel):
                logger.info(f"[ELIM_PREFIX] k {len(sel)} -> {len(best_prefix)}")
                sel = best_prefix
                base_cand = best_prefix_adv

            # --- 4b: block chunk elimination — remove contiguous blocks ---------------
            block = max(1, len(sel) // 4)
            while block >= 1 and time.time() < elim_dl:
                chunk_removed = False
                starts = list(range(max(0, len(sel) - block), -1, -block))
                if 0 not in starts:
                    starts.append(0)
                for start in starts:
                    if time.time() >= elim_dl:
                        break
                    end = min(start + block, len(sel))
                    if end <= start:
                        continue
                    trial_sel = sel[:start] + sel[end:]
                    if not trial_sel:
                        continue
                    trial_adv = _rebuild(trial_sel)
                    if _still_valid(trial_adv):
                        sel = trial_sel
                        base_cand = trial_adv
                        chunk_removed = True
                        logger.info(f"[ELIM_CHUNK] removed={end-start} k={len(sel)} block={block}")
                        break
                if not chunk_removed:
                    block //= 2

            # --- 4c: single-pixel gradient-guided elimination -------------------------
            pixel_improved = True
            while pixel_improved and time.time() < elim_dl:
                pixel_improved = False
                _, grad_e = _margin_and_grad(model, base_cand, target_index)
                g_e = grad_e.view(-1).abs()
                order = sorted(range(len(sel)), key=lambda i: g_e[sel[i]].item())
                for pos in order:
                    if time.time() > elim_dl:
                        break
                    trial_sel = sel[:pos] + sel[pos + 1:]
                    trial_adv = _rebuild(trial_sel)
                    if _still_valid(trial_adv):
                        sel = trial_sel
                        base_cand = trial_adv
                        pixel_improved = True
                        break  # restart with updated gradient

            logger.info(f"[ELIM_FINAL] k={len(sel)}")
            consider(base_cand)  # update best with the eliminated candidate

    # =========================== RETURN ==================================================
    if best is not None:
        return best["cand"].detach().clamp(0.0, 1.0)
    return _signed_step(clean, grad0, k_max * _Q).detach()


def _warmup(model: torch.nn.Module, device: torch.device) -> None:
    """One fwd+bwd at startup to pay CUDA/cuDNN init up front, so the first real challenge isn't
    slowed by lazy kernel autotuning. Best-effort: any failure is non-fatal."""
    try:
        x = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
        logits_for_images(model=model, image_bchw=x).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as err:  # pragma: no cover
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
    # bittensor's Axon reads port/external_* from EXPLICIT kwargs
    # (signature: wallet, config, port, ip, external_ip, external_port, max_workers).
    # The hand-rolled config object is not a real bt Config, so passing it makes the axon
    # silently fall back to its default port (8091) and skip the announce. Pass kwargs directly.
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

        self.model = load_efficientnet_v2_l(self.device)
        _warmup(self.model, self.device)  # pay one-time CUDA/cuDNN init before the first query

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
        # Stamp arrival time FIRST so decode + the whole handler counts against the deadline.
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
        timeout_seconds = float(getattr(synapse, "timeout_seconds", 15.0))  # live deadline, never hardcoded

        # Strong engine: quantization-aware discrete-k FMN with PNG-verified, time-budgeted search.
        # Wrapped so an unexpected failure still yields a well-formed reply instead of a dead round.
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
            synapse.perturbed_image_b64 = synapse.clean_image_b64  # safe fallback: reply 200, scores 0
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

    # Public endpoint to ANNOUNCE on-chain (axon still binds to config.axon.port internally).
    # Behind NAT (e.g. RunPod) the reachable port differs from the bind port, so prefer explicit
    # AXON_EXTERNAL_* and fall back to RunPod's injected RUNPOD_PUBLIC_IP / RUNPOD_TCP_PORT_<port>.
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
