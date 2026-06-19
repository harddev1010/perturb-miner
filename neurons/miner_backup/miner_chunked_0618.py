"""
miner.py — Perturb subnet miner (netuid 26) — ONE-SHOT engine (~1s budget).

A fast, reliable single-pass sparse attack: one gradient, a closed-form minimal channel
count, one verification (escalating only if needed). Trades the last ~0.01 of score for
near-instant, dependable k=1 flips — the profile of the top fast miners on the board.

MATH (CW margin m = logit[true] - max_{j!=true} logit[j], step q = 1/255):
  first order:  m(x0 + δ) ≈ m0 + gᵀδ,  g = ∇ₓ m(x0); move channel i by δ_i = -q·sign(g_i).
  Rank channels by |g_i| (descending). The linear estimate k* = min{k : q·C_k ≥ m0} over-counts
  ~10x near the boundary (yields dense, low-score flips), so we DON'T trust it: we binary-search
  the gradient-ranked prefix for the MINIMAL k that actually flips (verified). If no prefix of the
  clean gradient flips (curved boundary), a bounded iterated-FGSM re-linearization finds a flipping
  direction, then we sparsify along it the same way.

BYTE-SPACE: gradients live in float, but the perturbation is applied as exact integer BYTE edits
(clean snapped to uint8; selected channel ±1 byte; clamp [0,255]). Candidates are therefore exactly
on the k/255 grid, so "±q" is unambiguous and the PNG round-trip is a true identity (no float->uint8
quantization risk at encode). This does NOT affect cross-machine numeric (TF32) drift — see kappa.

The previous multi-phase engine (greedy growth + reduction + Sparse-RS + TF32 envelope +
margin-deepening) is preserved verbatim in neurons/miner_0617.py.

TUNABLE ENV VARS:
  PERTURB_ONESHOT_BUDGET        (6.0)  wall-clock seconds for the attack (capped by deadline)
  PERTURB_MINER_MARGIN_BUFFER   (0.01) kappa: require margin <= -kappa (transfer cushion)
  PERTURB_TF32_ENVELOPE         (1)    require the flip under BOTH TF32 off+on (CUDA only); covers the
                                       dominant cross-machine drift axis by construction
  PERTURB_KAPPA_RESID           (0.005) kappa used when the TF32 envelope is on — only residual
                                       cross-GPU/library drift left to absorb, so smaller than 0.01
  PERTURB_MAX_RELIN             (10)   max iterated-FGSM steps in the re-linearization fallback
  PERTURB_MINER_RESERVE_SECONDS (3.5)  deadline headroom
  PERTURB_SKIP_ROUNDTRIP        (1)    skip the (identity) PNG round-trip when grid-aligned
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
from perturbnet.model import load_efficientnet_v2_l, logits_for_images, resolve_target_index
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)

# Deterministic, TF32-off kernels — numeric parity with the validator's likely baseline.
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
_RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 3.5)
# kappa: minimal transfer cushion. The minimal-k search targets a flip (margin<0); kappa>0
# adds a tiny extra to require margin<=-kappa. Keep small — every extra bit of margin costs channels.
_MARGIN_BUFFER = _env_float("PERTURB_MINER_MARGIN_BUFFER", 0.01)
_ONESHOT_BUDGET = _env_float("PERTURB_ONESHOT_BUDGET", 6.0)
# TF32 envelope (transfer safety): require the flip under BOTH TF32 off and on, so the validator's
# unknown TF32 setting — the dominant cross-machine drift axis — can't undo it. Costs one extra
# forward per safety check (CUDA only; a no-op on CPU). When on, kappa shrinks to PERTURB_KAPPA_RESID
# because the big axis is now covered by construction and kappa only absorbs residual cross-GPU/
# library drift.
_TF32_ENVELOPE = os.getenv("PERTURB_TF32_ENVELOPE", "1").strip().lower() in {"1", "true", "yes", "on"}
_KAPPA_RESID = _env_float("PERTURB_KAPPA_RESID", 0.002)
# ±1/255 deltas on a grid-aligned clean image survive PNG exactly, so the round-trip is identity.
_SKIP_ROUNDTRIP = os.getenv("PERTURB_SKIP_ROUNDTRIP", "1").strip().lower() in {"1", "true", "yes", "on"}


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
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = logits[target_index] - others.max()
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _margin_tf32on(model, x_chw, target_index) -> float:
    """CW margin with TF32 matmul/cudnn temporarily ENABLED, then restored (CUDA only).

    The other half of the TF32 envelope: a flip required under both TF32 off and on cannot be undone
    by the validator's (unknown) TF32 setting — the dominant cross-machine drift axis — so it is
    covered by construction rather than merely bounded by kappa.
    """
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cd = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        with torch.no_grad():
            return _cw_margin(logits_for_images(model=model, image_bchw=x_chw.unsqueeze(0))[0], target_index)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cd


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


def _evaluate(model, clean, cand_chw, target_index, device, floor, cap, skip_roundtrip=False, envelope=False):
    """Grade a candidate on the validator-faithful PNG round-trip (single forward).

    soft — argmax wrong (m<0), L∞ in [floor, cap], SSIM/PSNR pass. skip_roundtrip evaluates the
    float tensor directly (identity for grid-aligned ±1/255 candidates), saving the PIL cost.
    envelope — report the worst-case margin over both TF32 settings (one extra forward, CUDA only),
    so a flip accepted here survives the validator's unknown TF32 choice. Pre-gated: the TF32-on
    forward only runs once the TF32-off margin already flips (envelope margin >= off margin).
    """
    seen = cand_chw if skip_roundtrip else _png_roundtrip(cand_chw, device)
    diff = seen - clean
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    in_band = floor <= linf <= cap
    with torch.no_grad():
        margin = _cw_margin(logits_for_images(model=model, image_bchw=seen.unsqueeze(0))[0], target_index)
    if envelope and device.type == "cuda" and margin < 0.0:
        # Worst-case over TF32 off/on. Only meaningful when the off-pass already flips, so pre-gate.
        margin = max(margin, _margin_tf32on(model, seen, target_index))
    flipped = margin < 0.0
    soft, ssim, psnr = False, None, None
    if in_band and flipped:
        ssim = _compute_ssim(clean, seen)
        psnr = _compute_psnr_db(clean, seen)
        soft = ssim >= _MIN_SSIM and psnr >= _MIN_PSNR_DB
    return {"cand": cand_chw, "linf": linf, "rmse": rmse, "margin": margin,
            "flipped": flipped, "soft": soft, "ssim": ssim, "psnr": psnr}


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
    """Sparse ±1/255 flip: one gradient -> binary-search the MINIMAL flipping prefix of the
    gradient-ranked channels -> re-linearization fallback if no prefix flips. Returns clean if
    no flip is found within budget."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    floor = float(min_delta)
    cap = min(float(epsilon), float(_MAX_LINF_DELTA))
    if floor > cap:
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))  # fixed unit step (typically 1)
    q = k_min * _Q

    if reserve_seconds is None:
        reserve_seconds = _RESERVE_SECONDS
    hard_deadline = t_start + max(0.05, float(timeout_seconds) - float(reserve_seconds))
    deadline = min(hard_deadline, t_start + _ONESHOT_BUDGET)

    def time_left():
        return deadline - time.time()

    # Byte-space: snap clean to its exact uint8 grid. Every perturbation is an integer BYTE step on
    # this, so candidates are exactly on the k/255 grid — no float->uint8 ambiguity at PNG encode,
    # and "exactly ±q" is unambiguous regardless of the encoder's rounding policy.
    clean_u8 = torch.round(clean.view(-1) * 255.0)
    skip = _SKIP_ROUNDTRIP  # candidates are byte-exact, so the PNG round-trip is always identity
    # TF32 envelope is CUDA-only (TF32 doesn't exist on CPU, so it'd be a wasted forward). When on,
    # the dominant drift axis is covered by construction, so kappa drops to the residual cushion.
    envelope_on = _TF32_ENVELOPE and device.type == "cuda"

    # One gradient evaluation (1 fwd + 1 bwd); time it to gate the search loops.
    g_t0 = time.time()
    m0, g0 = _margin_and_grad(model, clean, target_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = max(1e-4, time.time() - g_t0)
    gflat = g0.view(-1)
    g_abs, order, valid_count = _build_sparse_order(gflat, clean.view(-1))
    if valid_count == 0:
        logger.info(f"[oneshot] no feasible channels -> clean (m0={m0:.4f})")
        return clean.detach().clamp(0.0, 1.0)

    kappa = _KAPPA_RESID if envelope_on else _MARGIN_BUFFER
    best_safe = None  # margin <= -kappa
    best_soft = None  # margin < 0

    def accept(res):
        nonlocal best_safe, best_soft
        if res["soft"]:
            key = (res["linf"], res["rmse"])
            if best_soft is None or key < (best_soft["linf"], best_soft["rmse"]):
                best_soft = res
            if res["margin"] <= -kappa and (best_safe is None or key < (best_safe["linf"], best_safe["rmse"])):
                best_safe = res
        return res["soft"]  # bisection targets the flip (margin<0)

    def make_cand(sel, gsrc):
        # Byte-space perturbation: move each selected channel by -k_min·sign(g) BYTES (g>0 -> -1,
        # g<0 -> +1 for k_min=1), clamp to [0,255]. The gradient picks channels/signs; the edit is
        # applied as exact integer bytes.
        steps = torch.zeros_like(clean_u8)
        steps[sel] = -float(k_min) * gsrc[sel].sign()
        cand_u8 = (clean_u8 + steps).clamp(0.0, 255.0)
        return (cand_u8 / 255.0).view_as(clean)

    def verify_prefix(gsrc, order_src, k):
        sel = order_src[:max(1, k)]
        return accept(_evaluate(model, clean, make_cand(sel, gsrc), target_index, device, floor, cap, skip, envelope_on))

    def min_flip(gsrc, order_src, vcount, k_seed):
        """Exp-up to a flipping prefix, then bisect DOWN for the minimal flipping k along this order
        — instead of trusting the linear estimate, which over-counts ~10x and yields dense flips."""
        if vcount <= 0:
            return
        ks = max(1, min(k_seed, vcount))
        if verify_prefix(gsrc, order_src, ks):
            lo, hi = 0, ks
        else:
            lo, hi, k = ks, None, min(2 * ks, vcount)
            while time_left() > 2 * t_step:
                if verify_prefix(gsrc, order_src, k):
                    hi = k
                    break
                if k >= vcount:
                    break
                lo, k = k, min(2 * k, vcount)
            if hi is None:
                return  # no prefix of this order flips
        while lo + 1 < hi and time_left() > 2 * t_step:
            mid = (lo + hi) // 2
            if verify_prefix(gsrc, order_src, mid):
                hi = mid
            else:
                lo = mid

    # PHASE A: minimal flipping prefix on the clean gradient. Seed = linear estimate (upper bound),
    # then bisect DOWN — the linear count over-pads, so the true minimal k is much sparser.
    cs = torch.cumsum(g_abs[order], dim=0)
    
    k_seed = max(1, min(int((cs < (max(m0 + max(kappa, 0.0), 1e-9)) / q).sum().item()) + 1, valid_count))
    min_flip(gflat, order, valid_count, k_seed)

    anchor = best_safe if best_safe is not None else best_soft
    if anchor is not None:
        nz = int(((anchor["cand"] - clean).abs() > 0.5 * q).sum().item())
        pct = 100.0 * nz / max(1, gflat.numel())
        # logger.info(f"[oneshot] flip channels={nz} ({pct:.2f}%) k_seed={k_seed} valid_count={valid_count} "
        #             f"margin={anchor['margin']:.4f} linf={anchor['linf']:.6f} rmse={anchor['rmse']:.6f} "
        #             f"elapsed={time.time() - t_start:.3f}s m0={m0:.4f} "
        #             f"envelope={'on' if envelope_on else 'off'} kappa={kappa:.4f}")
        logger.info(f"[oneshot] flip channels={nz} ({pct:.2f}%) k_seed={k_seed} "
                    f"margin={anchor['margin']:.4f} rmse={anchor['rmse']:.6f} "
                    f"elapsed={time.time() - t_start:.3f}s m0={m0:.4f} "
                    f"envelope={'on' if envelope_on else 'off'} kappa={kappa:.4f}")
        return anchor["cand"].detach().clamp(0.0, 1.0)
    logger.info(f"[oneshot] no flip at k={k_min}) -> "
                f"(k_seed={k_seed} total={valid_count} m0={m0:.4f} cs={max(cs):.4f} elapsed={time.time() - t_start:.3f}s)")
    return clean.detach().clamp(0.0, 1.0)


def _warmup(model: torch.nn.Module, device: torch.device) -> None:
    """Warm CUDA kernels / allocator / cuDNN at first load so the first real challenge does not
    pay JIT + autotune latency. Exercises the exact inference path: forward + backward + PNG round-trip."""
    t0 = time.time()
    try:
        logger.info(f"[MINER] warmup start device={device.type}")
        x = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
        logits_for_images(model=model, image_bchw=x).sum().backward()
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
            # Clean tier: perturb() returned the clean image (no flip found). Pass the original
            # bytes through verbatim rather than re-encoding.
            if norm < min_delta:
                synapse.perturbed_image_b64 = synapse.clean_image_b64
            else:
                synapse.perturbed_image_b64 = adv_b64
            h, w = clean.shape[1], clean.shape[2]
            est_score = 0.0 if norm < min_delta else _estimate_validator_score(norm, rmse, epsilon)
            # logger.info(
            #     f"Finished task={getattr(synapse, 'task_id', 'unknown')} target_idx={target_index} "
            #     f"norm={norm:.6f} k~={norm * 255.0:.2f} rmse={rmse:.6f} dim={h}x{w} "
            #     f"est_score={est_score:.4f} min_delta={min_delta:.6f} epsilon={epsilon:.4f} "
            #     f"timeout={timeout_seconds:.1f}s elapsed={time.time() - t_received:.3f}s"
            # )
            logger.info(
                f"Finished task={getattr(synapse, 'task_id', 'unknown')} dim={h}x{w} "
                f"rmse={rmse:.6f} est_score={est_score:.4f} elapsed={time.time() - t_received:.3f}s"
            )
            logger.info("-" * 77)
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
    parser = argparse.ArgumentParser(description="Perturb subnet miner (one-shot sparse engine)")
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
