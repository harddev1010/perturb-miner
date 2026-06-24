"""Shared primitives for the perturb attack engine.

Everything the algorithms stand on: validator-faithful metrics, CW margin / loss gradients
(scalar + batched), byte-space candidate construction, the batched-forward evaluator, the
best-candidate Bank, and the Context bundle handed to every algorithm.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from perturbnet import constants as _C
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import logits_for_images

from . import constants as K


# ==========================================================================================
# Validator-faithful metrics (moved verbatim from the legacy miner).
# ==========================================================================================
def png_roundtrip(image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    return decode_image_b64(encode_image_b64(image_chw)).to(device)


def compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
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


def compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def estimate_validator_score(norm: float, rmse: float, synapse_epsilon: float) -> float:
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
# CW margin (scalar + batched) and loss gradients.
# ==========================================================================================
def logits_of(model, x_chw: torch.Tensor) -> torch.Tensor:
    """Single-image forward -> logits[0] (no grad). One choke point for the engine's probes."""
    with torch.no_grad():
        return logits_for_images(model=model, image_bchw=x_chw.unsqueeze(0))[0]


def cw_margin(logits: torch.Tensor, target_index: int) -> float:
    """CW margin m = logit[true] - max_{j!=true} logit[j]; m < 0 means flipped."""
    others = logits.clone()
    others[target_index] = float("-inf")
    return float((logits[target_index] - others.max()).item())


def cw_margin_batch(logits_bxc: torch.Tensor, target_index: int) -> torch.Tensor:
    """Vectorized CW margin over a [B, C] logit batch -> [B] margins."""
    others = logits_bxc.clone()
    others[:, target_index] = float("-inf")
    return logits_bxc[:, target_index] - others.max(dim=1).values


def top_wrong_classes(logits: torch.Tensor, target_index: int, m: int) -> list[int]:
    """The m highest-scoring wrong classes (the easiest competitors to push the true class below)."""
    others = logits.clone()
    others[target_index] = float("-inf")
    m = max(1, min(int(m), logits.numel() - 1))
    return torch.topk(others, m).indices.tolist()


def margin_and_grad(model, x_chw: torch.Tensor, target_index: int):
    """Hard CW margin and its input gradient (boundary direction ∇ℓ_true - ∇max_other)."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = logits[target_index] - others.max()
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def loss_grad(model, x_chw: torch.Tensor, target_index: int, kind: str,
              top_wrong: list[int] | None = None, tau: float = 1.0):
    """Gradient of one of the attack losses, plus the byte-move direction that reduces
    true-class confidence and the per-channel saliency score |g|.

    kind:
      "hard"     -> z_t - max_{j!=t} z_j          (move_dir = -sign(g))
      "ce"       -> cross-entropy at true label   (move_dir = +sign(g), i.e. maximize CE)
      "soft"     -> z_t - tau*logsumexp(top_wrong / tau)   (move_dir = -sign(g))
      "pair:<j>" -> z_t - z_j                      (move_dir = -sign(g))

    Returns (value: float, move_dir_flat: Tensor, score_flat: Tensor).
    """
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]

    if kind == "ce":
        value_t = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index], device=logits.device))
        ascend = True  # maximize CE -> move with +sign(g)
    elif kind == "hard":
        others = logits.clone()
        others[target_index] = float("-inf")
        value_t = logits[target_index] - others.max()
        ascend = False
    elif kind == "soft":
        idx = torch.tensor(top_wrong, device=logits.device)
        soft_wrong = tau * torch.logsumexp(logits[idx] / tau, dim=0)
        value_t = logits[target_index] - soft_wrong
        ascend = False
    elif kind.startswith("pair:"):
        j = int(kind.split(":", 1)[1])
        value_t = logits[target_index] - logits[j]
        ascend = False
    elif kind == "dlr":
        # DLR-normalized margin: (z_y - max_other) / (z_π1 - z_π3 + eps). Scale-invariant, so it does
        # not get dominated by raw logit magnitude on high-margin images. Lower => more flipped.
        others = logits.clone()
        others[target_index] = float("-inf")
        sorted_logits, _ = torch.sort(logits, descending=True)
        denom = sorted_logits[0] - sorted_logits[2] + 1e-12
        value_t = (logits[target_index] - others.max()) / denom
        ascend = False
    elif kind.startswith("dlrt:"):
        # Targeted DLR: push the true class below a SPECIFIC target class t, normalized (scale-free).
        # value = (z_y - z_t) / (z_π1 - (z_π3+z_π4)/2); lower => closer to flipping into t.
        j = int(kind.split(":", 1)[1])
        sorted_logits, _ = torch.sort(logits, descending=True)
        denom = sorted_logits[0] - 0.5 * (sorted_logits[2] + sorted_logits[3]) + 1e-12
        value_t = (logits[target_index] - logits[j]) / denom
        ascend = False
    else:
        raise ValueError(f"unknown loss kind: {kind}")

    grad = torch.autograd.grad(value_t, x)[0].detach().view(-1)
    move_dir = grad.sign() if ascend else -grad.sign()
    return float(value_t.item()), move_dir, grad.abs()


# ==========================================================================================
# TF32 envelope (transfer safety): worst-case margin over the two TF32 regimes.
# ==========================================================================================
def margins_with_tf32(model, batch_bchw: torch.Tensor, target_index: int, cudnn_enabled: bool) -> torch.Tensor:
    """Batched CW margins with cuDNN TF32 forced to `cudnn_enabled`, then restored (CUDA only).
    matmul TF32 is left untouched (kept off, matching the validator default) — the validator only ever
    varies the cuDNN convolution regime, so that is the only axis worth bracketing."""
    prev_cd = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_enabled
    try:
        with torch.no_grad():
            logits = logits_for_images(model=model, image_bchw=batch_bchw)
            return cw_margin_batch(logits, target_index)
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cd


# ==========================================================================================
# Byte-space candidate construction.
# ==========================================================================================
def movable(clean_flat: torch.Tensor, move_dir_flat: torch.Tensor) -> torch.Tensor:
    """Channels whose move_dir step would NOT clip at the [0,1] box edge."""
    return (
        ((move_dir_flat > 0) & (clean_flat < 1.0)) |
        ((move_dir_flat < 0) & (clean_flat > 0.0))
    )


def build_sparse_order(score_flat: torch.Tensor, move_dir_flat: torch.Tensor, clean_flat: torch.Tensor):
    """Descending-saliency channel order, masking signs that would clip at the box edge.
    Returns (masked_score, order, valid_count)."""
    can_move = movable(clean_flat, move_dir_flat)
    score = score_flat.clone()
    score[~can_move] = 0.0
    order = torch.argsort(score, descending=True)
    valid_count = int((score > 0).sum().item())
    return score, order, valid_count


def estimate_k(margin_plus_kappa: float, sorted_score: torch.Tensor, q: float) -> int:
    """Linear seed k* = min{k : q·Σ_{i<=k} score_i >= margin+kappa} along a descending order."""
    if sorted_score.numel() == 0:
        return 1
    cs = torch.cumsum(sorted_score, dim=0)
    need = max(margin_plus_kappa, 1e-9) / max(q, 1e-12)
    k = int((cs < need).sum().item()) + 1
    return max(1, min(k, sorted_score.numel()))


def apply_byte(clean_u8: torch.Tensor, move_dir_flat: torch.Tensor, mask: torch.Tensor,
               k_min: int, shape) -> torch.Tensor:
    """Move each masked channel by k_min bytes along move_dir; clamp [0,255]; return chw float.
    `mask` is either a bool mask or a LongTensor of selected flat indices."""
    steps = torch.zeros_like(clean_u8)
    steps[mask] = float(k_min) * move_dir_flat[mask]
    cand_u8 = (clean_u8 + steps).clamp(0.0, 255.0)
    return (cand_u8 / 255.0).view(shape)


def apply_delta_bytes(clean_u8: torch.Tensor, delta_bytes_flat: torch.Tensor, shape) -> torch.Tensor:
    """Add an arbitrary integer byte delta (a ternary ±k_min support); clamp [0,255]; return chw float."""
    cand_u8 = (clean_u8 + delta_bytes_flat).clamp(0.0, 255.0)
    return (cand_u8 / 255.0).view(shape)


# ==========================================================================================
# Context + Bank + batched evaluator.
# ==========================================================================================
@dataclass
class Context:
    model: torch.nn.Module
    device: torch.device
    clean: torch.Tensor          # chw float in [0,1]
    clean_u8: torch.Tensor       # flat, round(clean*255)
    shape: torch.Size            # clean.shape (C,H,W)
    target_index: int
    k_min: int
    q: float
    floor: float                 # L∞ band lower
    cap: float                   # L∞ band upper
    kappa: float
    skip_roundtrip: bool
    tf32_on: bool                # ambient cuDNN-TF32 regime for forward + backward (matches the validator)
    envelope: bool               # worst-case over the cuDNN-TF32 regime and its opposite (CUDA only)
    allow_unsafe: bool
    deadline: float
    t_step: float                # one fwd+bwd time, for budget gating
    time_left: Callable[[], float]
    bank: "Bank"
    m0: float = 0.0              # clean CW margin
    g0: torch.Tensor | None = None  # clean hard-margin gradient (flat)
    t_eval: float = 0.0          # live EMA of one full-batch eval chunk cost (set by batch_eval)


def out_of_budget(ctx: "Context") -> bool:
    """Deadline gate accounting for BOTH a backward pass (t_step) and a real eval chunk (t_eval). t_eval
    starts at 0 (== the legacy 2·t_step gate) and grows once batch_eval has measured a chunk, so loops stop
    while there is still time to finish the work they are about to start."""
    return ctx.time_left() <= 2.0 * ctx.t_step + K.OOT_EVAL_MARGIN * ctx.t_eval


def _better(a: dict, b: dict) -> bool:
    """Is candidate a strictly sparser/closer than b? Order by (|S|, linf, rmse)."""
    return (a["nz"], a["linf"], a["rmse"]) < (b["nz"], b["linf"], b["rmse"])


class Bank:
    """Tracks the best flipping candidate and the best envelope-safe candidate."""

    def __init__(self) -> None:
        self.best_flip: dict | None = None   # margin < 0, quality ok
        self.best_safe: dict | None = None   # margin <= -kappa, quality ok

    def consider(self, results: list[dict]) -> bool:
        """Fold a batch of results in; return True if a safe candidate now exists."""
        for r in results:
            if not r["quality"]:
                continue
            if self.best_flip is None or _better(r, self.best_flip):
                self.best_flip = r
            if r["safe"] and (self.best_safe is None or _better(r, self.best_safe)):
                self.best_safe = r
        return self.best_safe is not None

    @property
    def has_safe(self) -> bool:
        return self.best_safe is not None

    @property
    def has_flip(self) -> bool:
        return self.best_flip is not None

    def result(self, allow_unsafe: bool) -> dict | None:
        if self.best_safe is not None:
            return self.best_safe
        if allow_unsafe:
            return self.best_flip
        return None


def _forward_margins(ctx: Context, batch_bchw: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = logits_for_images(model=ctx.model, image_bchw=batch_bchw)
        return cw_margin_batch(logits, ctx.target_index)


def batch_eval(ctx: Context, cand_list: list[torch.Tensor]) -> list[dict]:
    """Grade candidates on the validator-faithful path with a single (batched) forward.

    Returns a result dict per candidate with: cand, nz (|S| changed channels), linf, rmse,
    margin (envelope worst-case when enabled), flipped, safe, quality.
    Envelope: for flipped candidates, a second batched TF32-on forward yields max(off,on).
    OOM-safe: halves the batch and retries on CUDA OOM.

    Deadline-aware: tracks an EMA of per-chunk wall time in ctx.t_eval and STOPS launching new chunks once
    time_left <= EVAL_TIME_MARGIN·t_eval, returning the results graded so far (always >=1 chunk). This is
    what keeps a large candidate list from running many forward batches past the deadline. Callers iterate
    or zip over the returned list, so a short (partial) result is safe everywhere.
    """
    if not cand_list:
        return []
    results: list[dict] = []
    bs = max(1, K.BATCH_SIZE)
    i = 0
    half_q = 0.5 * ctx.q
    while i < len(cand_list):
        # Once a chunk cost is known, don't start a chunk we cannot finish before the deadline.
        if results and ctx.t_eval > 0.0 and ctx.time_left() <= K.EVAL_TIME_MARGIN * ctx.t_eval:
            break
        chunk = cand_list[i:i + bs]
        t0 = time.monotonic()
        try:
            seen_list = [c if ctx.skip_roundtrip else png_roundtrip(c, ctx.device) for c in chunk]
            batch = torch.stack(seen_list, dim=0).to(ctx.device)
            margins = _forward_margins(ctx, batch)  # ambient regime (= ctx.tf32_on)
            # Envelope: recompute the flipped subset under the OPPOSITE TF32 regime, keep the worst case,
            # so an accepted flip holds under both TF32 on and off (pre-gated on the flipped subset).
            if ctx.envelope and ctx.device.type == "cuda":
                flipped_idx = (margins < 0.0).nonzero(as_tuple=True)[0]
                if flipped_idx.numel() > 0:
                    alt = margins_with_tf32(ctx.model, batch[flipped_idx], ctx.target_index, not ctx.tf32_on)
                    margins[flipped_idx] = torch.maximum(margins[flipped_idx], alt)
            for j, seen in enumerate(seen_list):
                diff = seen - ctx.clean
                linf = float(diff.abs().max().item())
                rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
                nz = int((diff.abs() > half_q).sum().item())
                margin = float(margins[j].item())
                flipped = margin < 0.0
                in_band = ctx.floor <= linf <= ctx.cap
                quality = False
                if flipped and in_band:
                    ssim = compute_ssim(ctx.clean, seen)
                    psnr = compute_psnr_db(ctx.clean, seen)
                    quality = ssim >= K.MIN_SSIM and psnr >= K.MIN_PSNR_DB
                results.append({
                    "cand": chunk[j], "nz": nz, "linf": linf, "rmse": rmse,
                    "margin": margin, "flipped": flipped,
                    "safe": flipped and margin <= -ctx.kappa, "quality": quality,
                })
            i += bs
        except torch.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            continue  # retry this chunk at a smaller batch; don't fold OOM time into the cost EMA
        # EMA of the cost of a FULL bs-sized chunk (normalize a short tail chunk up to bs).
        chunk_cost = (time.monotonic() - t0) * (bs / max(1, len(chunk)))
        ctx.t_eval = chunk_cost if ctx.t_eval <= 0.0 else 0.6 * ctx.t_eval + 0.4 * chunk_cost
    return results
