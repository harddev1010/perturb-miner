"""Online residual calibration for the transfer-safety cushion kappa.

The accept gate requires a flip's worst-case (TF32-envelope) margin <= -kappa. A FIXED kappa is either
too loose (rejects valid flips on stable hardware) or too tight (after a library / GPU / preprocessing
change). This module learns kappa online from the DANGEROUS RESIDUAL between the fast proxy margin used
during search and the exact validator-faithful margin:

    d = m_exact - m_proxy          (positive d = the exact path moved the candidate back toward true)

kappa is a high quantile of observed residuals plus a small cushion, floored by a cold-start default and
clamped to a ceiling. Because the search already keeps the worst of the two TF32 regimes (the envelope),
the residual represents only the drift that is left over — serialization, repeatability, library/GPU
differences. History is persisted per ENVIRONMENT FINGERPRINT (model, GPU, torch / CUDA / cuDNN versions,
preprocessing, image shape, TF32 settings), so later attacks start well-calibrated.

LIMITATION: the locally-measurable exact path (PNG round-trip + worst TF32) captures serialization and
numeric drift, not cross-machine validator drift. The cold-start default and the floor guard the part we
cannot observe locally; raise PERTURB_KAPPA_FLOOR to be more conservative.

The calibration target is exactly m_exact - m_proxy — NOT clean margin, gradient magnitude, support size,
or the candidate's own negative margin, which describe adversarial geometry rather than validator drift.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import deque

import torch

from . import constants as K

logger = logging.getLogger(__name__)


def candidate_kappa_vec(base: float, spread: torch.Tensor, coef: float,
                        floor: float, ceiling: float) -> torch.Tensor:
    """Per-candidate accept cushion: max(global kappa, coef·TF32-spread), clamped to [floor, ceiling].

    The envelope already takes the worst TF32 regime, so the on/off spread is a SECONDARY instability
    signal — a numerically unstable flip (large spread) must clear a deeper margin to be called safe."""
    base_t = torch.full_like(spread, float(base))
    return torch.clamp(torch.maximum(base_t, float(coef) * spread), float(floor), float(ceiling))


class MarginSafetyCalibrator:
    """Rolling residual history -> a high-quantile kappa, persisted per environment fingerprint."""

    def __init__(self, fingerprint: str, store_path: str, max_samples: int, cold_kappa: float,
                 floor: float, ceiling: float, quantile: float, cushion: float) -> None:
        self.fingerprint = fingerprint
        self.store_path = store_path
        self.cold_kappa = cold_kappa
        self.floor = floor
        self.ceiling = ceiling
        self.quantile = quantile
        self.cushion = cushion
        self.residuals: deque[float] = deque(maxlen=max(8, int(max_samples)))

    # --- learning ------------------------------------------------------------------------
    def update(self, proxy_margin: float, exact_margin: float) -> None:
        """Record one dangerous residual d = max(0, m_exact - m_proxy). Only positive drift (the exact
        path pulling back toward the true class) can invalidate a flip, so negative drift is clamped to 0."""
        residual = exact_margin - proxy_margin
        if math.isfinite(residual):
            self.residuals.append(max(0.0, float(residual)))

    def global_kappa(self) -> float:
        """Cold-start tiers: <5 samples -> conservative default; 5–19 -> observed max + cushion;
        20+ -> high empirical quantile + cushion. Always clamped to [floor, ceiling]."""
        n = len(self.residuals)
        if n < 5:
            value = self.cold_kappa
        elif n < 20:
            value = max(self.residuals) + self.cushion
        else:
            vals = torch.tensor(list(self.residuals), dtype=torch.float32)
            value = float(torch.quantile(vals, self.quantile).item()) + self.cushion
        return min(self.ceiling, max(self.floor, value))

    @property
    def n_samples(self) -> int:
        return len(self.residuals)

    # --- persistence (best-effort; never raises) -----------------------------------------
    def load(self) -> None:
        try:
            if not self.store_path or not os.path.exists(self.store_path):
                return
            with open(self.store_path, encoding="utf-8") as f:
                store = json.load(f)
            for d in store.get(self.fingerprint, []):
                if isinstance(d, (int, float)) and math.isfinite(d):
                    self.residuals.append(float(d))
            logger.info(f"[kappa] loaded {len(self.residuals)} residuals for fp={self.fingerprint[:48]}…")
        except Exception as err:
            logger.debug(f"[kappa] load skipped: {err}")

    def save(self) -> None:
        try:
            if not self.store_path:
                return
            os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
            store = {}
            if os.path.exists(self.store_path):
                try:
                    with open(self.store_path, encoding="utf-8") as f:
                        store = json.load(f)
                except Exception:
                    store = {}  # corrupt/partial file: start fresh rather than fail
            store[self.fingerprint] = [round(d, 8) for d in self.residuals]
            d = os.path.dirname(self.store_path) or "."
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f)
            os.replace(tmp, self.store_path)  # atomic: never leaves a half-written store
        except Exception as err:
            logger.debug(f"[kappa] save skipped: {err}")


# ==========================================================================================
# Process registry + environment fingerprint.
# ==========================================================================================
_REGISTRY: dict[str, MarginSafetyCalibrator] = {}
_FP_CACHE: dict[int, str] = {}


def get_calibrator(fingerprint: str) -> MarginSafetyCalibrator:
    """Process-cached calibrator for a fingerprint (loaded from disk once on first use)."""
    c = _REGISTRY.get(fingerprint)
    if c is None:
        c = MarginSafetyCalibrator(
            fingerprint=fingerprint, store_path=K.KAPPA_STORE, max_samples=K.KAPPA_SAMPLES,
            cold_kappa=K.KAPPA_RESID, floor=K.KAPPA_FLOOR, ceiling=K.KAPPA_CEILING,
            quantile=K.KAPPA_QUANTILE, cushion=K.KAPPA_CUSHION,
        )
        c.load()
        _REGISTRY[fingerprint] = c
    return c


def env_fingerprint(model, shape) -> str:
    """A stable key for the calibration environment: anything whose change can shift the validator's
    margin (model, GPU, torch / CUDA / cuDNN versions, TF32 regime, image shape) gets its own history."""
    cached = _FP_CACHE.get(id(model)) if model is not None else None
    if cached is not None and shape is None:
        return cached
    try:
        nparam = sum(p.numel() for p in model.parameters()) if model is not None else 0
        cls = type(model).__name__ if model is not None else "none"
    except Exception:
        nparam, cls = 0, "none"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:
        cudnn = 0
    c, h, w = (int(shape[0]), int(shape[1]), int(shape[2])) if shape is not None and len(shape) == 3 else (0, 0, 0)
    fp = (f"{cls}|n{nparam}|t{torch.__version__}|cu{torch.version.cuda}|cd{cudnn}|g{gpu}"
          f"|s{c}x{h}x{w}|tf32{int(K.TF32_ON)}|env{int(K.TF32_ENVELOPE)}")
    if model is not None:
        _FP_CACHE[id(model)] = fp
    return fp
