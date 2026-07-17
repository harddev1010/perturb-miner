"""Tunable constants for the perturb attack engine (DEV duplicate).

Every knob is an env var (PERTURB_*) so behavior is switchable at runtime without a redeploy.

This is the CLEANED dev constants surface. It keeps: (a) the primitives utils.py + calibration.py
still depend on (accept-gate / kappa / eval / quality knobs), and (b) the knobs for the current dev
engine — a one-shot FGSM candidate refined by a binary-search linear flip finder (see perturb.py).
Everything specific to the old multi-phase pipeline has been dropped.
"""

from __future__ import annotations

import os

import torch


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Core -------------------------------------------------------------------------------
Q = 1.0 / 255.0                                            # one byte in [0,1] space
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)  # validator L∞ cap
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
RESERVE_SECONDS = _env_float("PERTURB_MINER_RESERVE_SECONDS", 2)   # deadline headroom (base)
# Reserve scales with the measured per-forward cost so larger images / models leave enough post-search
# headroom for serialization + verification (added on top of RESERVE_SECONDS). t_step is one fwd+bwd.
RESERVE_FWD_MULT = _env_float("PERTURB_RESERVE_FWD_MULT", 4.0)
# Deadline gating for the batched evaluator: batch_eval stops launching new chunks once
# time_left <= EVAL_TIME_MARGIN·t_eval, and out_of_budget accounts for a backward + an eval chunk.
OOT_EVAL_MARGIN = _env_float("PERTURB_OOT_EVAL_MARGIN", 1.5)
EVAL_TIME_MARGIN = _env_float("PERTURB_EVAL_TIME_MARGIN", 1.25)
# ±1/255 edits on a grid-aligned clean image survive PNG exactly, so the round-trip is identity.
SKIP_ROUNDTRIP = _env_bool("PERTURB_SKIP_ROUNDTRIP", True)
IGNORE_TIMEOUT = _env_bool("PERTURB_IGNORE_TIMEOUT", False)  # ignore the deadline / budget guards
# Candidates evaluated per forward pass (halved on OOM).
BATCH_SIZE = _env_int("PERTURB_BATCH_SIZE", 32)

# --- Accept gate (transfer safety) ------------------------------------------------------
# kappa: require margin <= -kappa. With the TF32 envelope on, the dominant drift axis is
# covered by construction, so kappa drops to the residual cushion.
MARGIN_BUFFER = _env_float("PERTURB_MINER_MARGIN_BUFFER", 0.01)
KAPPA_RESID = _env_float("PERTURB_KAPPA_RESID", 0.004)   # cold-start kappa for the envelope regime
# Dynamic kappa (calibration.py): under the TF32 envelope, learn kappa online from the observed residual
# between the fast proxy margin and the exact validator-faithful margin (PNG round-trip + worst TF32).
DYNAMIC_KAPPA = _env_bool("PERTURB_DYNAMIC_KAPPA", True)
KAPPA_FLOOR = _env_float("PERTURB_KAPPA_FLOOR", 0.0005)   # smallest kappa once residuals look stable
KAPPA_CEILING = _env_float("PERTURB_KAPPA_CEILING", 0.02)  # cap against pathological residual spikes
KAPPA_QUANTILE = _env_float("PERTURB_KAPPA_QUANTILE", 0.99)  # residual quantile (20+ samples)
KAPPA_CUSHION = _env_float("PERTURB_KAPPA_CUSHION", 0.0002)  # tiny numerical cushion added to the quantile
KAPPA_SAMPLES = _env_int("PERTURB_KAPPA_SAMPLES", 256)   # rolling residual-history length
KAPPA_SPREAD_COEF = _env_float("PERTURB_KAPPA_SPREAD_COEF", 0.5)  # per-candidate TF32-spread weight
KAPPA_STORE = os.getenv("PERTURB_KAPPA_STORE",
                        os.path.join(os.path.expanduser("~"), ".cache", "perturb", "kappa_calib.json"))
# cuDNN-TF32 regime for the TF32-sensitive ops (forward + backward). Default: on for CUDA (match the
# validator), off for CPU (TF32 does not exist there).
TF32_ON = _env_bool("PERTURB_TF32_ON", torch.cuda.is_available())
# Require the flip under BOTH cuDNN-TF32 regimes (CUDA only): the ambient regime and its opposite.
TF32_ENVELOPE = _env_bool("PERTURB_TF32_ENVELOPE", True)
# 0 -> return only envelope-safe flips (else clean); 1 -> return any margin<0 flip (literal spec).
ALLOW_UNSAFE_FLIP = _env_bool("PERTURB_ALLOW_UNSAFE_FLIP", False)

# --- One-shot + binary-search linear flip finder (the dev engine) ------------------------
# The engine: step every movable channel along -sign(∇margin) to the full L∞ cap (a one-shot FGSM
# candidate). If that flips, binary-search the scalar step size t in [0,1] along that fixed direction
# for the SMALLEST t that still flips (lower t -> smaller L∞/RMSE -> better perturbation score). All
# probed candidates fold through the score-ranked Bank, so the returned flip is the best-scoring one
# seen. If the full one-shot step does not flip, return the clean image.
LINE_GRID = _env_int("PERTURB_LINE_GRID", 8)          # initial linear t-grid probes across (0, 1]
BISECT_ITERS = _env_int("PERTURB_BISECT_ITERS", 12)   # binary-search refinement steps on the flip boundary
