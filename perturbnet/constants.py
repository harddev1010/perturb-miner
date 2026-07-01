from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default

# Shared subnet identity/constants.
SUBNET_NAMESPACE = "perturb"
MODEL_NAME = "EfficientNetV2-L"

# Validator runtime state files.
VALIDATOR_STATE_FILENAME = "perturb_validator_state.json"

# Validator runtime constants.
# Challenge dataset is fixed: full ImageNet-100 train split from Hugging Face,
# auto-downloaded once into the local cache.
IMAGENET100_REPO_ID = "clane9/imagenet-100"
IMAGENET100_SPLIT = "train"
# If challenge generation fails (e.g. transient dataset issue), sleep this
# long and try again; there is no fallback image.
CHALLENGE_RETRY_DELAY_SECONDS = _env_int("PERTURB_CHALLENGE_RETRY_DELAY_SECONDS", 180)
IMAGE_SIZE = _env_int("PERTURB_IMAGE_SIZE", 64)
TIMEOUT_SECONDS = _env_int("PERTURB_TIMEOUT_SECONDS", 20)
QUERY_INTERVAL_SECONDS = _env_int("PERTURB_QUERY_INTERVAL_SECONDS", 20)
K_MINERS = _env_int("PERTURB_K_MINERS", 150)
HISTORY_SIZE = _env_int("PERTURB_HISTORY_SIZE", 300)
MIN_LINF_DELTA = _env_float("PERTURB_MIN_LINF_DELTA", 0.003)
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
LINF_COMPONENT_WEIGHT = _env_float("PERTURB_LINF_COMPONENT_WEIGHT", 0.7)
RMSE_COMPONENT_WEIGHT = _env_float("PERTURB_RMSE_COMPONENT_WEIGHT", 0.3)
MAX_CHALLENGE_ATTEMPTS = _env_int("PERTURB_MAX_CHALLENGE_ATTEMPTS", 12)
MINER_EXPLORATION_RATIO = _env_float("PERTURB_MINER_EXPLORATION_RATIO", 0.20)
ANALYZE_BUCKET_MARGIN_WEIGHT = _env_float("ANALYZE_BUCKET_MARGIN_WEIGHT", 0.03)
ANALYZE_BUCKET_NOVELTY_WEIGHT = _env_float("ANALYZE_BUCKET_NOVELTY_WEIGHT", 0.01)
ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS = _env_int("ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS", 8)
STORAGE_MODE = os.getenv("PERTURB_STORAGE_MODE", "latest").strip().lower() or "latest"
BURN_RATE_ENDPOINT = "https://api.perturbai.io/api/v1/burn-rate"
DEFAULT_BURN_RATE = 0.0
BURN_RATE_FETCH_TIMEOUT_SECONDS = 5.0
BURN_UID = 0
LEADERBOARD_REPORTING_ENABLED = True
LEADERBOARD_API_URL = "https://api.perturbai.io/api/v1/report"
LEADERBOARD_REPORT_TIMEOUT_SECONDS = 10.0
LEADERBOARD_NO_IMAGE_URL = "https://placeimg.dev/128x128/F3F4F6?text=None&textColor=6B7280"
R2_EXPORT_ENABLED = True
R2_PREFIX = "adversarial-dataset"
R2_PRESIGNED_URL_EXPIRES_SECONDS = 604800

VALIDATOR_CONFIG = {
    "imagenet100_repo_id": IMAGENET100_REPO_ID,
    "imagenet100_split": IMAGENET100_SPLIT,
    "image_size": IMAGE_SIZE,
    "timeout_seconds": TIMEOUT_SECONDS,
    "query_interval_seconds": QUERY_INTERVAL_SECONDS,
    "k_miners": K_MINERS,
    "history_size": HISTORY_SIZE,
    "min_linf_delta": MIN_LINF_DELTA,
    "max_linf_delta": MAX_LINF_DELTA,
    "min_ssim": MIN_SSIM,
    "min_psnr_db": MIN_PSNR_DB,
    "linf_component_weight": LINF_COMPONENT_WEIGHT,
    "rmse_component_weight": RMSE_COMPONENT_WEIGHT,
    "max_challenge_attempts": MAX_CHALLENGE_ATTEMPTS,
    "miner_exploration_ratio": MINER_EXPLORATION_RATIO,
    "analyze_bucket_margin_weight": ANALYZE_BUCKET_MARGIN_WEIGHT,
    "analyze_bucket_novelty_weight": ANALYZE_BUCKET_NOVELTY_WEIGHT,
    "analyze_bucket_novelty_target_pixels": ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS,
    "storage_mode": STORAGE_MODE,
    "burn_rate_endpoint": BURN_RATE_ENDPOINT,
    "default_burn_rate": DEFAULT_BURN_RATE,
    "burn_rate_fetch_timeout_seconds": BURN_RATE_FETCH_TIMEOUT_SECONDS,
    "burn_uid": BURN_UID,
    "leaderboard_reporting_enabled": LEADERBOARD_REPORTING_ENABLED,
    "leaderboard_api_url": LEADERBOARD_API_URL,
    "leaderboard_report_timeout_seconds": LEADERBOARD_REPORT_TIMEOUT_SECONDS,
    "leaderboard_no_image_url": LEADERBOARD_NO_IMAGE_URL,
    "r2_export_enabled": R2_EXPORT_ENABLED,
    "r2_prefix": R2_PREFIX,
    "r2_presigned_url_expires_seconds": R2_PRESIGNED_URL_EXPIRES_SECONDS,
}

# Validator scoring defaults.
SPEED_WEIGHT = _env_float("PERTURB_SPEED_WEIGHT", 0)
PERTURBATION_WEIGHT = _env_float("PERTURB_PERTURBATION_WEIGHT", 1)
GAMMA_HISTORY_WEIGHT = _env_float("PERTURB_GAMMA_HISTORY_WEIGHT", 0.7)

