from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging as pylogging
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import bittensor as bt
import numpy as np
import torch
import torch.nn.functional as F
try:
    import wandb  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import load_efficientnet_v2_l, normalize_prediction_label, predict_label
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)


@dataclass
class ChallengeSpec:
    task_id: str
    model_name: str
    clean_image_b64: str
    true_label: str
    epsilon: float
    norm_type: str
    timeout_seconds: int


@dataclass
class EvaluationResult:
    score: float
    reason: str
    model_prediction: str = ""
    response_time_ms: int = 0
    norm: float = 0.0
    rmse: float = 0.0
    epsilon: float = 0.0
    ssim: float = 0.0
    psnr_db: float = 0.0


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
    if hasattr(bt, "subtensor"):
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_dendrite(wallet):
    if hasattr(bt, "dendrite"):
        return bt.dendrite(wallet=wallet)
    dendrite_cls = getattr(bt, "Dendrite", None)
    if dendrite_cls is None:
        raise RuntimeError("No dendrite constructor found in bittensor.")
    return dendrite_cls(wallet=wallet)


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    if x_clean.ndim != 3 or x_adv.ndim != 3:
        return 0.0
    if x_clean.shape != x_adv.shape:
        return 0.0
    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


class _WandbConsoleHandler(pylogging.Handler):
    def __init__(self, owner: "PerturbValidator") -> None:
        super().__init__()
        self.owner = owner

    def emit(self, record: pylogging.LogRecord) -> None:
        run = self.owner.wandb_run
        if run is None:
            return
        if record.name.startswith("wandb"):
            return
        try:
            run.log(
                {
                    "validator/console_level": record.levelname,
                    "validator/console_logger": record.name,
                    "validator/console_message": self.format(record),
                    "validator/console_ts": float(record.created),
                },
                step=int(self.owner.step),
            )
        except Exception:
            # Never let console-forwarding failures affect validator runtime.
            pass


class PerturbValidator:
    def __init__(self, config: bt.config) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = _make_subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.dendrite = _make_dendrite(wallet=self.wallet)
        self._query_loop = asyncio.new_event_loop()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_random = random.SystemRandom()

        self.model = load_efficientnet_v2_l(self.device)
        self.step = 0
        self.last_weight_block = 0
        self.state_path = os.path.join(self.config.logging.logging_dir, C.VALIDATOR_STATE_FILENAME)
        self.wandb_run: Any | None = None
        self._wandb_console_handler: pylogging.Handler | None = None
        hotkey = getattr(self.wallet.hotkey, "ss58_address", "unknown")
        self.run_id = f"{str(hotkey)[:8]}-n{self.config.netuid}-p{os.getpid()}"
        self.reason_counts_total: Counter[str] = Counter()
        self.miner_emission_share = 1
        self.imagenet100_order_seed = 0
        self.imagenet100_order_cursor = 0
        self.imagenet100_order_fingerprint = ""
        self.imagenet100_order_epoch = 0
        self._imagenet100_index: list[tuple[str, int]] = []
        self._imagenet100_dataset: Any | None = None
        self._imagenet100_order_cache: list[str] = []
        self._imagenet100_order_cache_key: tuple[str, int] = ("", 0)

        self.processed_counts = np.zeros(int(self.metagraph.n), dtype=np.int32)
        self.score_histories: list[list[float]] = [[] for _ in range(int(self.metagraph.n))]
        self.uid_hotkeys: list[str] = list(self.metagraph.hotkeys[: int(self.metagraph.n)])

        self._load_state()
        self._init_wandb()

    def _log_step_start(self, step_name: str, **context: Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.debug(f"{step_name} {rendered}")
        else:
            logger.debug(step_name)

    def _log_summary(self, event: str, **context: Any) -> None:
        if context:
            rendered = " ".join([f"{k}={context[k]}" for k in sorted(context.keys())])
            logger.info(f"[run_id={self.run_id}] {event} {rendered}")
        else:
            logger.info(f"[run_id={self.run_id}] {event}")

    def sync(self) -> None:
        old_n = int(self.metagraph.n)
        self.metagraph.sync(subtensor=self.subtensor)
        new_n = int(self.metagraph.n)
        if new_n != old_n:
            resized_counts = np.zeros(new_n, dtype=np.int32)
            copied = min(len(self.processed_counts), new_n)
            resized_counts[:copied] = self.processed_counts[:copied]
            self.processed_counts = resized_counts
            if new_n > len(self.score_histories):
                self.score_histories.extend([[] for _ in range(new_n - len(self.score_histories))])
            else:
                self.score_histories = self.score_histories[:new_n]
            if new_n > len(self.uid_hotkeys):
                self.uid_hotkeys.extend([""] * (new_n - len(self.uid_hotkeys)))
            else:
                self.uid_hotkeys = self.uid_hotkeys[:new_n]
        self._reconcile_uid_identities()

    def _reset_uid_stats(self, uid: int, reason: str) -> None:
        self.processed_counts[uid] = 0
        self.score_histories[uid] = []
        logger.info(f"Reset uid={uid} stats due to {reason}.")

    def _reconcile_uid_identities(self) -> None:
        n = int(self.metagraph.n)
        if len(self.uid_hotkeys) < n:
            self.uid_hotkeys.extend([""] * (n - len(self.uid_hotkeys)))
        elif len(self.uid_hotkeys) > n:
            self.uid_hotkeys = self.uid_hotkeys[:n]

        for uid in range(n):
            current_hotkey = str(self.metagraph.hotkeys[uid])
            previous_hotkey = self.uid_hotkeys[uid]
            if previous_hotkey and previous_hotkey != current_hotkey:
                self._reset_uid_stats(uid, reason="hotkey_changed")
            self.uid_hotkeys[uid] = current_hotkey

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception as exc:
            # Saves are atomic, so this should never happen; never brick startup on it.
            logger.error(f"Validator state file is unreadable, starting fresh: {self.state_path} ({exc})")
            return
        self.step = int(state.get("step", 0))
        self.last_weight_block = int(state.get("last_weight_block", 0))

        saved_counts = state.get("processed_counts", [])
        copied = min(len(saved_counts), len(self.processed_counts))
        for idx in range(copied):
            self.processed_counts[idx] = int(saved_counts[idx])

        saved_histories = state.get("score_histories", [])
        copied_h = min(len(saved_histories), len(self.score_histories))
        for idx in range(copied_h):
            raw = saved_histories[idx]
            if isinstance(raw, list):
                self.score_histories[idx] = [float(x) for x in raw[-self.config.perturb.history_size :]]

        saved_hotkeys = state.get("uid_hotkeys", [])
        if isinstance(saved_hotkeys, list):
            copied_keys = min(len(saved_hotkeys), len(self.uid_hotkeys))
            for idx in range(copied_keys):
                value = saved_hotkeys[idx]
                if isinstance(value, str):
                    self.uid_hotkeys[idx] = value
        self.imagenet100_order_seed = max(0, int(state.get("imagenet100_order_seed", 0)))
        self.imagenet100_order_cursor = max(0, int(state.get("imagenet100_order_cursor", 0)))
        self.imagenet100_order_fingerprint = str(state.get("imagenet100_order_fingerprint", "") or "")
        self.imagenet100_order_epoch = max(0, int(state.get("imagenet100_order_epoch", 0)))
        self._reconcile_uid_identities()

    def _save_state(self) -> None:
        directory = os.path.dirname(self.state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "step": int(self.step),
            "last_weight_block": int(self.last_weight_block),
            "processed_counts": self.processed_counts.tolist(),
            "score_histories": [history[-self.config.perturb.history_size :] for history in self.score_histories],
            "uid_hotkeys": self.uid_hotkeys,
            "imagenet100_order_seed": int(self.imagenet100_order_seed),
            "imagenet100_order_cursor": int(self.imagenet100_order_cursor),
            "imagenet100_order_fingerprint": self.imagenet100_order_fingerprint,
            "imagenet100_order_epoch": int(self.imagenet100_order_epoch),
        }
        # Atomic write: a crash mid-save must never corrupt the state file,
        # otherwise a restart could reset the traversal and repeat images.
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.state_path)

    def _init_wandb(self) -> None:
        if not bool(getattr(self.config.perturb, "wandb_enabled", False)):
            return
        if wandb is None:
            logger.warning("PERTURB_WANDB_ENABLED is true, but `wandb` is not installed.")
            return
        try:
            init_kwargs: dict[str, Any] = {
                "project": str(getattr(self.config.perturb, "wandb_project", "perturb-validator")),
                "config": {
                    "netuid": int(self.config.netuid),
                    "network": str(getattr(self.config.subtensor, "network", "unknown")),
                    "k_miners": int(self.config.perturb.k_miners),
                    "history_size": int(self.config.perturb.history_size),
                    "min_processed_count": int(self.config.perturb.min_processed_count),
                },
            }
            entity = str(getattr(self.config.perturb, "wandb_entity", "")).strip()
            run_name = str(getattr(self.config.perturb, "wandb_run_name", "")).strip()
            mode = str(getattr(self.config.perturb, "wandb_mode", "online")).strip() or "online"
            if not run_name:
                uid_suffix = self._resolve_validator_uid_for_run_name()
                run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-uid{uid_suffix}"
            if entity:
                init_kwargs["entity"] = entity
            init_kwargs["name"] = run_name
            init_kwargs["mode"] = mode
            self.wandb_run = wandb.init(**init_kwargs)
            self._attach_wandb_console_handler()
            logger.info("W&B logging initialized for validator.")
        except Exception as exc:
            logger.warning(f"Failed to initialize W&B logging: {exc}")
            self.wandb_run = None

    def _resolve_validator_uid_for_run_name(self) -> str:
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", "") or "")
        try:
            if hotkey and hotkey in self.metagraph.hotkeys:
                return str(self.metagraph.hotkeys.index(hotkey))
        except Exception:
            pass
        return "unknown"

    def _attach_wandb_console_handler(self) -> None:
        if not bool(getattr(self.config.perturb, "wandb_log_console", True)):
            return
        if self._wandb_console_handler is not None:
            return
        handler = _WandbConsoleHandler(self)
        handler.setLevel(pylogging.NOTSET)
        handler.setFormatter(
            pylogging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
        )
        pylogging.getLogger().addHandler(handler)
        self._wandb_console_handler = handler

    def _detach_wandb_console_handler(self) -> None:
        if self._wandb_console_handler is None:
            return
        root_logger = pylogging.getLogger()
        root_logger.removeHandler(self._wandb_console_handler)
        self._wandb_console_handler = None

    def _wandb_log(self, payload: dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(payload, step=int(self.step))
        except Exception as exc:
            logger.warning(f"W&B log failed: {exc}")

    def _finish_wandb(self) -> None:
        self._detach_wandb_console_handler()
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
        except Exception as exc:
            logger.warning(f"W&B finish failed: {exc}")
        finally:
            self.wandb_run = None

    def _seed_from_block(self, block: int) -> int:
        digest = hashlib.sha256(f"{C.SUBNET_NAMESPACE}:{self.config.netuid}:{block}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _sample_epsilon(self, seed: int) -> float:
        # Deterministic epsilon in [0.06, 0.2]
        return 0.06 + (seed % 1400) / 10000.0

    def _load_imagenet100_index(self) -> list[tuple[str, int]]:
        """Open the full ImageNet-100 train split (~126k images) with random
        access; ids map to dataset row indices."""
        if self._imagenet100_index:
            return self._imagenet100_index

        from perturbnet.imagenet100_bootstrap import imagenet100_dataset_version, load_imagenet100

        repo_id = str(getattr(self.config.perturb, "imagenet100_repo_id", C.IMAGENET100_REPO_ID))
        split = str(getattr(self.config.perturb, "imagenet100_split", C.IMAGENET100_SPLIT))
        dataset = load_imagenet100(repo_id=repo_id, split=split)
        version = imagenet100_dataset_version(dataset=dataset, repo_id=repo_id, split=split)
        total_rows = int(dataset.num_rows)
        if total_rows <= 0:
            raise RuntimeError(f"ImageNet-100 dataset is empty: repo={repo_id} split={split}")
        self._imagenet100_dataset = dataset
        self._imagenet100_index = [(f"hf-{version}-{row:07d}", row) for row in range(total_rows)]
        logger.info(f"Loaded ImageNet-100 challenge index images={total_rows} repo={repo_id} split={split}")
        return self._imagenet100_index

    def _imagenet100_fingerprint(self, image_ids: Sequence[str]) -> str:
        digest = hashlib.sha256()
        for image_id in sorted(image_ids):
            digest.update(image_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _derive_imagenet100_order(self, image_ids: Sequence[str], seed: int) -> list[str]:
        # Deterministic shuffle from the persisted seed: restarts rebuild the
        # exact same traversal order without storing ~126k ids in state.
        order = sorted(image_ids)
        random.Random(seed).shuffle(order)
        return order

    def _reset_imagenet100_order(self, fingerprint: str, epoch: int) -> None:
        self.imagenet100_order_seed = self.system_random.randrange(1, 2**63)
        self.imagenet100_order_cursor = 0
        self.imagenet100_order_fingerprint = fingerprint
        self.imagenet100_order_epoch = epoch
        self._imagenet100_order_cache = []
        self._imagenet100_order_cache_key = ("", 0)

    def _ensure_imagenet100_order(self, image_ids: Sequence[str]) -> None:
        fingerprint = self._imagenet100_fingerprint(image_ids)
        if self.imagenet100_order_fingerprint != fingerprint or self.imagenet100_order_seed <= 0:
            self._reset_imagenet100_order(fingerprint=fingerprint, epoch=0)
            logger.info(f"Initialized ImageNet-100 random traversal images={len(image_ids)}")
        elif self.imagenet100_order_cursor >= len(image_ids):
            self._reset_imagenet100_order(fingerprint=fingerprint, epoch=self.imagenet100_order_epoch + 1)
            logger.info(
                f"ImageNet-100 traversal exhausted; reshuffled for epoch={self.imagenet100_order_epoch} "
                f"images={len(image_ids)}"
            )
        cache_key = (self.imagenet100_order_fingerprint, int(self.imagenet100_order_seed))
        if self._imagenet100_order_cache_key != cache_key or not self._imagenet100_order_cache:
            self._imagenet100_order_cache = self._derive_imagenet100_order(
                image_ids=image_ids, seed=int(self.imagenet100_order_seed)
            )
            self._imagenet100_order_cache_key = cache_key

    def _mark_imagenet100_image_used(self, image_id: str) -> None:
        order = self._imagenet100_order_cache
        if self.imagenet100_order_cursor < len(order) and order[self.imagenet100_order_cursor] != image_id:
            logger.warning("Accepted ImageNet-100 image does not match traversal cursor; advancing anyway.")
        self.imagenet100_order_cursor += 1

    def _imagenet100_image_bytes(self, source: int) -> bytes:
        if self._imagenet100_dataset is None:
            raise RuntimeError("ImageNet-100 dataset is not loaded")
        example = self._imagenet100_dataset[int(source)]
        image = example.get("image")
        if image is None:
            raise ValueError(f"ImageNet-100 row {source} has no image payload")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    def _sample_imagenet100_image(self) -> tuple[str, str]:
        index = self._load_imagenet100_index()
        source_by_id = {image_id: source for image_id, source in index}
        image_ids = list(source_by_id.keys())
        self._ensure_imagenet100_order(image_ids=image_ids)
        if not self._imagenet100_order_cache:
            raise RuntimeError("ImageNet-100 traversal order is empty")
        image_id = self._imagenet100_order_cache[self.imagenet100_order_cursor]
        try:
            raw = self._imagenet100_image_bytes(source_by_id[image_id])
            if not raw:
                raise ValueError(f"ImageNet-100 image is empty: {image_id}")
        except Exception:
            # Skip corrupt/unreadable entries permanently in this traversal.
            self._mark_imagenet100_image_used(image_id=image_id)
            self._save_state()
            raise
        return image_id, base64.b64encode(raw).decode("utf-8")

    def generate_challenge(self, block: int) -> ChallengeSpec:
        model_name = C.MODEL_NAME
        base_seed = self._seed_from_block(block)
        self._log_step_start(
            "generate_challenge",
            block=block,
            base_seed=base_seed,
            max_attempts=self.config.perturb.max_challenge_attempts,
        )
        for attempt in range(self.config.perturb.max_challenge_attempts):
            seed = base_seed + attempt
            self._log_summary(
                "challenge_attempt",
                attempt=attempt + 1,
                max_attempts=self.config.perturb.max_challenge_attempts,
                seed=seed,
                source="imagenet100",
            )
            self._log_step_start(
                "challenge_attempt_internal",
                attempt=attempt + 1,
            )
            try:
                self._log_step_start("challenge_sample_imagenet100", seed=seed)
                image_id, image_b64 = self._sample_imagenet100_image()
            except Exception as exc:
                logger.warning(f"ImageNet-100 challenge image sample failed, retrying: {exc}")
                continue

            epsilon = self._sample_epsilon(seed)
            task_id = f"{block}-{seed}"
            self._log_step_start("challenge_prepare", task_id=task_id, epsilon=f"{epsilon:.4f}")

            try:
                self._log_step_start("challenge_model_inference", task_id=task_id)
                image = decode_image_b64(image_b64).to(self.device)
                predicted = predict_label(self.model, image)
                predicted_label = normalize_prediction_label(predicted)
                self._log_summary("challenge_model_output", image_id=image_id, predicted_label=predicted_label)
            except Exception as exc:
                logger.warning(f"Challenge decode/model validation failed, retrying: {exc}")
                self._mark_imagenet100_image_used(image_id=image_id)
                self._save_state()
                continue

            self._mark_imagenet100_image_used(image_id=image_id)
            self._save_state()
            return ChallengeSpec(
                task_id=task_id,
                model_name=model_name,
                clean_image_b64=image_b64,
                # Use exact EfficientNet class label for miner targeting and response verification.
                true_label=predicted_label,
                epsilon=epsilon,
                norm_type="Linf",
                timeout_seconds=self.config.perturb.timeout_seconds,
            )

        raise RuntimeError("Unable to build a validated challenge after max attempts")

    def _available_miner_uids(self) -> list[int]:
        my_hotkey = self.wallet.hotkey.ss58_address
        candidate_uids: list[int] = []
        for uid in range(int(self.metagraph.n)):
            if self.metagraph.hotkeys[uid] == my_hotkey:
                continue
            if self.metagraph.axons[uid].ip == "0.0.0.0":
                continue
            candidate_uids.append(uid)

        if len(candidate_uids) <= 1:
            return candidate_uids

        # Merge miners that likely belong to the same operator:
        # - same coldkey OR same axon ip
        # Keep only the lowest uid representative from each merged group.
        parent: dict[int, int] = {uid: uid for uid in candidate_uids}

        def _find(uid: int) -> int:
            while parent[uid] != uid:
                parent[uid] = parent[parent[uid]]
                uid = parent[uid]
            return uid

        def _union(a: int, b: int) -> None:
            ra = _find(a)
            rb = _find(b)
            if ra == rb:
                return
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        first_uid_by_coldkey: dict[str, int] = {}
        first_uid_by_ip: dict[str, int] = {}
        coldkeys = getattr(self.metagraph, "coldkeys", [])
        for uid in candidate_uids:
            coldkey = ""
            if uid < len(coldkeys):
                coldkey = str(coldkeys[uid] or "").strip()
            if coldkey:
                seen_uid = first_uid_by_coldkey.get(coldkey)
                if seen_uid is None:
                    first_uid_by_coldkey[coldkey] = uid
                else:
                    _union(seen_uid, uid)

            ip = str(getattr(self.metagraph.axons[uid], "ip", "") or "").strip()
            if ip:
                seen_uid = first_uid_by_ip.get(ip)
                if seen_uid is None:
                    first_uid_by_ip[ip] = uid
                else:
                    _union(seen_uid, uid)

        min_uid_by_group: dict[int, int] = {}
        for uid in candidate_uids:
            root = _find(uid)
            current_min = min_uid_by_group.get(root)
            if current_min is None or uid < current_min:
                min_uid_by_group[root] = uid

        return sorted(min_uid_by_group.values())

    def _valuable_miner_uids(self, candidate_uids: Sequence[int]) -> list[int]:
        min_processed = int(self.config.perturb.min_processed_count)
        return [uid for uid in candidate_uids if int(self.processed_counts[uid]) >= min_processed]

    def _select_random_miners(self, candidate_uids: Sequence[int], seed: int) -> list[int]:
        if not candidate_uids:
            return []
        valuable = self._valuable_miner_uids(candidate_uids)
        pool = list(candidate_uids)
        k = min(int(self.config.perturb.k_miners), len(pool))
        rng = random.Random(seed)
        if k <= 0:
            return []
        if not valuable:
            return sorted(rng.sample(pool, k=k))

        valuable_set = set(valuable)
        newcomers = [uid for uid in pool if uid not in valuable_set]
        ratio = float(max(0.0, min(1.0, self.config.perturb.miner_exploration_ratio)))
        explore_k = min(len(newcomers), int(round(k * ratio)))
        if newcomers and ratio > 0.0 and explore_k == 0:
            explore_k = 1
        exploit_k = min(len(valuable), k - explore_k)
        if exploit_k + explore_k < k:
            explore_k = min(len(newcomers), explore_k + (k - (exploit_k + explore_k)))

        selected: list[int] = []
        if exploit_k > 0:
            selected.extend(rng.sample(list(valuable), k=exploit_k))
        if explore_k > 0:
            selected.extend(rng.sample(newcomers, k=explore_k))

        if len(selected) < k:
            remaining = [uid for uid in pool if uid not in set(selected)]
            selected.extend(rng.sample(remaining, k=min(k - len(selected), len(remaining))))
        return sorted(selected)

    async def _query_miners(self, uids: Sequence[int], challenge: ChallengeSpec):
        self._log_step_start(
            "query_miners",
            task_id=challenge.task_id,
            miner_count=len(uids),
            timeout=challenge.timeout_seconds,
        )
        axons = [self.metagraph.axons[uid] for uid in uids]
        synapse = AttackChallenge(
            task_id=challenge.task_id,
            model_name=challenge.model_name,
            clean_image_b64=challenge.clean_image_b64,
            true_label=challenge.true_label,
            epsilon=challenge.epsilon,
            norm_type=challenge.norm_type,
            min_delta=self.config.perturb.min_linf_delta,
            timeout_seconds=challenge.timeout_seconds,
        )
        responses = await self.dendrite.forward(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=challenge.timeout_seconds,
        )
        return responses

    def _run_query_miners(self, uids: Sequence[int], challenge: ChallengeSpec):
        # Keep a persistent event loop for dendrite calls; asyncio.run() closes
        # the loop each call and can trigger "Event loop is closed" on reuse.
        if self._query_loop.is_closed():
            self._query_loop = asyncio.new_event_loop()
        return self._query_loop.run_until_complete(self._query_miners(uids, challenge))

    def verify_and_score(
        self,
        challenge: ChallengeSpec,
        perturbed_image_b64: str,
        response_time_ms: int,
    ) -> EvaluationResult:
        try:
            x_clean = decode_image_b64(challenge.clean_image_b64).to(self.device)
            x_adv = decode_image_b64(perturbed_image_b64).to(self.device)
        except Exception as exc:
            return EvaluationResult(score=0.0, reason=f"decode_failed:{exc}", response_time_ms=response_time_ms)

        if x_adv.shape != x_clean.shape:
            return EvaluationResult(score=0.0, reason="shape_mismatch", response_time_ms=response_time_ms)
        if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
            return EvaluationResult(score=0.0, reason="value_out_of_range", response_time_ms=response_time_ms)

        prediction = ""
        try:
            prediction = predict_label(self.model, x_adv)
        except Exception as exc:
            return EvaluationResult(
                score=0.0,
                reason=f"model_inference_failed:{exc}",
                response_time_ms=response_time_ms,
            )

        if challenge.norm_type == "Linf":
            norm = (x_adv - x_clean).abs().max().item()
        elif challenge.norm_type == "L2":
            norm = float((x_adv - x_clean).norm(2).item())
        else:
            norm = float((x_adv - x_clean).ne(0).sum().item())

        if norm < self.config.perturb.min_linf_delta:
            return EvaluationResult(
                score=0.0,
                reason="below_min_delta",
                model_prediction=prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                epsilon=float(challenge.epsilon),
            )
        effective_max_delta = min(float(challenge.epsilon), float(self.config.perturb.max_linf_delta))
        if norm > effective_max_delta:
            return EvaluationResult(
                score=0.0,
                reason="above_max_delta",
                model_prediction=prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
                epsilon=float(challenge.epsilon),
            )

        normalized_prediction = normalize_prediction_label(prediction)
        # Successful perturbation means the response label changes from original model output.
        if normalized_prediction == challenge.true_label:
            return EvaluationResult(
                score=0.0,
                reason="label_match_with_original",
                model_prediction=normalized_prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
                epsilon=float(challenge.epsilon),
            )

        rmse = float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item())

        ssim = _compute_ssim(x_clean=x_clean, x_adv=x_adv)
        min_ssim = float(getattr(self.config.perturb, "min_ssim", 0.98))
        if ssim < min_ssim:
            return EvaluationResult(
                score=0.0,
                reason="below_min_ssim",
                model_prediction=normalized_prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(rmse),
                epsilon=float(challenge.epsilon),
                ssim=float(ssim),
            )

        psnr_db = _compute_psnr_db(x_clean=x_clean, x_adv=x_adv)
        min_psnr_db = float(getattr(self.config.perturb, "min_psnr_db", 0.0))
        if min_psnr_db > 0.0 and psnr_db < min_psnr_db:
            return EvaluationResult(
                score=0.0,
                reason="below_min_psnr_db",
                model_prediction=normalized_prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(rmse),
                epsilon=float(challenge.epsilon),
                ssim=float(ssim),
                psnr_db=float(psnr_db),
            )

        denom = max(1e-12, effective_max_delta - float(self.config.perturb.min_linf_delta))
        linf_ratio = (norm - float(self.config.perturb.min_linf_delta)) / denom
        linf_ratio = min(max(linf_ratio, 0.0), 1.0)
        linf_score = (1.0 - linf_ratio) ** 2

        rmse_ratio = rmse / max(1e-12, effective_max_delta)
        rmse_ratio = min(max(rmse_ratio, 0.0), 1.0)
        rmse_score = (1.0 - rmse_ratio) ** 2

        linf_weight = float(getattr(self.config.perturb, "linf_component_weight", 0.7))
        rmse_weight = float(getattr(self.config.perturb, "rmse_component_weight", 0.3))
        total_weight = max(1e-12, linf_weight + rmse_weight)
        perturbation_score = ((linf_weight * linf_score) + (rmse_weight * rmse_score)) / total_weight

        time_ratio = response_time_ms / (challenge.timeout_seconds * 1000.0)
        speed_score = 1.0 - min(time_ratio, 1.0)

        score = C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score
        return EvaluationResult(
            score=float(score),
            reason="success",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=float(norm),
            rmse=float(rmse),
            epsilon=float(challenge.epsilon),
            ssim=float(ssim),
            psnr_db=float(psnr_db),
        )

    def _update_histories(self, uids: Sequence[int], rewards: Sequence[float]) -> None:
        for uid, reward in zip(uids, rewards):
            self.processed_counts[uid] += 1
            self.score_histories[uid].append(float(reward))

    def _set_weights(self) -> None:
        self._log_step_start(
            "set_weights",
            min_processed=self.config.perturb.min_processed_count,
            history_size=self.config.perturb.history_size,
        )
        eligible: list[tuple[int, float]] = []
        history_size = int(self.config.perturb.history_size)
        min_processed = int(self.config.perturb.min_processed_count)
        for uid in range(int(self.metagraph.n)):
            if int(self.processed_counts[uid]) < min_processed:
                continue
            history = self.score_histories[uid]
            if len(history) < history_size:
                continue
            tail = history[-history_size:]
            avg_score = float(sum(tail) / history_size)
            eligible.append((uid, avg_score))

        if not eligible:
            logger.warning(f"No eligible miners with processed_count >= {min_processed}.")
            return

        eligible.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        n_eligible = len(eligible)
        emission_raw = np.zeros(int(self.metagraph.n), dtype=np.float32)

        rank_to_uid: dict[int, int] = {}
        for rank0, (uid, avg_score) in enumerate(eligible):
            rank = rank0 + 1
            rank_to_uid[rank] = uid

        # Fixed top-5 emission schedule; ranks 6+ intentionally receive zero.
        top5_shares = (0.70, 0.25, 0.03, 0.015, 0.005)
        for rank, share in enumerate(top5_shares, start=1):
            if rank <= n_eligible:
                emission_raw[rank_to_uid[rank]] = float(share)

        # Only miners with positive average score may receive non-zero emissions.
        positive_uids = [uid for uid, avg_score in eligible if avg_score > 0.0]
        if not positive_uids:
            logger.warning("No miners with positive average score; setting all weights to zero.")
            zero_weights = np.zeros(int(self.metagraph.n), dtype=np.float32)
            uids = list(range(len(zero_weights)))
            ok, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids,
                weights=[float(v) for v in zero_weights.tolist()],
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            if ok:
                logger.info("set_weights success (all zero)")
            else:
                logger.error(f"set_weights failed (all zero): {msg}")
            return
        active_emission_total = float(sum(float(emission_raw[uid]) for uid in positive_uids))
        if active_emission_total <= 0.0:
            logger.warning("No positive-score miners in weighted rank buckets; setting all weights to zero.")
            zero_weights = np.zeros(int(self.metagraph.n), dtype=np.float32)
            uids = list(range(len(zero_weights)))
            ok, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids,
                weights=[float(v) for v in zero_weights.tolist()],
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            if ok:
                logger.info("set_weights success (all zero)")
            else:
                logger.error(f"set_weights failed (all zero): {msg}")
            return

        normalized = np.zeros(int(self.metagraph.n), dtype=np.float32)
        for uid in positive_uids:
            normalized[uid] = float(emission_raw[uid]) / active_emission_total
        for rank0, (uid, avg_score) in enumerate(eligible[:10]):
            rank = rank0 + 1
            logger.debug(
                f"rank={rank} uid={uid} avg_score={avg_score:.6f} emission_raw={emission_raw[uid]:.6f} emission={normalized[uid]:.6f}"
            )
        top_weight_items: list[str] = []
        for rank, (uid, avg_score) in enumerate(eligible[:5], start=1):
            top_weight_items.append(f"r{rank}:uid{uid}:avg={avg_score:.4f}:w={normalized[uid]:.4f}")
        self._log_summary(
            "weights_summary",
            eligible=n_eligible,
            distributed=min(5, n_eligible),
            top5="|".join(top_weight_items) if top_weight_items else "none",
        )

        # Scale miner emissions by configured share; route remainder to uid 0.
        miner_share = float(min(max(self.miner_emission_share, 0.0), 1.0))
        scaled = normalized * miner_share
        remainder = 1.0 - miner_share
        if len(scaled) > 0:
            scaled[0] = remainder

        uids = list(range(len(scaled)))
        weights = [float(v) for v in scaled.tolist()]
        ok, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        if ok:
            logger.info("set_weights success")
        else:
            logger.error(f"set_weights failed: {msg}")

    def run(self) -> None:
        self._log_step_start("validator_boot")
        self.sync()
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Validator hotkey is not registered on this netuid.")

        tempo = self.subtensor.get_subnet_hyperparameters(self.config.netuid).tempo
        self._log_summary(
            "validator_config",
            timeout=self.config.perturb.timeout_seconds,
            k_miners=self.config.perturb.k_miners,
            history_size=self.config.perturb.history_size,
            min_processed=self.config.perturb.min_processed_count,
            min_linf=self.config.perturb.min_linf_delta,
            max_linf=self.config.perturb.max_linf_delta,
            min_ssim=self.config.perturb.min_ssim,
            min_psnr_db=self.config.perturb.min_psnr_db,
            perturb_weight=C.PERTURBATION_WEIGHT,
            speed_weight=C.SPEED_WEIGHT,
            tempo=tempo,
            run_id=self.run_id,
        )

        while True:
            try:
                self._log_step_start("loop_sync_metagraph")
                self.sync()
                self._log_step_start("loop_get_current_block")
                block = self.subtensor.get_current_block()
                self._log_step_start("loop_generate_challenge", block=block)
                try:
                    challenge = self.generate_challenge(block=block)
                except Exception as exc:
                    logger.warning(
                        f"Challenge generation failed; retrying in {C.CHALLENGE_RETRY_DELAY_SECONDS}s: {exc}"
                    )
                    time.sleep(C.CHALLENGE_RETRY_DELAY_SECONDS)
                    continue
                self._log_summary(
                    "challenge_summary",
                    task_id=challenge.task_id,
                    epsilon=f"{challenge.epsilon:.4f}",
                    true_label=challenge.true_label,
                )

                self._log_step_start("loop_discover_miners")
                available_uids = self._available_miner_uids()
                if not available_uids:
                    logger.warning("No miners available")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                self._log_step_start("loop_select_miners", candidate_count=len(available_uids))
                valuable_uids = self._valuable_miner_uids(available_uids)
                miner_uids = self._select_random_miners(available_uids, seed=self._seed_from_block(block))
                if not miner_uids:
                    logger.warning("Miner selection is empty")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                self._log_summary(
                    "miner_selection",
                    selected=len(miner_uids),
                    valuable_pool=len(valuable_uids),
                    total_pool=len(available_uids),
                )

                self._log_step_start("loop_query_miners", selected_count=len(miner_uids))
                responses = self._run_query_miners(miner_uids, challenge)
                self._log_step_start("loop_score_responses", response_count=len(responses))
                rewards: list[float] = []
                results_by_uid: list[tuple[int, EvaluationResult]] = []
                response_log_lines: list[str] = []
                for uid, response in zip(miner_uids, responses):
                    status_code = getattr(response.dendrite, "status_code", 0) if response.dendrite else 0
                    process_time = getattr(response.dendrite, "process_time", None) if response.dendrite else None
                    response_time_ms = int((process_time or challenge.timeout_seconds) * 1000)

                    if status_code != 200 or not response.perturbed_image_b64:
                        result = EvaluationResult(
                            score=0.0,
                            reason="response_missing_or_status_error",
                            model_prediction="unavailable",
                            response_time_ms=response_time_ms,
                        )
                    else:
                        result = self.verify_and_score(
                            challenge=challenge,
                            perturbed_image_b64=response.perturbed_image_b64,
                            response_time_ms=response_time_ms,
                        )
                    score = float(result.score)
                    rewards.append(score)
                    results_by_uid.append((uid, result))
                    self.reason_counts_total[result.reason] += 1
                    response_log_lines.append(
                        f"uid={uid} status={status_code} score={score:.6f} "
                        f"response_time_ms={result.response_time_ms} "
                        f"processed={int(self.processed_counts[uid]) + 1} "
                        f"reason={result.reason} "
                        f"norm={result.norm:.6f} rmse={result.rmse:.6f} epsilon={result.epsilon:.6f} "
                        f"ssim={result.ssim:.6f} psnr_db={result.psnr_db:.4f}"
                    )

                all_zero_scores = bool(rewards) and all(score <= 0.0 for score in rewards)

                self._log_step_start("loop_update_histories")
                if all_zero_scores:
                    logger.warning(
                        "Skipping history update because all selected miner scores are zero "
                        f"(block={block}, selected={len(miner_uids)})."
                    )
                else:
                    if response_log_lines:
                        logger.info(
                            f"miner_response_evaluations block={block} count={len(response_log_lines)}\n"
                            + "\n".join(response_log_lines)
                        )
                    self._update_histories(miner_uids, rewards)
                    available_uid_set = set(available_uids)
                    unavailable_all_uids = [uid for uid in range(int(self.metagraph.n)) if uid not in available_uid_set]
                    if unavailable_all_uids:
                        self._update_histories(unavailable_all_uids, [0.0] * len(unavailable_all_uids))
                    reason_counts = Counter(result.reason for _, result in results_by_uid)
                    success_count = int(reason_counts.get("success", 0))
                    avg_score = float(sum(rewards) / max(1, len(rewards)))
                    max_score = float(max(rewards)) if rewards else 0.0
                    min_score = float(min(rewards)) if rewards else 0.0
                    avg_norm = float(sum(result.norm for _, result in results_by_uid) / max(1, len(results_by_uid)))
                    avg_rmse = float(sum(result.rmse for _, result in results_by_uid) / max(1, len(results_by_uid)))
                    self._log_summary(
                        "loop_summary",
                        block=block,
                        selected=len(miner_uids),
                        success=f"{success_count}/{len(results_by_uid)}",
                        avg_score=f"{avg_score:.4f}",
                        min_score=f"{min_score:.4f}",
                        max_score=f"{max_score:.4f}",
                        avg_norm=f"{avg_norm:.5f}",
                        avg_rmse=f"{avg_rmse:.5f}",
                        reasons=",".join([f"{k}:{v}" for k, v in sorted(reason_counts.items())]),
                    )
                
                self._log_step_start("loop_save_state")
                self._save_state()

                blocks_since_weights = block - self.last_weight_block
                if blocks_since_weights >= tempo:
                    self._log_step_start("loop_maybe_set_weights", blocks_since_weights=blocks_since_weights, tempo=tempo)
                    self._set_weights()
                    self.last_weight_block = block

                self.step += 1
            except KeyboardInterrupt:
                logger.info("Validator stopped by user.")
                break
            except Exception as exc:
                logger.error(f"Validator loop error: {exc}")
                time.sleep(5)
        if not self._query_loop.is_closed():
            self._query_loop.close()
        self._finish_wandb()


def build_config() -> bt.config:
    parser = argparse.ArgumentParser(description="Perturb subnet validator")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
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

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))
    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    perturb_cfg = type("PerturbConfig", (), {})()
    config.perturb = perturb_cfg
    for key, value in C.VALIDATOR_CONFIG.items():
        setattr(config.perturb, key, value)
    return config


if __name__ == "__main__":
    validator = PerturbValidator(config=build_config())
    validator.run()
    