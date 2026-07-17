"""
miner.py — Perturb subnet miner (task-API flow) — bittensor plumbing + perturb().

Network flow (post axon-removal refactor): the miner polls the task API for the current clean image,
computes an adversarial perturbation, uploads the result image to object storage, then submits the
response URL back to the API. There is no axon / validator dendrite query anymore.

The attack engine lives in the neurons/perturb/ package (batched flip-first pipeline: multi-loss
qFGSM, one-byte PGD, top-M boundary, saliency, gradient-seeded Square). This file only wires the
task loop, loads the model, warms it up, and routes each clean image through perturb().

Tunable env vars + algorithm switches are documented in neurons/perturb/constants.py and perturb.py.
The previous inline "dynamic chunked" engine is preserved in neurons/miner_backup/miner_chunked_0618.py.
"""

from __future__ import annotations

import argparse
import hashlib
import logging as pylogging
import os
import time
import typing

import bittensor as bt
import torch

from perturbnet import challenge_store
from perturbnet import constants as C
from perturbnet.api_client import get_current_task, get_server_epoch, submit_miner_response
from perturbnet.constants import MAX_LINF_DELTA
from perturbnet.image_io import decode_image_b64, encode_image_b64, image_url_to_b64
from perturbnet.model import load_efficientnet_v2_l, logits_for_images, predict_label, resolve_target_index
from perturbnet.storage_uploader import ImageStorageUploader

from neurons.perturb import perturb
from neurons.perturb.utils import cw_margin, estimate_validator_score, png_roundtrip, validator_score

logger = pylogging.getLogger(__name__)

# Wall-clock budget for the perturb() attack loop itself. The validator accepts submissions from
# ~40s to ~90s after each task boundary (see PERTURB_VALIDATOR_EVALUATION_DELAY_SECONDS /
# PERTURB_VALIDATOR_EVALUATION_POLL_SECONDS in perturbnet/constants.py), and the download/upload/
# submit round-trip normally costs only a few seconds, so most of that window can go to the attack.
_ATTACK_TIMEOUT_SECONDS = float(os.getenv("PERTURB_ATTACK_TIMEOUT_SECONDS") or "35.0")

# Test mode (PERTURB_TEST_MODE=1): run the WHOLE pipeline — poll -> download -> attack -> score —
# but DON'T upload the result to object storage or submit it back to the API. The full validator
# score is already logged by _attack_image, so this lets you observe scores locally (dry run)
# without touching storage/the network or requiring a registered, submission-eligible hotkey.
_TEST_MODE = (os.getenv("PERTURB_TEST_MODE") or "").strip().lower() in {"1", "true", "yes", "on"}

# Persist each processed challenge (clean image + our score/time) to hippius under
# perturb_challenges/<task_id>.json so scripts/score_miner.py can later replay the CURRENT engine
# against the exact real challenges we saw. Storage happens AFTER submit so it never eats the
# submission window, and never in TEST_MODE (which is contractually storage/network-free).
_STORE_CHALLENGES = (os.getenv("PERTURB_STORE_CHALLENGES") or "1").strip().lower() in {"1", "true", "yes", "on"}


def _task_created_epoch(task_id: str) -> float | None:
    """Task ids look like '<unix_epoch>-hf-...'. Parse the leading epoch so we can measure how old a
    task already is by the time we fetch/submit it (age = now - created). A large age means we are
    discovering the task late (clock skew vs the server, or the server serving a stale task), which
    closes the submission window before our pipeline even runs. Returns None if unparseable."""
    head = str(task_id).split("-", 1)[0].strip()
    return float(head) if head.isdigit() else None


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


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


def _warmup(model: torch.nn.Module, device: torch.device) -> None:
    """Warm CUDA kernels / allocator / cuDNN at first load so the first real task does not pay
    JIT + autotune latency. Exercises the exact inference path: forward + backward + PNG round-trip."""
    t0 = time.time()
    try:
        logger.info(f"[MINER] warmup start device={device.type}")
        x = torch.rand(1, 3, 480, 480, device=device, requires_grad=True)
        logits_for_images(model=model, image_bchw=x).sum().backward()
        _ = png_roundtrip(torch.rand(3, 480, 480, device=device), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.info(f"[MINER] warmup done device={device.type} elapsed={time.time() - t0:.2f}s")
    except Exception as err:
        logger.warning(f"[MINER] warmup skipped: {err}")


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
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", "unknown"))
        self.response_exporter = ImageStorageUploader(
            run_id=f"miner-{hotkey[:8]}-{os.getpid()}",
            netuid=int(self.config.netuid),
            uploader_hotkey=hotkey,
        )
        self.last_processed_task_id = ""
        # Correction for local clock skew: server_time ≈ time.time() + _server_offset, measured from
        # the API's HTTP Date header. Lets us log the TRUE age of a task even when the host clock is wrong.
        self._server_offset = 0.0
        self._server_offset_ts = 0.0

    def _maybe_refresh_server_offset(self) -> None:
        """Refresh the local→server clock offset at most once a minute (cheap; skew drifts slowly).
        Call only while idle-polling, never mid-attack, so it never adds latency to the pipeline."""
        now = time.time()
        if now - self._server_offset_ts < 60.0:
            return
        self._server_offset_ts = now
        epoch = get_server_epoch(
            base_url=str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
            timeout_seconds=float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)),
        )
        if epoch is not None:
            self._server_offset = epoch - now
            logger.info(f"[clock] local→server offset={self._server_offset:+.1f}s (host clock is that far ahead if negative)")

    def _server_now(self) -> float:
        """Best estimate of the server's wall clock, correcting for measured local skew."""
        return time.time() + self._server_offset

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

    def _miner_uid(self) -> int:
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", ""))
        if hotkey not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")
        return int(self.metagraph.hotkeys.index(hotkey))

    def _attack_image(self, *, task_id: str, clean_image_b64: str) -> tuple[str, str, dict[str, typing.Any]]:
        """Run the neurons/perturb engine on the clean image. Returns (perturbed_image_b64,
        predicted_label, metrics). Falls back to the clean bytes verbatim when no envelope-safe flip is
        found. `metrics` carries the challenge params + this task's score/flip result for the challenge
        store (see challenge_store.build_record)."""
        t_received = time.time()
        clean = decode_image_b64(clean_image_b64).to(self.device)
        predicted_label = predict_label(self.model, clean)
        target_index = resolve_target_index(predicted_label)
        if target_index is None:
            raise RuntimeError(f"Unable to resolve predicted label for task={task_id}: {predicted_label}")

        epsilon = float(getattr(self.config.perturb, "max_linf_delta", C.MAX_LINF_DELTA))
        min_delta = float(getattr(self.config.perturb, "min_linf_delta", C.MIN_LINF_DELTA))

        adv = perturb(
            self.model,
            clean,
            target_index,
            epsilon,
            min_delta,
            self.device,
            timeout_seconds=_ATTACK_TIMEOUT_SECONDS,
            start_time=t_received,
        )
        adv_b64 = encode_image_b64(adv)
        seen = decode_image_b64(adv_b64).to(self.device)
        diff = seen - clean
        norm = float(diff.abs().max().item())
        rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())

        # Base challenge params — always recorded so score_miner.py can replay this exact image.
        metrics: dict[str, typing.Any] = {
            "true_label": str(predicted_label),
            "target_index": int(target_index),
            "epsilon": float(epsilon),
            "min_delta": float(min_delta),
            "flip": False,
            "q1_flip": False,
            "q2_flip": False,
            "score": 0.0,
            "pert_score": 0.0,
            "margin": 0.0,
            "linf": float(norm),
            "rmse": float(rmse),
            "k": 0,
            "changed_pixels": 0,
            "changed_channels": 0,
        }

        # Clean tier: perturb() returned the clean image (no flip found). Pass the original bytes
        # through verbatim rather than the re-encoded ones.
        if norm < min_delta:
            logger.info(
                f"No flip found task={task_id} target_idx={target_index} norm={norm:.6f} "
                f"min_delta={min_delta:.6f} elapsed={time.time() - t_received:.3f}s"
            )
            return clean_image_b64, str(predicted_label), metrics

        # Log BOTH: pert_score = L∞+RMSE component only; score = the FULL validator objective
        # (adds margin + novelty terms). The full score is what the validator actually assigns.
        pert_score = estimate_validator_score(norm, rmse, epsilon)
        with torch.no_grad():
            adv_logits = logits_for_images(model=self.model, image_bchw=seen.unsqueeze(0))[0]
        cw = cw_margin(adv_logits, target_index)  # true - best_other; validator margin = -cw
        changed_pixels = int((diff.abs() > (0.5 / 255.0)).any(dim=0).sum().item())
        changed_channels = int((diff.abs() > (0.5 / 255.0)).sum().item())
        full_score = validator_score(norm, rmse, cw, changed_pixels, min(epsilon, MAX_LINF_DELTA))
        # Fixed-q invariant: q=1 flips step every changed coord by exactly 1 byte (k=1); the q=2
        # fallback steps by 2 (k>=2). So k = round(linf*255) attributes the flip to its search path.
        k = int(round(norm * 255.0))
        metrics.update(
            flip=True,
            q1_flip=(k <= 1),
            q2_flip=(k >= 2),
            score=float(full_score),
            pert_score=float(pert_score),
            margin=float(cw),
            k=int(k),
            changed_pixels=int(changed_pixels),
            changed_channels=int(changed_channels),
        )
        logger.info(
            f"Finished task={task_id} target_idx={target_index} k={k} "
            f"linf={norm:.6f} rmse={rmse:.6f} pert_score={pert_score:.4f} score={full_score:.4f} "
            f"elapsed={time.time() - t_received:.3f}s"
        )
        return adv_b64, str(predicted_label), metrics

    def _upload_response(self, *, task_id: str, perturbed_image_b64: str) -> str:
        uid = self._miner_uid()
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", ""))
        miner_storage_key = self.response_exporter.miner_storage_key(miner_uid=uid, miner_hotkey=hotkey)
        safe_task_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
        image_hash = hashlib.sha256(perturbed_image_b64.encode("utf-8")).hexdigest()[:12]
        key = f"{C.STORAGE_PREFIX.strip().strip('/')}/miner-responses/{safe_task_id}/{miner_storage_key}_{image_hash}.png"
        return self.response_exporter.upload_image_b64(key=key, image_b64=perturbed_image_b64)

    def _submission_succeeded(self, response: typing.Any) -> bool:
        if response is None:
            return True
        if isinstance(response, dict):
            raw = response.get("success", response.get("ok", response.get("status")))
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                return raw.strip().lower() in {"success", "succeeded", "ok", "true"}
            has_miner_uid = any(key in response for key in ("miner_uid", "miner_id", "minerUid"))
            has_image_url = any(response.get(key) for key in ("imageURL", "imageUrl", "image_url"))
            if has_miner_uid and has_image_url:
                return True
        return False

    def _process_task(self, *, task_id: str, image_url: str) -> None:
        # The whole chain (download -> attack -> upload -> submit) must land inside the task's open
        # submission window, otherwise the API rejects the submit with HTTP 403. Time each phase so
        # the bottleneck is visible; the breakdown is logged even when the submit fails (see finally).
        started_at = time.time()
        api_timeout = float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS))

        t0 = time.time()
        clean_image_b64 = image_url_to_b64(image_url, timeout_seconds=api_timeout)
        download_seconds = time.time() - t0

        t0 = time.time()
        perturbed_image_b64, _, attack_metrics = self._attack_image(task_id=task_id, clean_image_b64=clean_image_b64)
        attack_seconds = time.time() - t0

        # Dry run: skip the competition submit (upload response + POST /submits), but STILL persist the
        # challenge record so test-mode runs populate the store that scripts/score_miner.py reads. The
        # response image is not uploaded; only our own challenge JSON is written (gated + best-effort).
        if _TEST_MODE:
            self._store_challenge_record(
                task_id=task_id,
                image_url=image_url,
                clean_image_b64=clean_image_b64,
                metrics=attack_metrics,
                time_spent_seconds=attack_seconds,
            )
            logger.info(
                f"TEST_MODE task={task_id} download={download_seconds:.2f}s attack={attack_seconds:.2f}s "
                f"total={time.time() - started_at:.2f}s (response upload+submit SKIPPED; challenge stored)"
            )
            return

        upload_seconds = 0.0
        submit_seconds = 0.0
        submit_response = None
        response_url = ""
        try:
            t0 = time.time()
            response_url = self._upload_response(task_id=task_id, perturbed_image_b64=perturbed_image_b64)
            upload_seconds = time.time() - t0

            t0 = time.time()
            try:
                submit_response = submit_miner_response(
                    base_url=str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
                    wallet=self.wallet,
                    image_url=response_url,
                    timeout_seconds=api_timeout,
                )
            finally:
                submit_seconds = time.time() - t0
                total_seconds = time.time() - started_at
                created = _task_created_epoch(task_id)
                age_str = f" true_age_at_submit={self._server_now() - created:.1f}s" if created is not None else ""
                logger.info(
                    f"Timing task={task_id} download={download_seconds:.2f}s attack={attack_seconds:.2f}s "
                    f"upload={upload_seconds:.2f}s submit={submit_seconds:.2f}s "
                    f"total_got_task_to_submit={total_seconds:.2f}s{age_str}"
                )
        finally:
            # Persist the challenge + our score regardless of how upload/submit went — success, a rejected
            # window (HTTP 403), an upload error, or any exception. Runs after the submit attempt so it
            # never delays the response, and is best-effort so it never masks the upload/submit error.
            self._store_challenge_record(
                task_id=task_id,
                image_url=image_url,
                clean_image_b64=clean_image_b64,
                metrics=attack_metrics,
                time_spent_seconds=attack_seconds,
            )
        if not self._submission_succeeded(submit_response):
            raise RuntimeError(f"Response submission failed task={task_id} api_response={submit_response}")
        logger.info(
            f"Submitted task={task_id} response_url={response_url} "
            f"total_got_task_to_submit={total_seconds:.2f}s"
        )

    def _store_challenge_record(
        self,
        *,
        task_id: str,
        image_url: str,
        clean_image_b64: str,
        metrics: dict[str, typing.Any],
        time_spent_seconds: float,
    ) -> None:
        """Best-effort: upload this task's challenge record to the cloud store (R2) at
        challenge_store.CHALLENGE_PREFIX (perturb/attack-challenges/<task_id>.json), reusing the same
        uploader as the response image. Called after the submit, on a fail-fast client. Any failure is
        logged and swallowed — persisting the challenge must never break the miner's task loop."""
        if not _STORE_CHALLENGES:
            logger.debug(f"Challenge store disabled (PERTURB_STORE_CHALLENGES) task={task_id}")
            return
        key = challenge_store.challenge_key(task_id)
        try:
            record = challenge_store.build_record(
                task_id=task_id,
                image_b64=clean_image_b64,
                true_label=str(metrics.get("true_label", "")),
                target_index=int(metrics.get("target_index", -1)),
                epsilon=float(metrics.get("epsilon", C.MAX_LINF_DELTA)),
                min_delta=float(metrics.get("min_delta", C.MIN_LINF_DELTA)),
                image_url=image_url,
                created_epoch=_task_created_epoch(task_id),
                metrics=metrics,
                time_spent_seconds=time_spent_seconds,
            )
            url = challenge_store.store_challenge(self.response_exporter, record)
            logger.info(f"Stored challenge task={task_id} key={key} url={url}")
        except Exception as exc:
            logger.warning(f"Challenge store FAILED task={task_id} key={key}: {exc!r}", exc_info=True)

    def _get_current_task(self):
        return get_current_task(
            base_url=str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
            timeout_seconds=float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)),
        )

    def run(self) -> None:
        # Task discovery is by CONTINUOUS POLLING, not local-clock boundaries. Boundary sync assumes
        # the miner's wall clock matches the server's; on a skewed host clock (common in containers we
        # can't set the time on) that makes us fetch tasks long after their submission window closed.
        # Polling for a changed task_id reacts to the server's real task-creation moment within
        # ~poll_seconds regardless of clock skew, so submissions land inside the open window.
        self.sync()
        self._miner_uid()
        poll_seconds = float(getattr(self.config.perturb, "task_poll_time", C.TASK_POLL_TIME))
        try:
            current_task = self._get_current_task()
        except Exception as exc:
            logger.warning(f"Initial task fetch failed: {exc}")
            current_task = None
        # Skip whatever is already current at startup; only act on the NEXT task the server publishes.
        self.last_processed_task_id = current_task.task_id if current_task is not None else ""
        logger.info(
            f"Miner started (poll mode, every {poll_seconds:.1f}s). "
            f"baseline_task_id={self.last_processed_task_id or '(none)'}"
        )
        while True:
            try:
                task = self._get_current_task()
                if task is None or task.task_id == self.last_processed_task_id:
                    self._maybe_refresh_server_offset()
                    time.sleep(poll_seconds)
                    continue

                created = _task_created_epoch(task.task_id)
                age_str = f" true_age_at_fetch={self._server_now() - created:.1f}s" if created is not None else ""
                logger.info(f"New task found task_id={task.task_id}{age_str} image_url={task.image_url}")
                try:
                    self._process_task(task_id=task.task_id, image_url=task.image_url)
                finally:
                    self.last_processed_task_id = task.task_id
            except Exception as exc:
                logger.warning(f"Miner task loop failed: {exc}")
                time.sleep(poll_seconds)


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner")
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

    if not hasattr(config, "perturb"):
        config.perturb = type("PerturbConfig", (), {})()
    for key, value in C.VALIDATOR_CONFIG.items():
        setattr(config.perturb, key, getattr(config.perturb, key, value))

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))
    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()
