"""
miner.py — Perturb subnet miner (netuid 26) — bittensor plumbing + forward().

The attack engine lives in the neurons/perturb/ package (batched flip-first pipeline: multi-loss
qFGSM, one-byte PGD, top-M boundary, saliency, gradient-seeded Square). This file only wires the
axon, loads the model, warms it up, and routes each AttackChallenge through perturb().

Tunable env vars + algorithm switches are documented in neurons/perturb/constants.py and perturb.py.
The previous inline "dynamic chunked" engine is preserved in neurons/miner_backup/miner_chunked_0618.py.
"""

import argparse
import json
import logging as pylogging
import os
import re
import time
import typing

import bittensor as bt
import torch

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_v2_l, logits_for_images, resolve_target_index
from perturbnet.protocol import AttackChallenge

from neurons.perturb import perturb
from neurons.perturb.utils import estimate_validator_score, png_roundtrip

logger = pylogging.getLogger(__name__)

# When perturb() finds no flip, dump the challenge's input params here (one JSON per case) for offline
# debugging/replay. Override the location with PERTURB_ERROR_CASES_DIR.
_ERROR_CASES_DIR = os.getenv("PERTURB_ERROR_CASES_DIR", "/workspace/Perturb_error_cases")

# Every incoming AttackChallenge is also archived here (one JSON per request). Only the newest
# _ATTACK_CHALLENGES_KEEP files are retained; older ones are rotated into _ATTACK_HISTORY_DIR.
_ATTACK_CHALLENGES_DIR = os.getenv("PERTURB_ATTACK_CHALLENGES_DIR", "/workspace/Perturb_attack_challenges")
_ATTACK_HISTORY_DIR = os.getenv("PERTURB_ATTACK_HISTORY_DIR", "/workspace/Perturb_attack_history")
_ATTACK_CHALLENGES_KEEP = int(os.getenv("PERTURB_ATTACK_CHALLENGES_KEEP", "200"))


def _rotate_attack_challenges() -> None:
    """Keep only the newest _ATTACK_CHALLENGES_KEEP files in _ATTACK_CHALLENGES_DIR; move the rest
    (oldest first, by mtime) into _ATTACK_HISTORY_DIR."""
    entries = [
        os.path.join(_ATTACK_CHALLENGES_DIR, n)
        for n in os.listdir(_ATTACK_CHALLENGES_DIR)
        if n.endswith(".json")
    ]
    if len(entries) <= _ATTACK_CHALLENGES_KEEP:
        return
    entries.sort(key=os.path.getmtime)  # oldest first
    overflow = entries[: len(entries) - _ATTACK_CHALLENGES_KEEP]
    os.makedirs(_ATTACK_HISTORY_DIR, exist_ok=True)
    for src in overflow:
        dst = os.path.join(_ATTACK_HISTORY_DIR, os.path.basename(src))
        try:
            os.replace(src, dst)
        except Exception as err:
            logger.warning(f"[attack-challenge] failed to archive {src}: {err}")


def _store_attack_challenge(synapse: AttackChallenge) -> None:
    """Persist every incoming challenge's AttackChallenge input params to {timestamp}_{task_id}.json
    under _ATTACK_CHALLENGES_DIR, then rotate so only the latest _ATTACK_CHALLENGES_KEEP remain.
    Best-effort: failures are logged, never raised, so they can't disturb the response."""
    try:
        os.makedirs(_ATTACK_CHALLENGES_DIR, exist_ok=True)
        task_id = str(getattr(synapse, "task_id", "unknown"))
        safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id) or "unknown"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_ATTACK_CHALLENGES_DIR, f"{timestamp}_{safe_task}.json")
        # Several requests can share a task_id / land in the same second; don't clobber.
        if os.path.exists(path):
            path = os.path.join(_ATTACK_CHALLENGES_DIR, f"{timestamp}_{safe_task}_{os.urandom(3).hex()}.json")
        payload = {
            "saved_at": timestamp,
            "task_id": task_id,
            "model_name": getattr(synapse, "model_name", None),
            "true_label": getattr(synapse, "true_label", None),
            "epsilon": getattr(synapse, "epsilon", None),
            "norm_type": getattr(synapse, "norm_type", None),
            "min_delta": getattr(synapse, "min_delta", None),
            "timeout_seconds": getattr(synapse, "timeout_seconds", None),
            "clean_image_b64": getattr(synapse, "clean_image_b64", None),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        _rotate_attack_challenges()
    except Exception as err:
        logger.warning(f"[attack-challenge] failed to store challenge task={getattr(synapse, 'task_id', 'unknown')}: {err}")


def _dump_error_case(synapse: AttackChallenge, reason: str) -> None:
    """Persist a no-flip challenge's AttackChallenge input params to {timestamp}_{task_id}.json.
    Best-effort: failures are logged, never raised, so they can't disturb the response."""
    try:
        os.makedirs(_ERROR_CASES_DIR, exist_ok=True)
        task_id = str(getattr(synapse, "task_id", "unknown"))
        safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id) or "unknown"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_ERROR_CASES_DIR, f"{timestamp}_{safe_task}.json")
        payload = {
            "reason": reason,
            "saved_at": timestamp,
            "task_id": task_id,
            "model_name": getattr(synapse, "model_name", None),
            "true_label": getattr(synapse, "true_label", None),
            "epsilon": getattr(synapse, "epsilon", None),
            "norm_type": getattr(synapse, "norm_type", None),
            "min_delta": getattr(synapse, "min_delta", None),
            "timeout_seconds": getattr(synapse, "timeout_seconds", None),
            "clean_image_b64": getattr(synapse, "clean_image_b64", None),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info(f"[error-case] saved no-flip challenge -> {path} (reason={reason})")
    except Exception as err:
        logger.warning(f"[error-case] failed to save challenge task={getattr(synapse, 'task_id', 'unknown')}: {err}")


def _warmup(model: torch.nn.Module, device: torch.device) -> None:
    """Warm CUDA kernels / allocator / cuDNN at first load so the first real challenge does not
    pay JIT + autotune latency. Exercises the exact inference path: forward + backward + PNG round-trip."""
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
        # _store_attack_challenge(synapse)
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
            # bytes through verbatim rather than re-encoding, and dump the case for offline replay.
            if norm < min_delta:
                synapse.perturbed_image_b64 = synapse.clean_image_b64
                _dump_error_case(synapse, "no_flip")
            else:
                synapse.perturbed_image_b64 = adv_b64
            h, w = clean.shape[1], clean.shape[2]
            est_score = 0.0 if norm < min_delta else estimate_validator_score(norm, rmse, epsilon)
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
            _dump_error_case(synapse, "exception")
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
