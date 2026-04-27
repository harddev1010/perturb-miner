from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging as pylogging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import bittensor as bt
import numpy as np
import requests
import torch

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import load_efficientnet_b5, normalize_prediction_label, predict_label
from perturbnet.protocol import AttackChallenge


@dataclass
class ChallengeSpec:
    task_id: str
    model_name: str
    prompt: str
    clean_image_b64: str
    true_label: str
    epsilon: float
    norm_type: str
    timeout_seconds: int


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


def _make_axon(wallet, config):
    resolved_config = config() if callable(config) else config
    if hasattr(bt, "axon"):
        try:
            return bt.axon(wallet=wallet, config=resolved_config)
        except Exception:
            return bt.axon(wallet=wallet)
    axon_cls = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("No axon constructor found in bittensor.")
    try:
        return axon_cls(wallet=wallet, config=resolved_config)
    except Exception:
        return axon_cls(wallet=wallet)


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    level = getattr(pylogging, level_name, pylogging.INFO)
    pylogging.getLogger().setLevel(level)
    bt_logger = getattr(bt, "logging", None)
    if bt_logger is None:
        return
    try:
        if level_name == "DEBUG" and hasattr(bt_logger, "set_debug"):
            bt_logger.set_debug(True)
        elif level_name in {"WARNING", "WARN"} and hasattr(bt_logger, "set_warning"):
            bt_logger.set_warning(True)
        elif level_name == "ERROR" and hasattr(bt_logger, "set_error"):
            bt_logger.set_error(True)
        elif hasattr(bt_logger, "set_info"):
            bt_logger.set_info(True)
    except Exception:
        pass


class PerturbValidator:
    def __init__(self, config: bt.config) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = _make_subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.dendrite = _make_dendrite(wallet=self.wallet)
        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self._query_loop = asyncio.new_event_loop()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_b5(self.device)
        self.step = 0
        self.last_weight_block = 0
        self.state_path = os.path.join(self.config.logging.logging_dir, C.VALIDATOR_STATE_FILENAME)

        self.processed_counts = np.zeros(int(self.metagraph.n), dtype=np.int32)
        self.score_histories: list[list[float]] = [[] for _ in range(int(self.metagraph.n))]
        self.uid_hotkeys: list[str] = list(self.metagraph.hotkeys[: int(self.metagraph.n)])

        self._load_state()

    def _log_step_start(self, step_name: str, **context: Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            bt.logging.info(f"[STEP_START] {step_name} {rendered}")
        else:
            bt.logging.info(f"[STEP_START] {step_name}")

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
        bt.logging.info(f"Reset uid={uid} stats due to {reason}.")

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
        with open(self.state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
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
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _seed_from_block(self, block: int) -> int:
        digest = hashlib.sha256(f"{C.SUBNET_NAMESPACE}:{self.config.netuid}:{block}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _sample_epsilon(self, seed: int) -> float:
        # Deterministic epsilon in [0.06, 0.2]
        return 0.06 + (seed % 1400) / 10000.0

    def _choose_prompt(self, seed: int) -> str:
        rng = random.Random(seed)
        return rng.choice(list(C.PROMPTS))

    def _parse_llm_endpoint_result(self, payload: Any) -> bool | None:
        if isinstance(payload, bool):
            return payload
        if not isinstance(payload, dict):
            return None

        for key in ("is_match", "match", "ok", "valid"):
            value = payload.get(key)
            if isinstance(value, bool):
                return value
        return None

    def _llm_endpoint_check(self, predicted_label: str, expected_label: str) -> bool:
        endpoint = str(
            getattr(
                self.config.perturb,
                "llm_endpoint_url",
                getattr(self.config.perturb, "label_match_endpoint", ""),
            )
            or ""
        ).strip()
        normalized_prediction = normalize_prediction_label(predicted_label)
        if not endpoint:
            bt.logging.error("LLM endpoint url is empty; rejecting verification check.")
            return False

        payload = {
            "prediction": normalized_prediction,
            "target_label": expected_label,
            "llm_model": str(
                getattr(
                    self.config.perturb,
                    "llm_endpoint_model",
                    getattr(self.config.perturb, "label_match_model", C.LLM_ENDPOINT_MODEL),
                )
            ),
        }
        self._log_step_start(
            "llm_endpoint_check",
            endpoint=endpoint,
            prediction=normalized_prediction,
            expected=expected_label,
        )
        try:
            response = requests.post(endpoint, json=payload, timeout=8)
            response.raise_for_status()
            parsed = self._parse_llm_endpoint_result(response.json())
            if parsed is None:
                bt.logging.error("LLM endpoint returned unrecognized payload shape; rejecting check.")
                return False
            return bool(parsed)
        except Exception as exc:
            bt.logging.error(f"LLM endpoint request failed ({exc}); rejecting check.")
            return False

    def _fetch_image_for_prompt(self, prompt: str, seed: int) -> str:
        endpoint = self.config.perturb.image_endpoint
        params = {
            "prompt": prompt,
            "seed": seed,
            "image_size": self.config.perturb.image_size,
            "random_mode": "true",
        }
        response = requests.get(endpoint, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        image_b64 = str(data.get("image_base64", ""))
        if not image_b64:
            raise ValueError("image response missing image_base64")
        return image_b64

    def _load_fallback_image_b64(self) -> str:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback_path = os.path.join(project_root, C.FALLBACK_IMAGE_RELATIVE_PATH)
        with open(fallback_path, "rb") as handle:
            raw = handle.read()
        if not raw:
            raise ValueError(f"fallback image is empty: {fallback_path}")
        return base64.b64encode(raw).decode("utf-8")

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
            chosen_prompt = self._choose_prompt(seed)
            self._log_step_start(
                "challenge_attempt",
                attempt=attempt + 1,
                seed=seed,
                prompt=chosen_prompt,
            )
            try:
                self._log_step_start("challenge_fetch_image", prompt=chosen_prompt, seed=seed)
                image_b64 = self._fetch_image_for_prompt(prompt=chosen_prompt, seed=seed)
                effective_prompt = chosen_prompt
            except Exception as exc:
                bt.logging.warning(f"Challenge image fetch failed ({exc}), using fallback dog image.")
                try:
                    self._log_step_start("challenge_load_fallback_image", label=C.FALLBACK_LABEL)
                    image_b64 = self._load_fallback_image_b64()
                    effective_prompt = C.FALLBACK_LABEL
                except Exception as fallback_exc:
                    bt.logging.warning(f"Fallback image load failed, retrying: {fallback_exc}")
                    continue

            epsilon = self._sample_epsilon(seed)
            task_id = f"{block}-{seed}"
            self._log_step_start("challenge_prepare", task_id=task_id, epsilon=f"{epsilon:.4f}")

            try:
                self._log_step_start("challenge_model_inference", task_id=task_id)
                image = decode_image_b64(image_b64).to(self.device)
                predicted = predict_label(self.model, image)
                predicted_label = normalize_prediction_label(predicted)
            except Exception as exc:
                bt.logging.warning(f"Challenge decode/model validation failed, retrying: {exc}")
                continue

            # Verify the candidate by semantically checking model output against the API prompt label.
            if not self._llm_endpoint_check(predicted_label, effective_prompt):
                bt.logging.debug(
                    f"Challenge rejected by classifier consistency: pred={predicted_label} expected={effective_prompt}"
                )
                continue

            return ChallengeSpec(
                task_id=task_id,
                model_name=model_name,
                prompt=effective_prompt,
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
        uids: list[int] = []
        for uid in range(int(self.metagraph.n)):
            if self.metagraph.hotkeys[uid] == my_hotkey:
                continue
            if self.metagraph.axons[uid].ip == "0.0.0.0":
                continue
            uids.append(uid)
        return uids

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
            prompt=challenge.prompt,
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

    def verify_and_score(self, challenge: ChallengeSpec, perturbed_image_b64: str, response_time_ms: int) -> float:
        self._log_step_start(
            "verify_and_score",
            task_id=challenge.task_id,
            response_time_ms=response_time_ms,
        )
        try:
            x_clean = decode_image_b64(challenge.clean_image_b64).to(self.device)
            x_adv = decode_image_b64(perturbed_image_b64).to(self.device)
        except Exception:
            return 0.0

        if x_adv.shape != x_clean.shape:
            return 0.0
        if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
            return 0.0

        if challenge.norm_type == "Linf":
            norm = (x_adv - x_clean).abs().max().item()
        elif challenge.norm_type == "L2":
            norm = float((x_adv - x_clean).norm(2).item())
        else:
            norm = float((x_adv - x_clean).ne(0).sum().item())

        if norm < self.config.perturb.min_linf_delta:
            return 0.0
        if norm > min(float(challenge.epsilon), float(self.config.perturb.max_linf_delta)):
            return 0.0

        prediction = predict_label(self.model, x_adv)
        # Successful perturbation means the response no longer verifies as original label.
        if self._llm_endpoint_check(prediction, challenge.true_label):
            return 0.0

        perturbation_ratio = norm / max(1e-12, float(challenge.epsilon))
        perturbation_score = 1.0 - min(perturbation_ratio, 1.0)

        time_ratio = response_time_ms / (challenge.timeout_seconds * 1000.0)
        speed_score = 1.0 - min(time_ratio, 1.0)

        return C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score

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
            bt.logging.warning(f"No eligible miners with processed_count >= {min_processed}.")
            return

        eligible.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        n_eligible = len(eligible)
        emission_raw = np.zeros(int(self.metagraph.n), dtype=np.float32)

        rank_to_uid: dict[int, int] = {}
        for rank0, (uid, avg_score) in enumerate(eligible):
            rank = rank0 + 1
            rank_to_uid[rank] = uid

        # Top-3 fixed emission.
        if n_eligible >= 1:
            emission_raw[rank_to_uid[1]] = 0.50
        if n_eligible >= 2:
            emission_raw[rank_to_uid[2]] = 0.30
        if n_eligible >= 3:
            emission_raw[rank_to_uid[3]] = 0.10

        # Ranks 4-10 share 5% with inverse-rank decay.
        start_4_10 = 4
        end_4_10 = min(10, n_eligible)
        if end_4_10 >= start_4_10:
            denom_4_10 = sum((1.0 / r) for r in range(start_4_10, end_4_10 + 1))
            if denom_4_10 > 0:
                for r in range(start_4_10, end_4_10 + 1):
                    emission_raw[rank_to_uid[r]] = float(0.05 * (1.0 / r) / denom_4_10)

        # Ranks 11+ share 5% with inverse-rank decay.
        start_11 = 11
        if n_eligible >= start_11:
            denom_11p = sum((1.0 / r) for r in range(start_11, n_eligible + 1))
            if denom_11p > 0:
                for r in range(start_11, n_eligible + 1):
                    emission_raw[rank_to_uid[r]] = float(0.05 * (1.0 / r) / denom_11p)

        # Only miners with positive average score may receive non-zero emissions.
        positive_uids = [uid for uid, avg_score in eligible if avg_score > 0.0]
        if not positive_uids:
            bt.logging.warning("No miners with positive average score; setting all weights to zero.")
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
                bt.logging.info("set_weights success (all zero)")
            else:
                bt.logging.error(f"set_weights failed (all zero): {msg}")
            return
        active_emission_total = float(sum(float(emission_raw[uid]) for uid in positive_uids))
        if active_emission_total <= 0.0:
            bt.logging.warning("No positive-score miners in weighted rank buckets; setting all weights to zero.")
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
                bt.logging.info("set_weights success (all zero)")
            else:
                bt.logging.error(f"set_weights failed (all zero): {msg}")
            return

        normalized = np.zeros(int(self.metagraph.n), dtype=np.float32)
        for uid in positive_uids:
            normalized[uid] = float(emission_raw[uid]) / active_emission_total
        for rank0, (uid, avg_score) in enumerate(eligible):
            rank = rank0 + 1
            bt.logging.info(
                f"rank={rank} uid={uid} avg100={avg_score:.6f} emission={emission_raw[uid]:.6f} final_weight={normalized[uid]:.6f}"
            )

        uids = list(range(len(normalized)))
        weights = [float(v) for v in normalized.tolist()]
        ok, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        if ok:
            bt.logging.info("set_weights success")
        else:
            bt.logging.error(f"set_weights failed: {msg}")

    def run(self) -> None:
        self._log_step_start("validator_boot")
        self.sync()
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Validator hotkey is not registered on this netuid.")

        self._log_step_start("validator_serve_axon", port=getattr(self.config.axon, "port", "unknown"))
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        tempo = self.subtensor.get_subnet_hyperparameters(self.config.netuid).tempo
        bt.logging.info(f"Validator started with tempo={tempo}")

        while True:
            try:
                self._log_step_start("loop_sync_metagraph")
                self.sync()
                self._log_step_start("loop_get_current_block")
                block = self.subtensor.get_current_block()
                self._log_step_start("loop_generate_challenge", block=block)
                challenge = self.generate_challenge(block=block)
                bt.logging.info(
                    f"Challenge task={challenge.task_id} prompt={challenge.prompt} eps={challenge.epsilon:.4f}"
                )

                self._log_step_start("loop_discover_miners")
                available_uids = self._available_miner_uids()
                if not available_uids:
                    bt.logging.warning("No miners available")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                self._log_step_start("loop_select_miners", candidate_count=len(available_uids))
                valuable_uids = self._valuable_miner_uids(available_uids)
                miner_uids = self._select_random_miners(available_uids, seed=self._seed_from_block(block))
                if not miner_uids:
                    bt.logging.warning("Miner selection is empty")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                bt.logging.info(
                    f"Selected {len(miner_uids)} miners (valuable pool={len(valuable_uids)}, total pool={len(available_uids)})"
                )

                self._log_step_start("loop_query_miners", selected_count=len(miner_uids))
                responses = self._run_query_miners(miner_uids, challenge)
                self._log_step_start("loop_score_responses", response_count=len(responses))
                rewards: list[float] = []
                for uid, response in zip(miner_uids, responses):
                    status_code = getattr(response.dendrite, "status_code", 0) if response.dendrite else 0
                    process_time = getattr(response.dendrite, "process_time", None) if response.dendrite else None
                    response_time_ms = int((process_time or challenge.timeout_seconds) * 1000)

                    if status_code != 200 or not response.perturbed_image_b64:
                        score = 0.0
                    else:
                        score = self.verify_and_score(
                            challenge=challenge,
                            perturbed_image_b64=response.perturbed_image_b64,
                            response_time_ms=response_time_ms,
                        )
                    rewards.append(score)
                    bt.logging.info(
                        f"uid={uid} status={status_code} score={score:.6f} processed={int(self.processed_counts[uid]) + 1}"
                    )

                self._log_step_start("loop_update_histories")
                self._update_histories(miner_uids, rewards)
                self._log_step_start("loop_save_state")
                self._save_state()

                blocks_since_weights = block - self.last_weight_block
                if blocks_since_weights >= tempo:
                    self._log_step_start("loop_maybe_set_weights", blocks_since_weights=blocks_since_weights, tempo=tempo)
                    self._set_weights()
                    self.last_weight_block = block

                self.step += 1
                self._log_step_start("loop_sleep", seconds=self.config.perturb.query_interval_seconds)
                time.sleep(self.config.perturb.query_interval_seconds)
            except KeyboardInterrupt:
                bt.logging.info("Validator stopped by user.")
                break
            except Exception as exc:
                bt.logging.error(f"Validator loop error: {exc}")
                time.sleep(5)
        if not self._query_loop.is_closed():
            self._query_loop.close()


def build_config() -> bt.config:
    parser = argparse.ArgumentParser(description="Perturb subnet validator")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("VALIDATOR_PORT", os.getenv("AXON_PORT", "8090"))),
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

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))
    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = int(getattr(config.axon, "port", getattr(config, "axon_port", 8090)))
    config.axon.ip = getattr(config.axon, "ip", os.getenv("VALIDATOR_IP", os.getenv("AXON_IP", "0.0.0.0")))
    config.axon.external_ip = getattr(config.axon, "external_ip", os.getenv("VALIDATOR_EXTERNAL_IP", None))
    config.axon.external_port = int(
        getattr(config.axon, "external_port", os.getenv("VALIDATOR_EXTERNAL_PORT", str(config.axon.port)))
    )
    config.axon.max_workers = int(getattr(config.axon, "max_workers", os.getenv("AXON_MAX_WORKERS", "10")))

    perturb_cfg = type("PerturbConfig", (), {})()
    config.perturb = perturb_cfg
    for key, value in C.VALIDATOR_CONFIG.items():
        setattr(config.perturb, key, value)
    return config


if __name__ == "__main__":
    validator = PerturbValidator(config=build_config())
    validator.run()

