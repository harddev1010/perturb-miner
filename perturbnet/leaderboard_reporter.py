from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderboardMinerResult:
    uid: int
    hotkey: str
    coldkey: str
    incentive: float
    avg_score: float
    last_score: float
    rmse: float
    norm: float
    result: str
    image_url: str


@dataclass(frozen=True)
class LeaderboardNetworkMetrics:
    total_miners: int
    available_miners: int
    avg_score: float
    avg_rmse: float
    avg_norm: float
    success_count: int


@dataclass(frozen=True)
class LeaderboardReport:
    task_id: str
    validator_hotkey: str
    network: LeaderboardNetworkMetrics
    miners: list[LeaderboardMinerResult]


class LeaderboardReporter:
    """Non-blocking signed reports to the Perturb leaderboard backend."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_url: str,
        timeout_seconds: float,
        wallet: Any,
        validator_hotkey: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.api_url = str(api_url).strip()
        self.timeout_seconds = float(timeout_seconds)
        self.wallet = wallet
        self.validator_hotkey = validator_hotkey
        self.queue: queue.Queue[LeaderboardReport | None] = queue.Queue(maxsize=128)
        self.thread: threading.Thread | None = None
        if not self.enabled:
            return
        if not self.api_url:
            logger.warning("Leaderboard reporting enabled but API URL is empty; reports disabled.")
            self.enabled = False
            return
        self.thread = threading.Thread(target=self._worker, name="leaderboard-reporter", daemon=True)
        self.thread.start()
        logger.info(f"Leaderboard reporting enabled url={self.api_url}")

    def submit(self, report: LeaderboardReport) -> None:
        if not self.enabled:
            return
        try:
            self.queue.put_nowait(report)
        except queue.Full:
            logger.warning("Leaderboard report queue is full; dropping report.")

    def close(self, timeout_seconds: float = 10.0) -> None:
        if not self.enabled or self.thread is None:
            return
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=timeout_seconds)

    def _worker(self) -> None:
        while True:
            report = self.queue.get()
            if report is None:
                return
            try:
                self._post_with_retry(report=report)
                logger.info(
                    f"Leaderboard report succeeded task_id={report.task_id} "
                    f"miners={len(report.miners)} valid={report.network.success_count}"
                )
            except Exception as exc:
                logger.warning(f"Leaderboard report failed task_id={report.task_id}: {exc}")
            finally:
                self.queue.task_done()

    def _post_with_retry(self, *, report: LeaderboardReport) -> None:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                self._post(report=report)
                return
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning(f"Leaderboard report attempt failed task_id={report.task_id}, retrying once: {exc}")
        if last_error is not None:
            raise last_error

    def _post(self, *, report: LeaderboardReport) -> None:
        # Timestamp must be current at send time for backend replay protection.
        payload = {
            "task_id": report.task_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "validator_hotkey": report.validator_hotkey,
            "network": asdict(report.network),
            "miners": [asdict(miner) for miner in report.miners],
        }
        # Serialize once, sign these exact bytes, and POST the same bytes with
        # data=body. Do not use requests.post(json=...), which re-serializes.
        body = json.dumps(payload).encode("utf-8")
        signature = self._sign_payload(body)
        response = requests.post(
            self.api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": self.validator_hotkey,
                "X-Signature": signature,
            },
            timeout=self.timeout_seconds,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    def _sign_payload(self, body: bytes) -> str:
        hotkey = getattr(self.wallet, "hotkey", None)
        if hotkey is None or not hasattr(hotkey, "sign"):
            raise RuntimeError("Wallet hotkey does not support signing leaderboard reports.")
        signature = hotkey.sign(body)
        if isinstance(signature, bytes):
            return "0x" + signature.hex()
        if isinstance(signature, str):
            return signature if signature.startswith("0x") else "0x" + signature
        if hasattr(signature, "hex"):
            return "0x" + signature.hex()
        raise RuntimeError(f"Unsupported signature type: {type(signature).__name__}")
