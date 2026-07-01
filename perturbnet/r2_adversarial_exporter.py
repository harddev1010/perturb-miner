from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import queue
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from perturbnet import constants as C

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdversarialExampleRecord:
    image_id: str
    hf_row: int | None
    task_id: str
    block: int
    netuid: int
    validator_hotkey: str
    model_name: str
    true_label: str
    adversarial_label: str
    score: float
    norm: float
    rmse: float
    epsilon: float
    ssim: float
    psnr_db: float
    response_time_ms: int
    image_sha256: str
    image_object_key: str
    created_at: str
    storage_mode: str
    miner_storage_key: str | None = None


@dataclass(frozen=True)
class AdversarialExportCandidate:
    miner_storage_key: str
    adversarial_label: str
    score: float
    norm: float
    rmse: float
    epsilon: float
    ssim: float
    psnr_db: float
    response_time_ms: int
    perturbed_image_b64: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_hf_row(image_id: str) -> int | None:
    try:
        return int(image_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


def _normalized_png_bytes(image_b64: str) -> bytes:
    from PIL import Image

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG", optimize=True)
        return output.getvalue()


def _image_sha256(image_b64: str) -> str:
    return hashlib.sha256(base64.b64decode(image_b64)).hexdigest()


class R2AdversarialDatasetExporter:
    """Non-blocking R2 writer for successful adversarial miner responses."""

    def __init__(
        self,
        *,
        run_id: str,
        netuid: int,
        validator_hotkey: str,
        storage_mode: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.netuid = int(netuid)
        self.validator_hotkey = validator_hotkey
        self.enabled = bool(C.R2_EXPORT_ENABLED)
        self.bucket = os.getenv("PERTURB_R2_BUCKET", "").strip()
        self.prefix = C.R2_PREFIX.strip().strip("/")
        self.storage_mode = (storage_mode or C.STORAGE_MODE).strip().lower()
        if not self.storage_mode:
            self.storage_mode = "latest"
        if self.storage_mode not in {"latest", "all"}:
            self._raise_config_error(f"Invalid storage_mode={self.storage_mode!r}; expected 'latest' or 'all'.")
        self.miner_key_secret = ""
        self.queue: queue.Queue[tuple[AdversarialExampleRecord, str] | None] = queue.Queue(
            maxsize=max(1, _env_int("PERTURB_R2_QUEUE_SIZE", 1024))
        )
        self.thread: threading.Thread | None = None
        self.client: Any | None = None

        if not self.enabled:
            return
        if not self.bucket:
            self._raise_config_error("R2 export is enabled but PERTURB_R2_BUCKET is empty.")
        try:
            import boto3  # type: ignore[reportMissingImports]
        except Exception as exc:
            self._raise_config_error(f"R2 export is enabled but boto3 is unavailable: {exc}")

        endpoint_url = os.getenv("PERTURB_R2_ENDPOINT_URL", "").strip()
        access_key = os.getenv("PERTURB_R2_ACCESS_KEY_ID", "").strip()
        secret_key = os.getenv("PERTURB_R2_SECRET_ACCESS_KEY", "").strip()
        if not endpoint_url or not access_key or not secret_key:
            self._raise_config_error(
                "R2 export is enabled and requires PERTURB_R2_ENDPOINT_URL, "
                "PERTURB_R2_ACCESS_KEY_ID, and PERTURB_R2_SECRET_ACCESS_KEY."
            )
        self.miner_key_secret = secret_key

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.getenv("PERTURB_R2_REGION", "auto"),
        )
        self.thread = threading.Thread(target=self._worker, name="r2-adversarial-exporter", daemon=True)
        self.thread.start()
        logger.info(
            f"R2 adversarial dataset export enabled bucket={self.bucket} "
            f"prefix={self.prefix} storage_mode={self.storage_mode}"
        )

    def _raise_config_error(self, message: str) -> None:
        logger.warning(message)
        raise RuntimeError(message)

    def miner_storage_key(self, *, miner_uid: int, miner_hotkey: str) -> str:
        """Opaque stable key for latest-mode object paths.

        We need a stable per-miner key to overwrite latest responses, but we do
        not send raw miner uid/hotkey to R2. The R2 secret key is reused as the
        HMAC secret so operators do not need another configuration value.
        """
        secret = self.miner_key_secret or "r2-export-disabled"
        message = f"{self.netuid}:{int(miner_uid)}:{miner_hotkey}".encode("utf-8")
        return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()[:24]

    def object_url(self, key: str) -> str:
        if self.client is not None and self.bucket:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=min(int(C.R2_PRESIGNED_URL_EXPIRES_SECONDS), 604800),
            )
        # Fallback for tests or disabled clients; browser UIs should receive
        # presigned HTTPS URLs when R2 export is configured.
        return f"r2://{self.bucket}/{key}"

    def image_key_for(
        self,
        *,
        image_id: str,
        task_id: str,
        rank: int,
        miner_storage_key: str,
        image_sha256: str,
    ) -> str:
        safe_task_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
        if self.storage_mode == "latest":
            return f"{self.prefix}/latest/{miner_storage_key}.png"
        return f"{self.prefix}/images/{image_id}/{safe_task_id}_rank{int(rank):02d}_{image_sha256[:12]}.png"

    def submit_top_unique(
        self,
        *,
        image_id: str,
        task_id: str,
        block: int,
        model_name: str,
        true_label: str,
        candidates: list[AdversarialExportCandidate],
    ) -> dict[str, str]:
        if not self.enabled or not candidates:
            return {}
        selected: list[tuple[int, AdversarialExportCandidate, str]] = []
        seen_image_hashes: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: -float(item.score)):
            if float(candidate.score) <= 0.0:
                continue
            image_hash = _image_sha256(candidate.perturbed_image_b64)
            if image_hash in seen_image_hashes:
                continue
            seen_image_hashes.add(image_hash)
            selected.append((len(selected) + 1, candidate, image_hash))
        image_url_by_storage_key: dict[str, str] = {}
        failed_uploads = 0
        for rank, candidate, image_hash in selected:
            try:
                self.submit(
                    image_id=image_id,
                    task_id=task_id,
                    block=block,
                    rank=rank,
                    miner_storage_key=candidate.miner_storage_key,
                    model_name=model_name,
                    true_label=true_label,
                    adversarial_label=candidate.adversarial_label,
                    score=candidate.score,
                    norm=candidate.norm,
                    rmse=candidate.rmse,
                    epsilon=candidate.epsilon,
                    ssim=candidate.ssim,
                    psnr_db=candidate.psnr_db,
                    response_time_ms=candidate.response_time_ms,
                    perturbed_image_b64=candidate.perturbed_image_b64,
                    image_sha256=image_hash,
                    sync=True,
                )
                image_url_by_storage_key[candidate.miner_storage_key] = self.object_url(
                    self.image_key_for(
                        image_id=image_id,
                        task_id=task_id,
                        rank=rank,
                        miner_storage_key=candidate.miner_storage_key,
                        image_sha256=image_hash,
                    )
                )
            except Exception as exc:
                failed_uploads += 1
                logger.warning(
                    f"R2 adversarial dataset export failed before leaderboard URL "
                    f"task_id={task_id} image_id={image_id}: {exc}"
                )
        if selected:
            uploaded_count = len(image_url_by_storage_key)
            if failed_uploads:
                logger.warning(
                    f"R2 export completed with failures task_id={task_id} image_id={image_id} "
                    f"uploaded={uploaded_count} failed={failed_uploads} attempted={len(selected)} "
                    f"storage_mode={self.storage_mode}"
                )
            else:
                logger.info(
                    f"R2 export succeeded task_id={task_id} image_id={image_id} "
                    f"uploaded={uploaded_count} storage_mode={self.storage_mode}"
                )
        return image_url_by_storage_key

    def submit(
        self,
        *,
        image_id: str,
        task_id: str,
        block: int,
        rank: int,
        miner_storage_key: str,
        model_name: str,
        true_label: str,
        adversarial_label: str,
        score: float,
        norm: float,
        rmse: float,
        epsilon: float,
        ssim: float,
        psnr_db: float,
        response_time_ms: int,
        perturbed_image_b64: str,
        image_sha256: str | None = None,
        sync: bool = False,
    ) -> None:
        if not self.enabled:
            return
        image_hash = image_sha256 or _image_sha256(perturbed_image_b64)
        created_at = datetime.now(UTC)
        image_key = self.image_key_for(
            image_id=image_id,
            task_id=task_id,
            rank=rank,
            miner_storage_key=miner_storage_key,
            image_sha256=image_hash,
        )
        record = AdversarialExampleRecord(
            image_id=image_id,
            hf_row=_parse_hf_row(image_id),
            task_id=task_id,
            block=int(block),
            netuid=self.netuid,
            validator_hotkey=self.validator_hotkey,
            model_name=model_name,
            true_label=true_label,
            adversarial_label=adversarial_label,
            score=float(score),
            norm=float(norm),
            rmse=float(rmse),
            epsilon=float(epsilon),
            ssim=float(ssim),
            psnr_db=float(psnr_db),
            response_time_ms=int(response_time_ms),
            image_sha256=image_hash,
            image_object_key=image_key,
            created_at=created_at.isoformat(),
            storage_mode=self.storage_mode,
            miner_storage_key=miner_storage_key if self.storage_mode == "latest" else None,
        )
        if sync:
            self._upload(record=record, perturbed_image_b64=perturbed_image_b64)
            return
        try:
            self.queue.put_nowait((record, perturbed_image_b64))
        except queue.Full:
            logger.warning("R2 adversarial dataset export queue is full; dropping example.")

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
            item = self.queue.get()
            if item is None:
                return
            record, perturbed_image_b64 = item
            try:
                self._upload(record=record, perturbed_image_b64=perturbed_image_b64)
            except Exception as exc:
                logger.warning(f"R2 adversarial dataset export failed image_id={record.image_id}: {exc}")
            finally:
                self.queue.task_done()

    def _upload(self, *, record: AdversarialExampleRecord, perturbed_image_b64: str) -> None:
        if self.client is None:
            return
        image_bytes = _normalized_png_bytes(perturbed_image_b64)
        self.client.put_object(
            Bucket=self.bucket,
            Key=record.image_object_key,
            Body=image_bytes,
            ContentType="image/png",
            CacheControl="no-store, max-age=0",
        )

        created = datetime.fromisoformat(record.created_at)
        if record.storage_mode == "latest" and record.miner_storage_key:
            manifest_key = f"{self.prefix}/latest/{record.miner_storage_key}.jsonl"
        else:
            manifest_key = (
                f"{self.prefix}/manifests/date={created:%Y-%m-%d}/hour={created:%H}/"
                f"{self.run_id}_{record.task_id}_{record.image_sha256[:12]}.jsonl"
            )
        body = (json.dumps(asdict(record), sort_keys=True) + "\n").encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=body,
            ContentType="application/x-ndjson",
        )
