from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
from typing import Any

from perturbnet import constants as C

logger = logging.getLogger(__name__)


def _normalized_png_bytes(image_b64: str) -> bytes:
    from PIL import Image

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG", optimize=True)
        return output.getvalue()


class ImageStorageUploader:
    """S3-compatible image uploader for task and response images."""

    def __init__(
        self,
        *,
        run_id: str,
        netuid: int,
        uploader_hotkey: str = "",
    ) -> None:
        self.run_id = run_id
        self.netuid = int(netuid)
        self.uploader_hotkey = uploader_hotkey
        self.backend = str(C.STORAGE_BACKEND).strip().lower() or "hippius"
        if self.backend not in {"r2", "hippius"}:
            self._raise_config_error(f"Invalid PERTURB_STORAGE_BACKEND={self.backend!r}; expected 'r2' or 'hippius'.")
        self.bucket = C.STORAGE_BUCKET
        self.prefix = C.STORAGE_PREFIX.strip().strip("/")
        self.miner_key_secret = ""
        self.client: Any | None = None
        # Fail-fast client for best-effort writes (challenge JSON). Falls back to self.client if unset.
        self._besteffort_client: Any | None = None

        if not self.bucket:
            self._raise_config_error(f"{self.backend} storage requires PERTURB_STORAGE_BUCKET.")
        try:
            import boto3  # type: ignore[reportMissingImports]
        except Exception as exc:
            self._raise_config_error(f"{self.backend} storage requires boto3: {exc}")

        endpoint_url = self._endpoint_url_for_backend()
        access_key = C.STORAGE_ACCESS_KEY_ID
        secret_key = C.STORAGE_SECRET_ACCESS_KEY
        if not endpoint_url or not access_key or not secret_key:
            self._raise_config_error(
                f"{self.backend} storage requires endpoint URL, access key ID, and secret access key."
            )
        self.miner_key_secret = secret_key

        client_kwargs = {
            "endpoint_url": endpoint_url,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": self._region_for_backend(),
        }
        # Both hippius and R2 are S3-compatible via SigV4 + path-style addressing. Path-style avoids
        # virtual-hosted DNS / presigned-URL pitfalls (e.g. bucket names with dots) and is the mode
        # Cloudflare R2 recommends, so the presigned response URL the validator fetches always resolves.
        from botocore.config import Config as BotoConfig

        base_config = BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"})
        client_kwargs["config"] = base_config
        self.client = boto3.client("s3", **client_kwargs)

        # A second, fail-fast client for best-effort writes (challenge JSON, uploaded after submit):
        # bounded retries + short timeouts so a transient backend blip can't stall the miner loop for
        # the full default retry budget and delay the next task. Never break startup — on any failure
        # fall back to self.client (upload_json handles None), so behaviour is at worst unchanged.
        try:
            fast_config = base_config.merge(
                BotoConfig(retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=4, read_timeout=6)
            )
            self._besteffort_client = boto3.client("s3", **{**client_kwargs, "config": fast_config})
        except Exception as exc:
            logger.warning(f"Fail-fast storage client unavailable, using default client for best-effort writes: {exc}")
            self._besteffort_client = None
        logger.info(f"Image storage enabled backend={self.backend} bucket={self.bucket} prefix={self.prefix}")

    def _raise_config_error(self, message: str) -> None:
        logger.warning(message)
        raise RuntimeError(message)

    def _endpoint_url_for_backend(self) -> str:
        if C.STORAGE_ENDPOINT_URL:
            return C.STORAGE_ENDPOINT_URL
        if self.backend == "hippius":
            return "https://s3.hippius.com"
        return ""

    def _region_for_backend(self) -> str:
        if C.STORAGE_REGION:
            return C.STORAGE_REGION
        if self.backend == "hippius":
            return "decentralized"
        return "auto"

    def miner_storage_key(self, *, miner_uid: int, miner_hotkey: str) -> str:
        secret = self.miner_key_secret or "response-storage-disabled"
        message = f"{self.netuid}:{int(miner_uid)}:{miner_hotkey}".encode("utf-8")
        return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()[:24]

    def object_url(self, key: str) -> str:
        if self.client is not None and self.bucket:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=min(int(C.STORAGE_PRESIGNED_URL_EXPIRES_SECONDS), 604800),
            )
        return f"s3://{self.bucket}/{key}"

    def upload_image_b64(self, *, key: str, image_b64: str) -> str:
        if self.client is None:
            raise RuntimeError("Image storage is not configured.")
        image_bytes = _normalized_png_bytes(image_b64)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=image_bytes,
            ContentType="image/png",
            CacheControl="no-store, max-age=0",
        )
        return self.object_url(key)

    def upload_json(self, *, key: str, obj: Any) -> str:
        """Store a JSON document (challenge record + miner score/time) at `key`. Returns its URL.

        Uses the fail-fast client (bounded retries + short timeouts): this is a best-effort write done
        after the submit, and must never block the miner loop while the backend is degraded."""
        client = self._besteffort_client or self.client
        if client is None:
            raise RuntimeError("Image storage is not configured.")
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="no-store, max-age=0",
        )
        return self.object_url(key)

    def list_objects(self, *, prefix: str) -> list[dict[str, Any]]:
        """List every object under `prefix` as {key, last_modified, size}, following pagination."""
        if self.client is None:
            raise RuntimeError("Image storage is not configured.")
        items: list[dict[str, Any]] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for entry in page.get("Contents", []) or []:
                items.append(
                    {
                        "key": entry.get("Key", ""),
                        "last_modified": entry.get("LastModified"),
                        "size": int(entry.get("Size", 0)),
                    }
                )
        return items

    def download_bytes(self, *, key: str) -> bytes:
        if self.client is None:
            raise RuntimeError("Image storage is not configured.")
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def download_json(self, *, key: str) -> Any:
        return json.loads(self.download_bytes(key=key).decode("utf-8"))
