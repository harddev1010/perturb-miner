from __future__ import annotations

import base64
import hashlib
import hmac
import io
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
        if self.backend == "hippius":
            from botocore.config import Config as BotoConfig

            client_kwargs["config"] = BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"})
        self.client = boto3.client("s3", **client_kwargs)
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
