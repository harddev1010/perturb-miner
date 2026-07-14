from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from perturbnet import constants as C
from perturbnet.api_client import post_task
from perturbnet.image_io import decode_image_b64
from perturbnet.model import load_efficientnet_v2_l, normalize_prediction_label, predict_label
from perturbnet.storage_uploader import ImageStorageUploader


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    image_url: str
    image_id: str
    true_label: str


class TaskGenerator:
    def __init__(self, *, state_path: str | os.PathLike[str] = "task_generator_state.json") -> None:
        self.state_path = Path(state_path)
        self.system_random = random.SystemRandom()
        self.order_seed = 0
        self.order_cursor = 0
        self.order_fingerprint = ""
        self.order_epoch = 0
        self._order_cache: list[str] = []
        self._order_cache_key: tuple[str, int] = ("", 0)
        self._dataset: Any | None = None
        self._index: list[tuple[str, int]] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_efficientnet_v2_l(self.device)
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.order_seed = max(0, int(state.get("order_seed", 0)))
        self.order_cursor = max(0, int(state.get("order_cursor", 0)))
        self.order_fingerprint = str(state.get("order_fingerprint", "") or "")
        self.order_epoch = max(0, int(state.get("order_epoch", 0)))

    def _save_state(self) -> None:
        payload = {
            "order_seed": int(self.order_seed),
            "order_cursor": int(self.order_cursor),
            "order_fingerprint": self.order_fingerprint,
            "order_epoch": int(self.order_epoch),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    def _load_index(self) -> list[tuple[str, int]]:
        if self._index:
            return self._index
        from perturbnet.imagenet100_bootstrap import imagenet100_dataset_version, load_imagenet100

        dataset = load_imagenet100(repo_id=C.IMAGENET100_REPO_ID, split=C.IMAGENET100_SPLIT)
        version = imagenet100_dataset_version(dataset=dataset, repo_id=C.IMAGENET100_REPO_ID, split=C.IMAGENET100_SPLIT)
        total_rows = int(dataset.num_rows)
        if total_rows <= 0:
            raise RuntimeError("ImageNet-100 dataset is empty.")
        self._dataset = dataset
        self._index = [(f"hf-{version}-{row:07d}", row) for row in range(total_rows)]
        return self._index

    def _fingerprint(self, image_ids: Sequence[str]) -> str:
        digest = hashlib.sha256()
        for image_id in sorted(image_ids):
            digest.update(image_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _reset_order(self, *, fingerprint: str, epoch: int) -> None:
        self.order_seed = self.system_random.randrange(1, 2**63)
        self.order_cursor = 0
        self.order_fingerprint = fingerprint
        self.order_epoch = epoch
        self._order_cache = []
        self._order_cache_key = ("", 0)

    def _ensure_order(self, image_ids: Sequence[str]) -> None:
        fingerprint = self._fingerprint(image_ids)
        if self.order_fingerprint != fingerprint or self.order_seed <= 0:
            self._reset_order(fingerprint=fingerprint, epoch=0)
        elif self.order_cursor >= len(image_ids):
            self._reset_order(fingerprint=fingerprint, epoch=self.order_epoch + 1)

        cache_key = (self.order_fingerprint, int(self.order_seed))
        if self._order_cache_key != cache_key or not self._order_cache:
            order = sorted(image_ids)
            random.Random(int(self.order_seed)).shuffle(order)
            self._order_cache = order
            self._order_cache_key = cache_key

    def _image_bytes(self, row: int) -> bytes:
        if self._dataset is None:
            raise RuntimeError("ImageNet-100 dataset is not loaded.")
        example = self._dataset[int(row)]
        image = example.get("image")
        if image is None:
            raise ValueError(f"ImageNet-100 row {row} has no image payload.")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    def _sample_image(self) -> tuple[str, str]:
        index = self._load_index()
        source_by_id = {image_id: source for image_id, source in index}
        image_ids = list(source_by_id.keys())
        self._ensure_order(image_ids)
        image_id = self._order_cache[self.order_cursor]
        raw = self._image_bytes(source_by_id[image_id])
        self.order_cursor += 1
        self._save_state()
        return image_id, base64.b64encode(raw).decode("utf-8")

    def generate(self) -> tuple[str, str, str]:
        for _ in range(C.MAX_CHALLENGE_ATTEMPTS):
            image_id, image_b64 = self._sample_image()
            image = decode_image_b64(image_b64).to(self.device)
            label = normalize_prediction_label(predict_label(self.model, image))
            if label:
                return image_id, image_b64, label
        raise RuntimeError("Unable to generate task after max attempts.")


def generate_and_publish_task(
    *,
    state_path: str | os.PathLike[str] = "task_generator_state.json",
    status: str = "open",
) -> GeneratedTask:
    generator = TaskGenerator(state_path=state_path)
    image_id, image_b64, label = generator.generate()
    task_id = f"{int(time.time())}-{image_id}"
    exporter = ImageStorageUploader(
        run_id="task-generator",
        netuid=int(os.getenv("NETUID", "26")),
        uploader_hotkey="task-generator",
    )
    image_key = f"{C.STORAGE_PREFIX.strip().strip('/')}/tasks/current.png"
    image_url = exporter.upload_image_b64(key=image_key, image_b64=image_b64)
    post_task(
        base_url=C.PERTURB_API_BASE_URL,
        api_key=C.PERTURB_API_KEY,
        task_id=task_id,
        image_url=image_url,
        status=status,
        timeout_seconds=C.PERTURB_API_TIMEOUT_SECONDS,
    )
    return GeneratedTask(task_id=task_id, image_url=image_url, image_id=image_id, true_label=label)
