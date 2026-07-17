"""Challenge store on the S3-compatible cloud backend (R2).

The miner (neurons/miner.py) writes one record per task it processes to
``{CHALLENGE_PREFIX}/<task_id>.json`` — the clean image it attacked plus the score/time it achieved.
The offline harness (scripts/score_miner.py) reads the LATEST N of those records back and replays the
current perturb() engine against the exact same images. Keeping the key convention + record shape in
one module is what keeps the writer and reader in sync.

Storage is the SAME backend the miner uploads its responses to (default R2, backend/creds from
perturbnet.constants). The miner uploads the record AFTER it submits its response, on a best-effort
fail-fast client, so persisting a challenge never delays or breaks the submission. Only the miner
writes here (store_challenge); the harness is strictly read-only (list/load).
"""

from __future__ import annotations

import os
import time
from typing import Any

from perturbnet import constants as _C

# Where challenge records live in the bucket. Default: perturb/attack-challenges/<id>.json — nested
# under the storage prefix so it shares the same bucket/credential scope as the miner's response
# uploads. Override with PERTURB_CHALLENGE_PREFIX.
_DEFAULT_CHALLENGE_PREFIX = f"{_C.STORAGE_PREFIX.strip().strip('/')}/attack-challenges"
CHALLENGE_PREFIX = (os.getenv("PERTURB_CHALLENGE_PREFIX") or _DEFAULT_CHALLENGE_PREFIX).strip().strip("/")

# Record schema version, so a future reader can tell an old record from a new one.
CHALLENGE_RECORD_VERSION = 1


def safe_task_id(task_id: str) -> str:
    """Filesystem/URL-safe task id for the object key (mirrors miner._upload_response sanitisation)."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(task_id))


def challenge_key(task_id: str) -> str:
    """Object key for a task's challenge record: ``perturb/attack-challenges/<safe_task_id>.json``."""
    return f"{CHALLENGE_PREFIX}/{safe_task_id(task_id)}.json"


def build_record(
    *,
    task_id: str,
    image_b64: str,
    true_label: str,
    target_index: int,
    epsilon: float,
    min_delta: float,
    image_url: str = "",
    created_epoch: float | None = None,
    metrics: dict[str, Any] | None = None,
    time_spent_seconds: float = 0.0,
) -> dict[str, Any]:
    """Assemble the JSON record: the challenge (enough to replay it) + the miner's own score/time.

    `image_b64` is the PNG the miner actually attacked (the same bytes the validator scores), so the
    harness can reconstruct a byte-identical clean image. `metrics` is the miner's per-task result
    (see miner._attack_image): score, pert_score, margin, linf, rmse, k, flip flags, changed counts.
    """
    return {
        "version": CHALLENGE_RECORD_VERSION,
        "task_id": str(task_id),
        "image_url": str(image_url),
        "image_b64": str(image_b64),
        "true_label": str(true_label),
        "target_index": int(target_index),
        "epsilon": float(epsilon),
        "min_delta": float(min_delta),
        "created_epoch": float(created_epoch) if created_epoch is not None else None,
        "stored_epoch": time.time(),
        "miner": dict(metrics or {}),
        "time_spent_seconds": float(time_spent_seconds),
    }


def store_challenge(uploader: Any, record: dict[str, Any]) -> str:
    """Upload one challenge record to the cloud store. Returns the object URL."""
    return uploader.upload_json(key=challenge_key(record["task_id"]), obj=record)


def list_latest_challenges(uploader: Any, count: int) -> list[dict[str, Any]]:
    """The `count` most-recently-stored challenge objects (newest first) as {key, last_modified, size}.

    Sorted by the object's LastModified so it works regardless of task-id format or host clock skew.
    """
    objects = [o for o in uploader.list_objects(prefix=f"{CHALLENGE_PREFIX}/") if o["key"].endswith(".json")]

    def _sort_key(obj: dict[str, Any]):
        modified = obj.get("last_modified")
        # LastModified is a tz-aware datetime; fall back to the key (task ids lead with an epoch) if absent.
        return (modified.timestamp() if modified is not None else 0.0, obj.get("key", ""))

    objects.sort(key=_sort_key, reverse=True)
    return objects[: max(0, int(count))]


def load_challenge(uploader: Any, key: str) -> dict[str, Any]:
    """Download and parse a single challenge record by object key."""
    return uploader.download_json(key=key)
