from __future__ import annotations

from typing import Any, Sequence


DUPLICATE_RESPONSE_REASON = "duplicate_response"


def zero_duplicate_responses(
    *,
    results_by_uid: Sequence[tuple[int, Any]],
    response_hash_by_uid: dict[int, str],
) -> None:
    """Zero all duplicated responses.

    Duplicate equality is exact-match on response content hash, calculated from
    the base64-decoded submitted image bytes by the caller. Only positive-score
    responses participate.
    """
    result_by_uid = {uid: result for uid, result in results_by_uid}
    grouped_uids_by_hash: dict[str, list[int]] = {}
    for uid, result in results_by_uid:
        if float(result.score) <= 0.0:
            continue
        response_hash = response_hash_by_uid.get(uid)
        if response_hash:
            grouped_uids_by_hash.setdefault(response_hash, []).append(uid)

    for duplicate_group in grouped_uids_by_hash.values():
        if len(duplicate_group) <= 1:
            continue
        for uid in duplicate_group:
            result_by_uid[uid].score = 0.0
            result_by_uid[uid].reason = DUPLICATE_RESPONSE_REASON
