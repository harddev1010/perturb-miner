from __future__ import annotations

from typing import Any, Sequence


DUPLICATE_RESPONSE_REASON = "duplicate_response_fastest_wins"


def apply_fastest_wins(
    *,
    results_by_uid: Sequence[tuple[int, Any]],
    response_hash_by_uid: dict[int, str],
) -> None:
    """Zero slower duplicate responses, keeping only the fastest scorer.

    Duplicate equality is exact-match on response content hash, calculated from
    the base64-decoded submitted image bytes by the caller. Only positive-score
    responses participate. If latencies tie, lower UID wins deterministically.
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
        winner_uid = min(
            duplicate_group,
            key=lambda uid: (int(result_by_uid[uid].response_time_ms), int(uid)),
        )
        for uid in duplicate_group:
            if uid == winner_uid:
                continue
            result_by_uid[uid].score = 0.0
            result_by_uid[uid].reason = DUPLICATE_RESPONSE_REASON
