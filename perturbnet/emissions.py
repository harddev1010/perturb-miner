from __future__ import annotations

from typing import Sequence


def ranked_emission_shares(ranked_uids: Sequence[int]) -> dict[int, float]:
    """70 / 15 / 15 emission schedule with rank-weighted tail.

    Ranks 3+ split the final 15% by descending rank weight, so rank 3 gets
    more than rank 4, rank 4 gets more than rank 5, etc.
    """
    uids = [int(uid) for uid in ranked_uids]
    if not uids:
        return {}
    if len(uids) == 1:
        return {uids[0]: 1.0}
    if len(uids) == 2:
        return {uids[0]: 0.70, uids[1]: 0.30}

    shares = {uids[0]: 0.70, uids[1]: 0.15}
    tail = uids[2:]
    rank_weights = list(range(len(tail), 0, -1))
    total_rank_weight = float(sum(rank_weights))
    shares.update({uid: 0.15 * weight / total_rank_weight for uid, weight in zip(tail, rank_weights)})
    return shares
