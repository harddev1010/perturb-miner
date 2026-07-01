from __future__ import annotations

from typing import Any, Sequence

from perturbnet import constants as C
from perturbnet.leaderboard_reporter import LeaderboardMinerResult, LeaderboardNetworkMetrics, LeaderboardReport


def update_score_histories(histories: list[list[float]], uids: Sequence[int], rewards: Sequence[float], window: int) -> None:
    for uid, reward in zip(uids, rewards):
        histories[uid].append(float(reward))
        histories[uid] = histories[uid][-int(window):]


def avg_score(histories: list[list[float]], uid: int, window: int) -> float:
    history = histories[uid][-int(window):]
    if not history:
        return 0.0
    return float(sum(history) / len(history))


def result_status(result: Any) -> str:
    if result.reason == "success":
        return "Valid"
    if result.reason == "duplicate_response_fastest_wins":
        return "Duplicate"
    if result.reason == "response_missing_or_status_error":
        return "Inactive"
    if result.reason == "leaderboard_unavailable":
        return "Unavailable"
    return "Invalid"


def network_metrics(
    *,
    total_miners: int,
    available_miners: int,
    results_by_uid: Sequence[tuple[int, Any]],
) -> LeaderboardNetworkMetrics:
    successful_results = [result for _, result in results_by_uid if result_status(result) == "Valid"]
    success_count = len(successful_results)
    if success_count == 0:
        return LeaderboardNetworkMetrics(
            total_miners=int(total_miners),
            available_miners=int(available_miners),
            avg_score=0.0,
            avg_rmse=0.0,
            avg_norm=0.0,
            success_count=0,
        )
    return LeaderboardNetworkMetrics(
        total_miners=int(total_miners),
        available_miners=int(available_miners),
        avg_score=float(sum(result.score for result in successful_results) / success_count),
        avg_rmse=float(sum(result.rmse for result in successful_results) / success_count),
        avg_norm=float(sum(result.norm for result in successful_results) / success_count),
        success_count=int(success_count),
    )


def build_report(
    *,
    task_id: str,
    validator_hotkey: str,
    total_miners: int,
    available_miners: int,
    hotkeys: Sequence[str],
    coldkeys: Sequence[str],
    incentives_by_uid: dict[int, float],
    score_histories: list[list[float]],
    avg_window: int,
    results_by_uid: Sequence[tuple[int, Any]],
    image_url_by_uid: dict[int, str],
) -> LeaderboardReport:
    miners: list[LeaderboardMinerResult] = []
    for uid, result in results_by_uid:
        miners.append(
            LeaderboardMinerResult(
                uid=int(uid),
                hotkey=str(hotkeys[uid]) if uid < len(hotkeys) else "",
                coldkey=str(coldkeys[uid]) if uid < len(coldkeys) else "",
                incentive=float(incentives_by_uid.get(uid, 0.0)),
                avg_score=avg_score(score_histories, uid, avg_window),
                last_score=float(result.score),
                rmse=float(result.rmse),
                norm=float(result.norm),
                result=result_status(result),
                image_url=image_url_by_uid.get(uid) or C.LEADERBOARD_NO_IMAGE_URL,
            )
        )
    return LeaderboardReport(
        task_id=task_id,
        validator_hotkey=validator_hotkey,
        network=network_metrics(
            total_miners=total_miners,
            available_miners=available_miners,
            results_by_uid=results_by_uid,
        ),
        miners=miners,
    )
