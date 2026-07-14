from __future__ import annotations

import time


def next_task_boundary(now: float | None = None, cadence_seconds: int = 120) -> float:
    """Return the next wall-clock cadence boundary, e.g. even minute at :00."""
    current = time.time() if now is None else float(now)
    cadence = max(1, int(cadence_seconds))
    return (int(current // cadence) + 1) * cadence


def seconds_until_next_task_boundary(cadence_seconds: int = 120) -> float:
    return max(0.0, next_task_boundary(cadence_seconds=cadence_seconds) - time.time())


def sleep_until_next_task_boundary(cadence_seconds: int = 120) -> float:
    sleep_seconds = seconds_until_next_task_boundary(cadence_seconds=cadence_seconds)
    if sleep_seconds > 0.0:
        time.sleep(sleep_seconds)
    return sleep_seconds
