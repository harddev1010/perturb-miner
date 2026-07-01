from __future__ import annotations

from typing import Any


VALIDATOR_STAKE_THRESHOLD = 100_000.0


def _sequence_value(values: Any, uid: int, default: Any = None) -> Any:
    try:
        return values[uid]
    except Exception:
        return default


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "tao"):
        return float(value.tao)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _as_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def is_validator_neuron(metagraph: Any, uid: int) -> bool:
    permits = getattr(metagraph, "validator_permit", getattr(metagraph, "validator_permits", []))
    stakes = getattr(metagraph, "stake", getattr(metagraph, "S", []))
    has_permit = _as_bool(_sequence_value(permits, uid, False))
    stake = _as_float(_sequence_value(stakes, uid, 0.0))
    return has_permit and stake > VALIDATOR_STAKE_THRESHOLD


def count_miners(metagraph: Any) -> int:
    total_neurons = int(getattr(metagraph, "n", 0))
    validators = sum(1 for uid in range(total_neurons) if is_validator_neuron(metagraph, uid))
    return max(0, total_neurons - validators)


def miner_incentive(metagraph: Any, uid: int) -> float:
    incentives = getattr(metagraph, "I", getattr(metagraph, "incentive", []))
    return _as_float(_sequence_value(incentives, uid, 0.0))
