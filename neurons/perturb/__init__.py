"""Perturb attack engine: the batched flip-first pipeline that powers the miner.

Public surface is just `perturb` (same signature the miner has always called). Internals are split
into constants.py (env-tunable knobs), utils.py (shared primitives), and perturb.py (algorithms +
orchestrators).
"""

from .perturb import perturb

__all__ = ["perturb"]
