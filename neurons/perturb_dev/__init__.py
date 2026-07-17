"""Perturb attack engine (DEV duplicate).

A standalone copy of the perturb engine used by scripts/score_miner.py for experimentation, so the
production neurons/perturb/ engine the miner runs stays untouched. Public surface is just `perturb`
(same signature the miner calls). utils.py / calibration.py are exact copies of the production ones;
perturb.py + constants.py are the cleaned dev surface (currently: one-shot + binary-search linear
flip finder).
"""

from .perturb import perturb

__all__ = ["perturb"]
