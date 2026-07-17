#!/usr/bin/env bash
# run_scorer.sh — set up the venv + env and run the offline scoring harness (scripts/score_miner.py).
#
# Reuses scripts/miner.env so the scorer fetches challenges from the SAME cloud store (R2) the miner
# uploads to — PERTURB_STORAGE_* and NETUID come straight from there. Every argument is forwarded to
# score_miner.py, e.g.:
#   bash ./scripts/run_scorer.sh                               # latest 200 from the cloud store
#   bash ./scripts/run_scorer.sh --challenge-count 50 --attack-timeout 10 --device cuda
#   bash ./scripts/run_scorer.sh --dir /path/to/records        # score a local folder instead
#   bash ./scripts/run_scorer.sh --challenge-count 200 > report.txt   # report is on stdout
#
# By default it just activates the existing .venv (fast). Set SCORER_INSTALL=1 to (re)install deps.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"
ENV_FILE="$SCRIPT_DIR/miner.env"

# Load the miner env (storage creds + NETUID) so cloud-mode fetch works with zero extra config.
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "Note: $ENV_FILE not found; relying on already-exported PERTURB_STORAGE_*/NETUID env (or --dir)." >&2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Create the venv on first run (and force a dependency install for it).
if [[ ! -d ".venv" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
  SCORER_INSTALL=1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Deps are usually already installed (the miner runs in the same venv); only (re)install on request.
if [[ "${SCORER_INSTALL:-0}" == "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install -e .
fi

echo "Running scorer: score_miner.py $*" >&2
exec python scripts/score_miner.py "$@"
