#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROLE="${1:-validator}"
ROLE="$(echo "$ROLE" | tr '[:upper:]' '[:lower:]')"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ and rerun."
  exit 1
fi

if [[ "$ROLE" != "miner" && "$ROLE" != "validator" ]]; then
  echo "Usage: bash ./scripts/setup_common.sh [miner|validator]"
  exit 1
fi

# Both run_validator.sh and run_miner.sh default to PM2 process management.
if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js (which includes npm) and rerun."
  echo "macOS: brew install node"
  echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y nodejs npm"
  exit 1
fi

echo "Installing PM2..."
npm install -g pm2

echo "Creating/updating virtual environment..."
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install bittensor bittensor-cli
python -m pip install -e .

if [[ "$ROLE" == "validator" ]]; then
  VALIDATOR_ENV_FILE="$ROOT_DIR/scripts/validator.env"
  if [[ -f "$VALIDATOR_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$VALIDATOR_ENV_FILE"
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
  fi
  echo "Preparing ImageNet-100 challenge dataset..."
  python scripts/bootstrap_imagenet100.py
fi

echo "Setup complete for role: ${ROLE}"
