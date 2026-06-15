# Perturb Subnet

Perturb is a decentralized adversarial robustness network built on Bittensor. Miners compete to find adversarial examples — imperceptible input perturbations that cause state-of-the-art image classifiers to fail — while validators construct challenges from real images, verify every response with mathematical precision, and reward the best attackers with on-chain emissions.

Modern AI models achieve remarkable accuracy on clean data yet remain catastrophically brittle: a perturbation invisible to any human observer can make a production classifier misclassify a tumor scan, a stop sign, or a fraudulent transaction. The tooling to systematically discover these vulnerabilities is fragmented, expensive, and static. Perturb replaces it with a financially incentivized, continuously improving adversarial testing network — every day miners compete, attacks get stronger and the network's outputs get more valuable.

The network produces two commercially valuable outputs:

- **Adversarial training dataset** — a continuously growing corpus of verified adversarial examples, the raw material for adversarial training (the most effective known defense)
- **Model robustness certificates** — on-chain, auditable proof of adversarial evaluation, relevant to EU AI Act conformity and enterprise AI procurement

Why Bittensor: finding an adversarial example is computationally hard, but verifying one is trivially cheap — run the model, compare the prediction, measure the perturbation norm. This verification asymmetry makes the incentive mechanism clean, objective, and manipulation-resistant, while TAO emissions drive a level of continuous attack research no salaried red team can match.

Read the full vision and roadmap in the [Perturb whitepaper](https://www.perturbai.io/whitepaper).

This repository provides:

- validator node implementation (`neurons/validator.py`)
- baseline miner implementation (`neurons/miner.py`)
- one-command launchers for validator and miner

## Architecture

### Validator responsibilities

- Sample challenge images from the full ImageNet-100 train split (~126k images, auto-downloaded)
- Run fixed classifier (`EfficientNetV2-L`) on pulled image
- Build and broadcast `AttackChallenge` synapse to selected miners
- Verify miner responses and compute rewards
- Maintain rolling histories and set on-chain weights periodically

### Miner responsibilities

- Receive `AttackChallenge` over Axon
- Run baseline PGD-style attack
- Return only `perturbed_image_b64`
- Let validator handle all authoritative verification and scoring

### Challenge lifecycle

1. Validator samples an image from a persisted random traversal over the full ImageNet-100 train split (no duplicates until all ~126k images are used)
2. Validator runs `EfficientNetV2-L` and gets exact model label string
3. Validator creates challenge where `true_label` is the exact EfficientNet label
4. Validator sends challenge to sampled miners and scores returned perturbations

## Hardware and System Requirements

### Miner

- Minimum: 4 vCPU, 16 GB RAM, 50 GB SSD, stable 20+ Mbps network
- Recommended: 8 vCPU, 32 GB RAM, NVIDIA GPU with 8+ GB VRAM, 100+ GB SSD

### Validator

- Minimum: 8 vCPU, 32 GB RAM, NVIDIA GPU with 12+ GB VRAM, 100 GB SSD
- Recommended: 16 vCPU, 64 GB RAM, NVIDIA GPU with 24+ GB VRAM, 200 GB SSD

### Common software prerequisites

- Python 3.10+
- Node.js 18+ (includes `npm`) for PM2 installation
- `pip` and virtualenv support (`python -m venv`)
- OS build tools needed by Python wheels
- For GPU usage: correct NVIDIA driver + CUDA stack compatible with installed PyTorch

## Common Installation (Do Once)

Run role-specific setup once before starting nodes:

```bash
git clone https://github.com/0xsigurd/Perturb
cd Perturb
```

For miner setup:

```bash
bash ./scripts/setup_common.sh miner
```

For validator setup:

```bash
bash ./scripts/setup_common.sh validator
```

`setup_common.sh` behavior by role:

- both roles: install PM2, create `.venv`, install Python/Bittensor dependencies
- `validator`: additionally bootstraps the ImageNet-100 challenge cache

If `npm: command not found`, install Node.js first, then rerun:

macOS (Homebrew):

```bash
brew install node
node --version
npm --version
bash ./scripts/setup_common.sh validator
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y nodejs npm
node --version
npm --version
bash ./scripts/setup_common.sh validator
```

## Installation and Setup (Validator Side)

This section is specifically for validator operators.

### 1) ImageNet-100 challenge data

Validator setup automatically downloads the **full ImageNet-100 train split** (~126k images, ~8 GB download, ~17 GB on disk including the converted Arrow cache) from Hugging Face (`clane9/imagenet-100`) into the local Hugging Face cache. The download happens once during `bash ./scripts/setup_common.sh validator` and is re-checked by `bash ./scripts/run_validator.sh`; subsequent starts reuse the cache instantly. Validators traverse all images in a persisted random order with no repeats until the entire split has been used, then reshuffle for the next epoch.
Optionally set `HF_TOKEN` in `scripts/validator.env` for a faster first download.

### 2) Configure validator runtime

Create validator env:

```bash
cp scripts/validator.env.example scripts/validator.env
```

Edit required fields in `scripts/validator.env`:

- `WALLET_NAME`
- `WALLET_HOTKEY`
- `NETUID`
- `NETWORK`

Important validator-specific fields:

- `HF_TOKEN` (optional in `scripts/validator.env`, speeds up the one-time ImageNet-100 download)
- `PERTURB_K_MINERS`
- `PERTURB_HISTORY_SIZE`
- `PERTURB_MIN_PROCESSED_COUNT`
- `PERTURB_MIN_LINF_DELTA`
- `PERTURB_MAX_LINF_DELTA`
- `PERTURB_WANDB_ENABLED` (`true` to enable validator metrics logging to Weights & Biases)
- `PERTURB_WANDB_PROJECT`, `PERTURB_WANDB_ENTITY`, `PERTURB_WANDB_RUN_NAME`, `PERTURB_WANDB_MODE`
- `PERTURB_WANDB_LOG_CONSOLE` (`true` to forward validator console logs to W&B as well)
- `LOG_LEVEL` (`DEBUG` default, set `INFO`/`WARNING`/`ERROR` if you want quieter logs)

### 3) Start validator

```bash
bash ./scripts/run_validator.sh
```

Expected log behavior:

- challenge generation messages
- miner selection messages
- per-miner score logs
- periodic `set_weights` attempts

### 4) Validator-side notes

- Challenge generation does not depend on external image APIs or LLM verification.
- ImageNet-100 selection walks the entire train split (~126k images) once in random order before reshuffling for the next epoch; the order, cursor, and epoch persist across restarts so no image repeats within an epoch.

## Installation and Setup (Miner Side)

This section is specifically for miner operators.

### 1) Configure miner runtime

Create miner env:

```bash
cp scripts/miner.env.example scripts/miner.env
```

Edit required fields in `scripts/miner.env`:

- `WALLET_NAME`
- `WALLET_HOTKEY`
- `NETUID`
- `NETWORK`

Optional:

- `PYTHON_BIN`
- `LOG_LEVEL` (`DEBUG` default, set `INFO`/`WARNING`/`ERROR` if you want quieter logs)
- `MINER_EXTRA_ARGS`

### 2) Start miner

```bash
bash ./scripts/run_miner.sh
```

Expected log behavior:

- `Serving miner axon...`
- `Miner started. Waiting for validator queries.`

### 3) Miner-side notes

- Baseline miner is intentionally simple; competitive miners should optimize attack logic.
- Validators handle all challenge verification and scoring.

## API and Protocol Contracts

### ImageNet-100 input contract (validator challenge source)

- The source is the full `clane9/imagenet-100` train split (~126k images), fixed in `perturbnet/constants.py`, downloaded once into the local Hugging Face cache (~8 GB download, ~17 GB on disk) and accessed by row index at runtime.
- Optional: set `HF_TOKEN` in `scripts/validator.env` for faster, higher-rate-limit downloads from Hugging Face.
- Validator persists the traversal seed, cursor, dataset fingerprint, and epoch in state; the shuffled order is rebuilt deterministically from the seed, so restarts/resumes continue the traversal without duplicate selections until the full split is exhausted.
- Validator converts image bytes to base64 internally.
- The model-predicted EfficientNet label becomes `true_label`.

### Synapse contract (`AttackChallenge`)

Key fields sent to miners:

- `task_id`
- `model_name` (fixed `EfficientNetV2-L`)
- `clean_image_b64`
- `true_label` (exact EfficientNet class label)
- `epsilon`, `norm_type`, `min_delta`, `timeout_seconds`

Miner response field:

- `perturbed_image_b64`

## Scoring and Weighting

Per-response score (if verification passes):

- Hard gates:
  - `min_linf_delta <= norm <= min(epsilon, max_linf_delta)`
  - `ssim(clean, adv) >= min_ssim`
  - `psnr_db(clean, adv) >= min_psnr_db`
  - predicted label must differ from the original label
- `linf_ratio = clamp((norm - min_linf_delta) / (min(epsilon, max_linf_delta) - min_linf_delta), 0, 1)`
- `rmse_ratio = clamp(rmse / min(epsilon, max_linf_delta), 0, 1)`
- `linf_score = (1 - linf_ratio)^2`
- `rmse_score = (1 - rmse_ratio)^2`
- `perturbation_score = weighted_avg(linf_score, rmse_score)` using `PERTURB_LINF_COMPONENT_WEIGHT` and `PERTURB_RMSE_COMPONENT_WEIGHT`
- `speed_score = 1 - min(response_time / timeout, 1)`
- `final = PERTURB_PERTURBATION_WEIGHT * perturbation_score + PERTURB_SPEED_WEIGHT * speed_score`

Any verification or constraint failure gets `0.0`.

Weight setting:

- Only miners with `processed_count >= PERTURB_MIN_PROCESSED_COUNT` (default `50`) and a full score history are weight-eligible
- Emission schedule: top-5 only with fixed shares `70%, 25%, 3%, 1.5%, 0.5%` (ranks 6+ receive 0)

## Integration Smoke Test

Run after setup (reuses the downloaded ImageNet-100 dataset):

```bash
python scripts/integration_smoke_test.py
```

The smoke test validates:

- ImageNet-100 local image load
- local EfficientNetV2-L inference path
- direct challenge label selection from model prediction

## Troubleshooting

- Validator cannot generate challenges: verify internet access to Hugging Face for the first dataset download.
- No miner scoring activity: ensure miner hotkeys are registered and publicly reachable.
- Dependency install issues: install CUDA/CPU-specific PyTorch build compatible with your host.

## Readiness

Use `docs/READINESS_CHECKLIST.md` before long-run validation or deployment.

## Repository Map

- `neurons/validator.py`: validator loop, challenge build, verification, scoring, set_weights
- `neurons/miner.py`: baseline miner logic and Axon serving
- `perturbnet/protocol.py`: `AttackChallenge` synapse schema
- `perturbnet/model.py`: EfficientNet model load and label prediction helpers
- `perturbnet/image_io.py`: base64 image encode/decode helpers
- `perturbnet/imagenet100_bootstrap.py`: ImageNet-100 full-split download/open helpers
- `scripts/run_validator.sh`: start/restart validator with PM2
- `scripts/run_miner.sh`: start/restart miner with PM2
- `scripts/setup_common.sh`: role-aware bootstrap (PM2 + Python deps; `validator` also pre-downloads ImageNet-100)
- `scripts/bootstrap_imagenet100.py`: manual ImageNet-100 pre-download CLI
- `scripts/integration_smoke_test.py`: local integration test

