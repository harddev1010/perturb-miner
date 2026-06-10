# Perturb Subnet

Perturb is a Bittensor subnet where validators create adversarial-image challenges and miners return perturbed images under bounded distortion constraints.

This repository provides:

- validator node implementation (`neurons/validator.py`)
- baseline miner implementation (`neurons/miner.py`)
- optional local semantic verification tooling (`tools/llm_endpoint_service.py`)
- one-command launchers for validator, miner, and llm endpoint

## Architecture

### Validator responsibilities

- Sample challenge images from a local ImageNet-100 dataset (`PERTURB_IMAGENET100_ROOT`)
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

1. Validator samples a local image from a persisted random ImageNet-100 traversal order
2. Validator runs `EfficientNetV2-L` and gets exact model label string
3. Validator creates challenge where `prompt` and `true_label` use the exact EfficientNet label
4. Validator sends challenge to sampled miners and scores returned perturbations

## Hardware and System Requirements

### Miner

- Minimum: 4 vCPU, 16 GB RAM, 50 GB SSD, stable 20+ Mbps network
- Recommended: 8 vCPU, 32 GB RAM, NVIDIA GPU with 8+ GB VRAM, 100+ GB SSD

### Validator

- Minimum: 8 vCPU, 32 GB RAM, NVIDIA GPU with 12+ GB VRAM, 100 GB SSD
- Recommended: 16 vCPU, 64 GB RAM, NVIDIA GPU with 24+ GB VRAM, 200 GB SSD

### Optional llm_endpoint

- Minimum: 2 vCPU, 4 GB RAM (assuming model already served by Ollama)
- Recommended: run on same private network/host as validator for low latency

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

- `miner`: creates `.venv`, installs Python/Bittensor dependencies only
- `validator`: also installs PM2 for process management

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

Validator setup automatically bootstraps a local ImageNet-100 challenge cache from Hugging Face. By default it downloads from `clane9/imagenet-100`, exports a shuffled multi-class sample of the train split into:

```bash
assets/imagenet-100
```

This happens during `bash ./scripts/setup_common.sh validator` and is checked again by `bash ./scripts/run_validator.sh`. The validator recursively scans common image extensions (`jpg`, `jpeg`, `png`, `webp`, `bmp`). You may still provide a custom local directory or manifest if needed.

### 2) Optional: configure and run local llm_endpoint

The validator no longer needs `llm_endpoint` to build challenges. The endpoint remains useful for manual label-similarity testing.

Create endpoint config if needed:

```bash
cp scripts/llm_endpoint.env.example scripts/llm_endpoint.env
```

Edit `scripts/llm_endpoint.env`:

- `LLM_ENDPOINT_HOST` (default `127.0.0.1`)
- `LLM_ENDPOINT_PORT` (default `8081`)
- `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- `PERTURB_LLM_ENDPOINT_MODEL` (default `qwen2.5:1.5b-instruct`)

Start llm_endpoint:

```bash
bash ./scripts/run_llm_endpoint.sh
```

Health check:

```bash
curl "http://127.0.0.1:8081/health"
```

### 3) Configure validator runtime

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

- `PERTURB_IMAGENET100_ROOT` (directory containing ImageNet-100 images)
- `PERTURB_IMAGENET100_MANIFEST` (optional manifest path)
- `PERTURB_IMAGENET100_AUTO_DOWNLOAD` (`true` by default)
- `PERTURB_IMAGENET100_REPO_ID` (default `clane9/imagenet-100`)
- `PERTURB_IMAGENET100_SPLIT` (default `train`)
- `PERTURB_IMAGENET100_MAX_IMAGES` (default `5000`)
- `PERTURB_IMAGENET100_MIN_IMAGES` (default `1000`)
- `PERTURB_K_MINERS`
- `PERTURB_HISTORY_SIZE`
- `PERTURB_MIN_PROCESSED_COUNT`
- `PERTURB_MIN_LINF_DELTA`
- `PERTURB_MAX_LINF_DELTA`
- `PERTURB_WANDB_ENABLED` (`true` to enable validator metrics logging to Weights & Biases)
- `PERTURB_WANDB_PROJECT`, `PERTURB_WANDB_ENTITY`, `PERTURB_WANDB_RUN_NAME`, `PERTURB_WANDB_MODE`
- `PERTURB_WANDB_LOG_CONSOLE` (`true` to forward validator console logs to W&B as well)
- `LOG_LEVEL` (`DEBUG` default, set `INFO`/`WARNING`/`ERROR` if you want quieter logs)

### 4) Start validator

```bash
bash ./scripts/run_validator.sh
```

Expected log behavior:

- challenge generation messages
- miner selection messages
- per-miner score logs
- periodic `set_weights` attempts

### 5) Validator-side notes

- Challenge generation no longer depends on external image APIs or LLM verification.
- ImageNet-100 selection walks every cached image once in random order before reshuffling for the next epoch.

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
- Miner does not run llm_endpoint; validators handle all challenge verification and scoring.

## API and Protocol Contracts

### ImageNet-100 input contract (validator challenge source)

- The normal setup/start scripts automatically populate `PERTURB_IMAGENET100_ROOT` from Hugging Face if the local cache is missing or too small.
- The default source is `clane9/imagenet-100`, split `train`, capped at `5000` exported images. The export interleaves parquet shards and shuffles the stream so the capped cache spans many classes instead of only the first few.
- Optional: export `HF_TOKEN` before setup for faster, higher-rate-limit downloads from Hugging Face.
- `PERTURB_IMAGENET100_ROOT` points to the local image cache directory.
- Optional `PERTURB_IMAGENET100_MANIFEST` points to a text manifest with one image path per line.
- Validator persists the shuffled image order, cursor, dataset fingerprint, and epoch in state, so restarts/resumes continue the traversal without duplicate selections until the cache is exhausted.
- Validator converts image bytes to base64 internally.
- The model-predicted EfficientNet label becomes both `prompt` and `true_label`.

### llm_endpoint contract (optional manual verification tooling)

- Endpoint: `POST /verify-label`
- Request JSON:

```json
{
  "prediction": "<efficientnet_label>",
  "target_label": "<prompt_label>",
  "llm_model": "<optional model hint>"
}
```

- Response JSON must contain a boolean verdict key, typically:

```json
{
  "is_match": true,
  "reason": "short explanation",
  "method": "ollama"
}
```

Operations endpoints:

- `GET /health`
- `GET /metrics`

### Synapse contract (`AttackChallenge`)

Key fields sent to miners:

- `task_id`
- `model_name` (fixed `EfficientNetV2-L`)
- `prompt` (exact EfficientNet class label)
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

- Only miners with `processed_count > 100` are weight-eligible
- Emission schedule: top-5 only with fixed shares `62%, 24%, 9%, 4%, 1%` (ranks 6+ receive 0)
- Final weights combine normalized rolling average and normalized rank bonus, then normalize to sum 1

## Integration Smoke Test

Run after setup or validator startup has prepared the ImageNet-100 cache:

```bash
python scripts/integration_smoke_test.py
```

The smoke test validates:

- ImageNet-100 local image load
- local EfficientNetV2-L inference path
- direct challenge label selection from model prediction

## Troubleshooting

- Validator cannot generate challenges: verify internet access to Hugging Face for first bootstrap, or check `PERTURB_IMAGENET100_ROOT` / optional manifest paths.
- No miner scoring activity: ensure miner hotkeys are registered and publicly reachable.
- Dependency install issues: install CUDA/CPU-specific PyTorch build compatible with your host.
- Slow optional verifier responses: reduce model size or place llm_endpoint closer to the caller.

## Readiness

Use `docs/READINESS_CHECKLIST.md` before long-run validation or deployment.

## Repository Map

- `neurons/validator.py`: validator loop, challenge build, verification, scoring, set_weights
- `neurons/miner.py`: baseline miner logic and Axon serving
- `perturbnet/protocol.py`: `AttackChallenge` synapse schema
- `perturbnet/model.py`: EfficientNet model load and label prediction helpers
- `perturbnet/image_io.py`: base64 image encode/decode helpers
- `tools/llm_endpoint_service.py`: optional semantic verification service
- `scripts/run_llm_endpoint.sh`: start/restart llm endpoint with PM2 (auto-ensures Ollama + model)
- `scripts/run_validator.sh`: start/restart validator with PM2
- `scripts/run_miner.sh`: start/restart miner with PM2
- `scripts/setup_common.sh`: role-aware bootstrap (`miner` = Python deps only, `validator` = adds PM2)
- `scripts/integration_smoke_test.py`: local integration test

