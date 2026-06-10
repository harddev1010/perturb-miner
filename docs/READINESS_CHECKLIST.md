# Perturb Subnet Readiness Checklist

Use this before long validator uptime tests or mainnet deployment.

## 1) Environment

- [ ] Wallet hotkeys are registered for validator and miners on target `NETUID`
- [ ] `scripts/validator.env` and `scripts/miner.env` are configured
- [ ] First validator setup/start can reach Hugging Face, or `PERTURB_IMAGENET100_ROOT` already has a populated cache
- [ ] GPU drivers/CUDA stack matches installed PyTorch build

## 2) Optional LLM Endpoint Service

- [ ] If using manual verifier checks, local llm_endpoint starts with `./scripts/run_llm_endpoint.sh`
- [ ] If running, `GET /health` returns `status=ok`

## 3) Validator + Miner Launch

- [ ] Miner starts with `bash ./scripts/run_miner.sh`
- [ ] Validator starts with `bash ./scripts/run_validator.sh`
- [ ] Validator setup/start logs that the ImageNet-100 challenge cache is ready
- [ ] Validator logs challenge creation and miner selection each loop
- [ ] Validator logs periodic `set_weights` attempts

## 4) Integration Smoke Test

- [ ] Run:
  - `python scripts/integration_smoke_test.py`
- [ ] Check output reports:
  - ImageNet-100 image load succeeds
  - EfficientNetV2-L prediction succeeds
  - challenge target label is selected from model prediction
  - validator logs include `ssim` and `psnr_db` for scored responses

## 5) Long-Run Reliability

- [ ] Run validator/miner together for 6-24 hours
- [ ] No repeated validator crash loops
- [ ] No persistent ImageNet-100 loading failures

## 6) Operational Guardrails

- [ ] Log rotation policy configured for long-running nodes
- [ ] Alerting in place for validator exceptions
- [ ] Backups enabled for validator state/log artifacts if required
