#!/usr/bin/env bash
# Convenience wrapper to train DeepSeek-V3 "deepseek_aghilora" with 8 GPUs.
#
# Usage:
#   ./run_deepseek_aghilora_8gpu.sh
#
# Optional overrides:
#   NGPU=8 LOG_RANK=0,1 COMM_MODE=fake_backend ./run_deepseek_aghilora_8gpu.sh --training.steps=1
#
# Notes:
# - This delegates to ./run_train.sh, which runs `python -m torchtitan.train --job.config_file ...`
# - The config uses `checkpoint.initial_load_in_hf=true` to load base HF safetensors.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NGPU="${NGPU:-8}"
export LOG_RANK="${LOG_RANK:-0}"
export TRAIN_FILE="${TRAIN_FILE:-torchtitan.train}"
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/torchtitan/models/deepseek_v3/train_configs/deepseek_aghilora.toml}"

exec "${REPO_ROOT}/run_train.sh" "$@"

