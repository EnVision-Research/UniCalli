#!/usr/bin/env bash
set -e

export FLUX_DEV="/data/user/txu647/.cache/huggingface/hub/flux1-dev.safetensors"
echo $FLUX_DEV

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DISABLE_TELEMETRY=YES
export WANDB_MODEL=offline

ACCEL=8.yaml
TRAIN=train_configs/test_finetune.yaml

DEBUG=false
[[ "${1:-}" == "--debug" ]] && DEBUG=true

if $DEBUG; then
  NP=1; DBG=true
else
  NP=8; DBG=false
fi

TMP="$(mktemp).yaml"
cp "$TRAIN" "$TMP"

# 写入/覆盖 debug_mode
if grep -qE '^[[:space:]]*debug_mode:' "$TMP"; then
  sed -i -E "s|^([[:space:]]*debug_mode:[[:space:]]*).*$|\1$DBG|" "$TMP"
else
  printf "\ndebug_mode: %s\n" "$DBG" >> "$TMP"
fi

accelerate launch --config_file "$ACCEL" --num_processes "$NP" \
  train_flux_deepspeed.py --config "$TMP"
# accelerate launch --config_file 8.yaml train_flux_deepspeed.py --config "train_configs/test_finetune.yaml"

# --config_file 0.yaml
# bash train.sh