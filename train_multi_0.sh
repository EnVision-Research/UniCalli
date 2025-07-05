export FLUX_DEV="/data/user/user64/.cache/huggingface/hub/flux1-dev.safetensors"
echo $FLUX_DEV

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DISABLE_TELEMETRY=YES
export WANDB_MODEL=offline

accelerate launch --config_file 8_0.yaml train_flux_deepspeed_controlnet.py --config "train_configs/test_controlnet.yaml"

# --config_file 0.yaml