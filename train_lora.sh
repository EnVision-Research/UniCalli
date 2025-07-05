export FLUX_DEV="/hpc2hdd/home/txu647/.cache/huggingface/hub/flux1-dev.safetensors"
echo $FLUX_DEV

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DISABLE_TELEMETRY=YES
export WANDB_MODEL=offline

accelerate launch --config_file 8.yaml train_flux_lora_deepspeed.py --config "train_configs/test_lora.yaml"

# --config_file 0.yaml
# bash train_lora.sh