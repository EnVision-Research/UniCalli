export FLUX_DEV="/data/user/txu647/.cache/huggingface/hub/flux1-dev.safetensors"
echo $FLUX_DEV

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DISABLE_TELEMETRY=YES
export WANDB_MODEL=offline

accelerate launch --config_file 8.yaml train_flux_deepspeed_consist_loss.py --config "train_configs/test_finetune.yaml"

# --config_file 0.yaml
# bash train.sh