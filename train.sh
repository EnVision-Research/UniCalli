export FLUX_DEV="/hpc2hdd/home/txu647/.cache/huggingface/hub/flux1-dev.safetensors"
echo $FLUX_DEV

accelerate launch --config_file 8.yaml train_flux_deepspeed.py --config "train_configs/test_finetune.yaml"

# --config_file 0.yaml
# bash train.sh