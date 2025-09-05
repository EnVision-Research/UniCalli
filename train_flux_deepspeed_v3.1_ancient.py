import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoModel
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5)
from src.flux.xflux_pipeline import XFluxSampler
from image_datasets.dataset_oracle import loader_oracle
from image_datasets.dataset_eg import loader_eg

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config

def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'validation'), exist_ok=True)

    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)

    dit.init_module_embeddings(tokens_num=64)
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit = dit.to(torch.float32)
    dit.train()

    intern_path = "/data/user/txu647/.cache/InternVL3-1B"
    embed_tokens = AutoModel.from_pretrained(
        intern_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    ).language_model.model.embed_tokens.eval()
    embed_tokens.requires_grad_(False)
    # emb_token_weights = embed_tokens.weight.clone().detach()  #  detach: requires_grad False

    optimizer_cls = torch.optim.AdamW
    #you can train your own layers
    for n, param in dit.named_parameters():
        # param.requires_grad = True
        if 'txt_attn' not in n:
            param.requires_grad = False
        # if 'attn' not in n:
        #     param.requires_grad = False
    # dit.module_embeddings.requires_grad_(True)

    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1000000, 'M parameters')
    optimizer = optimizer_cls(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.dataset_type == 'eg':
        train_dataloader = loader_eg(**args.data_config)
    elif args.dataset_type == 'oracle':
        train_dataloader = loader_oracle(**args.data_config)
    else:
        raise NotImplementedError
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            dit_state = torch.load(os.path.join(args.output_dir, path, 'dit.bin'), map_location='cpu', weights_only=False)
            dit_state2 = {}
            for k in dit_state.keys():
                dit_state2[k[len('module.'):]] = dit_state[k]
            
            # use (1, N, C) modulate embedding shape
            if dit_state2['module_embeddings'].shape != dit.module_embeddings.shape:
                dit_state2['module_embeddings'] = dit_state2['module_embeddings'][:, :dit.module_embeddings.shape[1]]
                
            dit.load_state_dict(dit_state2)
            # optimizer_state = torch.load(os.path.join(args.output_dir, path, 'optimizer.bin'), map_location='cpu', weights_only=False)
            # optimizer.load_state_dict(optimizer_state['base_optimizer_state'])

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    dit, optimizer, _, lr_scheduler = accelerator.prepare(
        dit, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    print_shape_flag = True
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit):
                img, prompts, cond_image, text_token = batch
                text_latent = embed_tokens(text_token).to(accelerator.device)

                if print_shape_flag:
                    print_shape_flag = False
                    print(f"img shape: {img.shape}")
                    print(f"cond_image shape: {cond_image.shape}")
                    print(f"prompts: {prompts}")
                    print(f"grad norm: {args.max_grad_norm}")
                    print(f"learning rate: {args.learning_rate}")

                cond_image = cond_image.to(accelerator.device)
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    cond_latent = vae.encode(cond_image.to(torch.float32))
                    # inp = prepare(t5=t5, clip=clip, 
                    #     img=torch.cat((x_1, cond_latent), dim=3), prompt=prompts)
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)

                bs = img.shape[0]
                mode = random.choice(['cond', 'img'])
                # mode = 'img'
                if mode == 'img':  # pred cond, by given pure img
                    t = torch.sigmoid(torch.randn((1,), device=accelerator.device))
                    t_cond = torch.zeros_like(t).to(accelerator.device)
                else:
                    t_cond = torch.sigmoid(torch.randn((1,), device=accelerator.device))
                    t = torch.zeros_like(t_cond).to(accelerator.device)
                    
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t) * x_1 + t * x_0
                b, c, h, w = x_t.shape
                
                cond_0 = torch.randn_like(cond_latent).to(accelerator.device)
                cond_t = (1 - t_cond) * cond_latent + t_cond * cond_0

                cond_txt_0 = torch.randn_like(text_latent).to(accelerator.device)
                cond_txt_t = (1 - t_cond) * text_latent + t_cond * cond_txt_0
                # x_t = torch.cat((x_t, cond_t), dim=3)
                
                x_t = rearrange(x_t, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                cond_t = rearrange(cond_t, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                x_t = torch.cat((x_t, cond_t), dim=1)
                # b, c, h, w = x_t.shape
                # x_t = rearrange(x_t, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                # Predict the noise residual and compute loss
                model_pred, txt_pred = dit(img=x_t.to(weight_dtype),
                                img_ids=inp['img_ids'].to(weight_dtype),
                                txt=inp['txt'].to(weight_dtype),
                                txt_ids=inp['txt_ids'].to(weight_dtype),
                                cond_txt_latent=cond_txt_t.to(weight_dtype),
                                y=inp['vec'].to(weight_dtype),
                                timesteps=t.to(weight_dtype),
                                timesteps2=t_cond.to(weight_dtype),
                                guidance=guidance_vec.to(weight_dtype),)

                img_pred, cond_pred = model_pred.chunk(2, dim=1)
                img_pred = rearrange(img_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h//2, w=w//2, ph=2, pw=2)
                cond_pred = rearrange(cond_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h//2, w=w//2, ph=2, pw=2)

                loss_img = F.mse_loss(img_pred.float(), (x_0 - x_1).float(), reduction="mean")
                loss_cond = F.mse_loss(cond_pred.float(), (cond_0 - cond_latent).float(), reduction="mean")
                loss_cond_txt = 0.1 * F.mse_loss(txt_pred.float(), (cond_txt_0 - text_latent).float(), reduction="mean")

                if mode == 'img':
                    loss = loss_img + 0.01 * loss_cond + 0.01 * loss_cond_txt
                else:
                    loss = loss_cond + loss_cond_txt + 0.01 * loss_img
                # loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if not args.disable_sampling and global_step % args.sample_every == 0:
                    os.makedirs(f"{args.output_dir}/validation/{global_step}", exist_ok=True)
                    if accelerator.is_main_process:
                        print(f"Sampling images for step {global_step}...")
                        sampler = XFluxSampler(clip=clip, t5=t5, ae=vae, 
                                model=dit, device=accelerator.device, intern_vlm_path=intern_path)
                        # images = []
                        with open(f"test_data/{args.dataset_type}/cond.txt", "r", encoding="utf-8") as f:
                            text = f.read()
                        cond_text_list = text.splitlines()

                        for i, prompt in enumerate(args.sample_prompts):
                            # generation
                            idx = i
                            cond_text = cond_text_list[i]
                            print('cond_text:', cond_text)
                            cond_image = Image.open(f'test_data/{args.dataset_type}/cond_{idx}.png')
                            # cond_image = Image.open(f'test_data_rec/img_{idx}.png')
                            result, _ = sampler(prompt=prompt,
                                            width=args.sample_width,
                                            height=args.sample_height,
                                            num_steps=args.sample_steps,
                                            controlnet_image=cond_image,
                                            is_generation=True,
                                            cond_text=cond_text,
                                            required_chars=5
                                            )
                            # images.append(wandb.Image(result))
                            result.save(f"{args.output_dir}/validation/{global_step}/gen_{idx}.png")

                        for i, prompt in enumerate(args.sample_prompts):
                            # recognition
                            idx = i
                            cond_image = Image.open(f'test_data/{args.dataset_type}/img_{idx}.png')
                            # cond_image = Image.open(f'test_data_rec/img_{idx}.png')
                            result, text = sampler(prompt=prompt,
                                            width=args.sample_width,
                                            height=args.sample_height,
                                            num_steps=args.sample_steps,
                                            controlnet_image=cond_image,
                                            is_generation=False,
                                            required_chars=5
                                            )
                            # images.append(wandb.Image(result))
                            # try:
                            #     result.save(f"{args.output_dir}/validation/{global_step}/rec_{idx}_{text[0]}.png")
                            # except:
                            result.save(f"{args.output_dir}/validation/{global_step}/rec_{idx}.png")
                            print(text)
                        # wandb.log({f"Results, step {global_step}": images})

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        torch.save(dit.state_dict(), os.path.join(save_path, 'dit.bin'))
                        torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.bin'))
                        #accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()