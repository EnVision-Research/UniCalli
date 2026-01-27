# -*- coding: utf-8 -*-
"""
Chinese Calligraphy Generation with Flux Model
Author and font style controllable generation
"""

import os
import json
import torch
# Quantization options
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    from optimum.quanto import quantize, freeze, qint4
    HAS_QUANTO = True
except ImportError:
    HAS_QUANTO = False
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, List, Union, Dict, Any
from einops import rearrange
from pypinyin import lazy_pinyin

from src.flux.util import configs, load_ae, load_clip, load_t5
from src.flux.model import Flux
from src.flux.xflux_pipeline import XFluxSampler


def convert_to_pinyin(text):
    return ' '.join([item[0] if isinstance(item, list) else item for item in lazy_pinyin(text)])


class CalligraphyGenerator:
    """
    Chinese Calligraphy Generator using Flux model

    Attributes:
        device: torch device for computation
        model_name: name of the flux model (flux-dev or flux-schnell)
        font_styles: available font styles for generation
        authors: available calligrapher authors
    """

    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda",
        offload: bool = True,
        checkpoint_path: Optional[str] = None,
        intern_vlm_path: Optional[str] = None,
        ref_latent_path: Optional[str] = None,
        font_descriptions_path: str = "chirography.json",
        author_descriptions_path: str = "calligraphy_styles_en.json",
        use_deepspeed: bool = False,
        use_4bit_quantization: bool = False,
        deepspeed_config: Optional[str] = None
    ):
        """
        Initialize the calligraphy generator

        Args:
            model_name: flux model name (flux-dev or flux-schnell)
            device: device for computation
            offload: whether to offload model to CPU when not in use
            checkpoint_path: path to model checkpoint if using fine-tuned model
            intern_vlm_path: path to InternVLM model for text embedding
            ref_latent_path: path to reference latents for recognition mode
            font_descriptions_path: path to font style descriptions JSON
            author_descriptions_path: path to author style descriptions JSON
            use_deepspeed: whether to use DeepSpeed ZeRO for memory optimization
            deepspeed_config: path to DeepSpeed config JSON file
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.offload = offload
        self.is_schnell = model_name == "flux-schnell"
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        self.use_4bit_quantization = use_4bit_quantization

        # Load font and author style descriptions
        if os.path.exists(font_descriptions_path):
            with open(font_descriptions_path, 'r', encoding='utf-8') as f:
                self.font_style_des = json.load(f)
        else:
            raise FileNotFoundError(f"Font descriptions file not found: {font_descriptions_path}")

        if os.path.exists(author_descriptions_path):
            with open(author_descriptions_path, 'r', encoding='utf-8') as f:
                self.author_style = json.load(f)
        else:
            raise FileNotFoundError(f"Author descriptions file not found: {author_descriptions_path}")

        # Load models
        print("Loading models...")
        # When using DeepSpeed, load text encoders on CPU first to save memory during initialization
        # They will be moved to GPU after DeepSpeed initializes the main model
        if self.use_deepspeed:
            text_encoder_device = "cpu"
        elif offload:
            text_encoder_device = "cpu"  # Will be moved to GPU during inference
        else:
            text_encoder_device = self.device

        self.t5 = load_t5(text_encoder_device, max_length=256 if self.is_schnell else 512)
        self.clip = load_clip(text_encoder_device)
        self.clip.requires_grad_(False)

        # If checkpoint provided, load from checkpoint directly without loading flux weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            # When using DeepSpeed, don't move to GPU yet - let DeepSpeed handle it
            self.model = self._load_model_from_checkpoint(
                checkpoint_path, model_name,
                offload=offload,
                use_deepspeed=self.use_deepspeed
            )

            # Initialize DeepSpeed if requested
            if self.use_deepspeed:
                self.model = self._init_deepspeed(self.model)
        else:
            raise ValueError("Checkpoint path must be provided and exist for calligraphy generation.")

        # Load VAE
        if self.use_deepspeed or offload:
            vae_device = "cpu"
        else:
            vae_device = self.device

        self.vae = load_ae(model_name, device=vae_device)

        # Move VAE to GPU only if offload (not DeepSpeed)
        if offload and not self.use_deepspeed:
            self.vae = self.vae.to(self.device)

        # After DeepSpeed init, move text encoders to GPU
        if self.use_deepspeed:
            print("Moving text encoders to GPU...")
            self.t5 = self.t5.to(self.device)
            self.clip = self.clip.to(self.device)
            self.vae = self.vae.to(self.device)

        # Load reference latents if provided
        self.ref_latent = None
        if ref_latent_path and os.path.exists(ref_latent_path):
            print(f"Loading reference latents from {ref_latent_path}")
            self.ref_latent = torch.load(ref_latent_path, map_location='cpu')

        # Create sampler
        self.sampler = XFluxSampler(
            clip=self.clip,
            t5=self.t5,
            ae=self.vae,
            ref_latent=self.ref_latent,
            model=self.model,
            device=self.device,
            intern_vlm_path=intern_vlm_path
        )

        # Font for generating condition images
        self.font_path = "./checkpoints/FangZhengKaiTiFanTi-1.ttf"
        self.default_font_size = 102  # 128 * 0.8

    def _load_model_from_checkpoint(self, checkpoint_path: str, model_name: str, offload: bool, use_deepspeed: bool = False):
        """
        Load model from checkpoint without loading flux pretrained weights.
        This creates an empty model, initializes module embeddings, then loads your checkpoint.

        Args:
            checkpoint_path: Path to your checkpoint file
            model_name: flux model name (for config)
            offload: whether to offload to CPU
            use_deepspeed: whether using DeepSpeed (keeps model on CPU)

        Returns:
            model with loaded checkpoint
        """
        print(f"Creating empty flux model structure...")
        # Load checkpoint on CPU first to save memory
        # If using DeepSpeed, keep on CPU; otherwise move to GPU after loading
        load_device = "cpu"

        # Create model structure without loading pretrained weights (using "meta" device)
        with torch.device("meta"):
            model = Flux(configs[model_name].params)

        # Initialize module embeddings (must be done before loading checkpoint)
        print("Initializing module embeddings...")
        model.init_module_embeddings(tokens_num=320, cond_txt_channel=896)

        # Move model to loading device
        print(f"Moving model to {load_device} for loading...")
        model = model.to_empty(device=load_device)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = self._load_checkpoint_file(checkpoint_path)

        # Load weights into model
        model.load_state_dict(checkpoint, strict=False)

        # Apply 4-bit quantization if requested
        if hasattr(self, 'use_4bit_quantization') and self.use_4bit_quantization:
            if HAS_BNB:
                print("Applying bitsandbytes NF4 quantization...")
                model = self._quantize_model_bnb(model)
                model._is_quantized = True
                print("NF4 quantization complete!")
            elif HAS_QUANTO:
                print("Applying quanto 4-bit quantization...")
                model = model.float()
                quantize(model, weights=qint4)
                freeze(model)
                model._is_quantized = True
                print("4-bit quantization complete!")
            else:
                print("Warning: No quantization library available, running in full precision")

        # Move to GPU only if NOT using DeepSpeed (DeepSpeed will handle device placement)
        if not use_deepspeed:
            print(f"Moving model to {self.device}...")
            model = model.to(self.device)

        return model

    def _quantize_model_bnb(self, model):
        """
        Quantize model using bitsandbytes NF4.
        Replaces Linear layers with Linear4bit for true 4-bit inference.
        """
        import torch.nn as nn
        
        def replace_linear_with_4bit(module, name=''):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, nn.Linear):
                    # Create 4-bit linear layer
                    new_layer = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.bfloat16,
                        compress_statistics=True,
                        quant_type='nf4'
                    )
                    # Copy weights (will be quantized when moved to GPU)
                    new_layer.weight = bnb.nn.Params4bit(
                        child.weight.data,
                        requires_grad=False,
                        quant_type='nf4'
                    )
                    if child.bias is not None:
                        new_layer.bias = nn.Parameter(child.bias.data)
                    
                    setattr(module, child_name, new_layer)
                else:
                    replace_linear_with_4bit(child, full_name)
        
        print("Replacing Linear layers with Linear4bit...")
        replace_linear_with_4bit(model)
        return model

    def _init_deepspeed(self, model):
        """
        Initialize DeepSpeed for the model with ZeRO-3 inference optimization.

        Args:
            model: PyTorch model to wrap with DeepSpeed

        Returns:
            DeepSpeed inference engine
        """
        try:
            import deepspeed
        except ImportError:
            raise ImportError("DeepSpeed is not installed. Install it with: pip install deepspeed")

        # Load DeepSpeed config
        if self.deepspeed_config is None:
            self.deepspeed_config = "ds_config_zero2.json"

        if not os.path.exists(self.deepspeed_config):
            raise FileNotFoundError(f"DeepSpeed config not found: {self.deepspeed_config}")

        print(f"Initializing DeepSpeed Inference with config: {self.deepspeed_config}")

        # Initialize distributed environment for single GPU if not already initialized
        if not torch.distributed.is_initialized():
            import random
            # Set environment variables for single-process mode
            # Use a random port to avoid conflicts
            port = random.randint(29500, 29600)
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(port)
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'

            # Initialize process group
            try:
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=1,
                    rank=0
                )
                print(f"Initialized single-GPU distributed environment for DeepSpeed on port {port}")
            except RuntimeError as e:
                if "address already in use" in str(e):
                    print(f"Port {port} in use, trying again...")
                    # Try a different port
                    port = random.randint(29600, 29700)
                    os.environ['MASTER_PORT'] = str(port)
                    torch.distributed.init_process_group(
                        backend='nccl',
                        init_method='env://',
                        world_size=1,
                        rank=0
                    )
                    print(f"Initialized single-GPU distributed environment for DeepSpeed on port {port}")
                else:
                    raise

        # Use DeepSpeed inference API instead of initialize
        # This doesn't require an optimizer
        with open(self.deepspeed_config) as f:
            ds_config = json.load(f)

        model_engine = deepspeed.init_inference(
            model=model,
            mp_size=1,  # model parallel size
            dtype=torch.bfloat16 if ds_config.get('bf16', {}).get('enabled', False) else torch.float16,
            replace_with_kernel_inject=False,  # Don't replace with DeepSpeed kernels for custom models
        )

        print("DeepSpeed Inference initialized successfully")
        return model_engine

    def _load_checkpoint_file(self, checkpoint_path: str) -> dict:
        """
        Load checkpoint file and extract state dict.

        Args:
            checkpoint_path: Path to checkpoint file, can be:
                - Full checkpoint with model, optimizer, etc. (from training)
                - State dict only file
                - Directory containing checkpoint files
                - Safetensors file(s)

        Returns:
            state_dict: model state dictionary
        """
        # Check if it's a directory containing checkpoint files
        if os.path.isdir(checkpoint_path):
            # Look for safetensors index first (sharded), then single file, then .bin/.pt
            index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
            single_safetensors = os.path.join(checkpoint_path, "model.safetensors")
            
            if os.path.exists(index_path):
                # Load sharded safetensors
                return self._load_sharded_safetensors(checkpoint_path, index_path)
            elif os.path.exists(single_safetensors):
                # Load single safetensors file
                from safetensors.torch import load_file
                print(f"Loading safetensors: {single_safetensors}")
                return load_file(single_safetensors)
            
            # Fall back to .bin/.pt files
            possible_files = [
                'model.pt', 'model.pth', 'model.bin',
                'checkpoint.pt', 'checkpoint.pth',
                'pytorch_model.bin', 'model_state_dict.pt'
            ]

            checkpoint_file = None
            for filename in possible_files:
                full_path = os.path.join(checkpoint_path, filename)
                if os.path.exists(full_path):
                    checkpoint_file = full_path
                    print(f"Found checkpoint file: {filename}")
                    break

            if checkpoint_file is None:
                import glob
                pt_files = glob.glob(os.path.join(checkpoint_path, "*.pt")) + \
                          glob.glob(os.path.join(checkpoint_path, "*.pth")) + \
                          glob.glob(os.path.join(checkpoint_path, "*.bin"))
                if pt_files:
                    checkpoint_file = pt_files[0]
                    print(f"Found checkpoint file: {os.path.basename(checkpoint_file)}")
                else:
                    raise ValueError(f"No checkpoint files found in directory: {checkpoint_path}")

            checkpoint_path = checkpoint_file

        # Handle safetensors files
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            print(f"Loading safetensors: {checkpoint_path}")
            return load_file(checkpoint_path)

        # Load the checkpoint (.bin, .pt, .pth)
        print(f"Loading checkpoint file: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'epoch' in checkpoint:
                print(f"Checkpoint from epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                print(f"Checkpoint from step: {checkpoint['global_step']}")
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value
                         for key, value in state_dict.items()}
            print("Removed 'module.' prefix from state dict keys")

        return state_dict
    
    def _load_sharded_safetensors(self, checkpoint_dir: str, index_path: str) -> dict:
        """Load sharded safetensors checkpoint"""
        import json
        from safetensors.torch import load_file
        
        with open(index_path) as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        shard_files = set(weight_map.values())
        
        print(f"Loading {len(shard_files)} safetensors shards...")
        state_dict = {}
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(checkpoint_dir, shard_file)
            print(f"  Loading {shard_file}...")
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)
        
        return state_dict

    def text_to_cond_image(
        self,
        text: str,
        img_size: int = 128,
        font_scale: float = 0.8,
        font_path: Optional[str] = None
    ) -> Image.Image:
        """
        Convert text to condition image - text must be exactly 5 characters
        Matches the logic from image_datasets/get_cond.py

        Args:
            text: Chinese text to convert (must be 5 characters)
            img_size: size of each character block (default 128)
            font_scale: scale of font relative to image size (default 0.8)
            font_path: path to font file

        Returns:
            PIL Image with text rendered
        """
        if len(text) != 5:
            raise ValueError(f"Text must be exactly 5 characters, got {len(text)}")

        if font_path is None:
            font_path = self.font_path

        # Create font - font size is scaled down from img_size
        font_size_scaled = int(font_scale * img_size)
        font = ImageFont.truetype(font_path, font_size_scaled)

        # Calculate image dimensions for 5 characters
        img_width = img_size
        img_height = img_size * len(text)  # 5 characters

        # Create white background image
        cond_img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        cond_draw = ImageDraw.Draw(cond_img)

        # Draw each character
        # Note: font_size for positioning should be img_size, not the scaled font size
        for i, char in enumerate(text):
            font_space = font_size_scaled * (1 - font_scale) // 2
            # Position based on img_size blocks, not scaled font size
            font_position = (font_space, img_size * i + font_space)
            cond_draw.text(font_position, char, font=font, fill=(0, 0, 0))

        return cond_img

    def build_prompt(
        self,
        font_style: str = "楷",
        author: str = None,
        is_traditional: bool = True,
    ) -> str:
        """
        Build prompt for generation following dataset.py logic

        Args:
            font_style: font style (楷/草/行)
            author: author name (Chinese or None for synthetic)
            is_traditional: whether generating traditional calligraphy

        Returns:
            formatted prompt string
        """
        # Validate font style
        if font_style not in self.font_style_des:
            raise ValueError(f"Font style must be one of: {list(self.font_style_des.keys())}")

        # Convert font style to pinyin
        font_style_pinyin = convert_to_pinyin(font_style)

        # Build prompt based on traditional or synthetic
        if is_traditional and author and author in self.author_style:
            # Traditional calligraphy with specific author
            prompt = f"Traditional Chinese calligraphy works, background: black, font: {font_style_pinyin}, "
            prompt += self.font_style_des[font_style]
            author_info = self.author_style[author]
            prompt += f" author: {author_info}"
        else:
            # Synthetic calligraphy
            prompt = f"Synthetic calligraphy data, background: black, font: {font_style_pinyin}, "
            prompt += self.font_style_des[font_style]

        return prompt

    @torch.no_grad()
    def generate(
        self,
        text: str,
        font_style: str = "楷",
        author: str = None,
        width: int = 128,
        height: int = 640,  # Fixed for 5 characters
        num_steps: int = 50,
        guidance: float = 3.5,
        seed: int = None,
        is_traditional: bool = None,
        save_path: Optional[str] = None
    ) -> tuple[Image.Image, Image.Image]:
        """
        Generate calligraphy image from text

        Args:
            text: Chinese text to generate (must be exactly 5 characters)
            font_style: font style (楷/草/行)
            author: author/calligrapher name from the style list
            width: image width (default 128)
            height: image height (default 640 for 5 characters)
            num_steps: number of denoising steps
            guidance: guidance scale
            seed: random seed for generation
            is_traditional: whether generating traditional calligraphy (auto-determined if None)
            save_path: optional path to save the generated image

        Returns:
            tuple of (generated_image, condition_image)
        """
        # Validate text length
        if len(text) != 5:
            raise ValueError(f"Text must be exactly 5 characters, got {len(text)}: '{text}'")

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        # Fixed height for 5 characters
        height = width * 5

        # Auto-determine traditional vs synthetic
        if is_traditional is None:
            is_traditional = author is not None and author in self.author_style

        # Generate condition image
        cond_img = self.text_to_cond_image(text, img_size=width)

        # Build prompt
        prompt = self.build_prompt(
            font_style=font_style,
            author=author,
            is_traditional=is_traditional,
        )

        print(f"Generating with prompt: {prompt}")
        print(f"Text: {text}, Seed: {seed}")
        # Generate image
        result_img, recognized_text = self.sampler(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            controlnet_image=cond_img,
            is_generation=True,
            cond_text=text,
            required_chars=5,  # Fixed to 5
            seed=seed
        )

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            result_img.save(save_path)
            print(f"Image saved to {save_path}")

        return result_img, cond_img

    def batch_generate(
        self,
        texts: List[str],
        font_styles: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        output_dir: str = "./outputs",
        **kwargs
    ) -> List[tuple[Image.Image, Image.Image]]:
        """
        Batch generate calligraphy images

        Args:
            texts: list of texts to generate (each must be 5 characters)
            font_styles: list of font styles (if None, use default)
            authors: list of authors (if None, use synthetic)
            output_dir: directory to save outputs
            **kwargs: additional arguments for generate()

        Returns:
            list of (generated_image, condition_image) tuples
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        # Default styles and authors if not provided
        if font_styles is None:
            font_styles = ["楷"] * len(texts)
        if authors is None:
            authors = [None] * len(texts)

        for i, (text, font, author) in enumerate(zip(texts, font_styles, authors)):
            # Clean author name for filename
            author_name = author if author else "synthetic"
            if author and author in self.author_style:
                author_name = convert_to_pinyin(author)

            save_path = os.path.join(
                output_dir,
                f"{text}_{font}_{author_name}_{i}.png"
            )

            result_img, cond_img = self.generate(
                text=text,
                font_style=font,
                author=author,
                save_path=save_path,
                **kwargs
            )
            results.append((result_img, cond_img))

        return results

    def get_available_authors(self) -> List[str]:
        """Get list of available author styles"""
        return list(self.author_style.keys())

    def get_available_fonts(self) -> List[str]:
        """Get list of available font styles"""
        return list(self.font_style_des.keys())


# Hugging Face Pipeline wrapper
class FluxCalligraphyPipeline:
    """Hugging Face compatible pipeline for calligraphy generation"""

    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize the pipeline"""
        self.generator = CalligraphyGenerator(
            model_name=model_name,
            device=device,
            checkpoint_path=checkpoint_path,
            **kwargs
        )

    def __call__(
        self,
        text: Union[str, List[str]],
        font_style: Union[str, List[str]] = "楷",
        author: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate calligraphy images

        Args:
            text: text or list of texts to generate (each must be 5 characters)
            font_style: font style(s) (楷/草/行)
            author: author name(s) from the style list
            num_inference_steps: number of denoising steps
            guidance_scale: guidance scale for generation
            generator: torch generator for reproducibility

        Returns:
            generated image(s)
        """
        # Handle single text
        if isinstance(text, str):
            seed = None
            if generator is not None:
                seed = generator.initial_seed()

            result, _ = self.generator.generate(
                text=text,
                font_style=font_style,
                author=author,
                num_steps=num_inference_steps,
                guidance=guidance_scale,
                seed=seed,
                **kwargs
            )
            return result

        # Handle batch
        else:
            if isinstance(font_style, str):
                font_style = [font_style] * len(text)
            if isinstance(author, str) or author is None:
                author = [author] * len(text)

            results = []
            for t, f, a in zip(text, font_style, author):
                seed = None
                if generator is not None:
                    seed = generator.initial_seed()

                result, _ = self.generator.generate(
                    text=t,
                    font_style=f,
                    author=a,
                    num_steps=num_inference_steps,
                    guidance=guidance_scale,
                    seed=seed,
                    **kwargs
                )
                results.append(result)

            return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate Chinese calligraphy")
    parser.add_argument("--text", type=str, default="暴富且平安", help="Text to generate (must be 5 characters)")
    parser.add_argument("--font", type=str, default="楷", help="Font style (楷/草/行)")
    parser.add_argument("--author", type=str, default=None, help="Author/calligrapher name")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--list-authors", action="store_true", help="List available authors")
    parser.add_argument("--list-fonts", action="store_true", help="List available font styles")

    args = parser.parse_args()

    # Initialize generator
    generator = CalligraphyGenerator(
        model_name="flux-dev",
        device=args.device,
        checkpoint_path=args.checkpoint
    )

    # List available options
    if args.list_authors:
        print("Available authors:")
        for author in generator.get_available_authors()[:20]:  # Show first 20
            print(f"  - {author}")
        print(f"  ... and {len(generator.get_available_authors()) - 20} more")
        exit(0)

    if args.list_fonts:
        print("Available font styles:")
        for font in generator.get_available_fonts():
            print(f"  - {font}: {generator.font_style_des[font]}")
        exit(0)

    # Validate text length
    if len(args.text) != 5:
        print(f"Error: Text must be exactly 5 characters, got {len(args.text)}: '{args.text}'")
        print("Example: 暴富且平安, 心想事成达, 万事如意成")
        exit(1)

    # Generate
    result_img, cond_img = generator.generate(
        text=args.text,
        font_style=args.font,
        author=args.author,
        num_steps=args.steps,
        seed=args.seed,
        save_path=args.output
    )

    print(f"Generation complete! Saved to {args.output}")