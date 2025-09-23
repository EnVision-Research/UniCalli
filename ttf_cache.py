import os
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import torch
from torch import Tensor
from einops import rearrange
from PIL import Image
import numpy as np

from src.flux.modules.autoencoder import AutoEncoder


def _load_and_preprocess_image(path: str | Path, target_size: int = 256) -> Tensor:
    """
    Load PNG, convert to RGB, resize to target_size x target_size, normalize to [-1, 1].
    Returns CHW float tensor.
    """
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.size[0] != target_size or img.size[1] != target_size:
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    x = torch.from_numpy((np.array(img) / 127.5) - 1.0)  # HWC in [-1,1]
    if x.ndim != 3:
        # in case of grayscale after conversion issues
        x = x.expand(3, *x.shape[-2:]) if x.ndim == 2 else x
    x = x.permute(2, 0, 1).contiguous()  # CHW
    return x.float()


@torch.no_grad()
def build_cache_from_dir(
    input_dir: str | Path,
    ae: AutoEncoder,
    device: torch.device | str = 'cuda',
    batch_size: int = 128,
    target_size: int = 256,
) -> Tuple[List[str], Tensor]:
    """
    Encode all PNGs under input_dir into VAE latents.

    Returns:
    - keys: list of filenames (without directory)
    - latents: Tensor [N, C, H, W]
    """
    input_dir = Path(input_dir)
    files = sorted([p for p in input_dir.glob('*.png')])
    keys = [p.name for p in files]

    ae = ae.to(device)
    ae.eval()

    latents: List[Tensor] = []
    print(len(files), batch_size)
    for i in range(0, len(files), batch_size):
        print(i)
        batch_files = files[i:i + batch_size]
        batch = torch.stack([
            _load_and_preprocess_image(p, target_size=target_size) for p in batch_files
        ], dim=0).to(device)
        z = ae.encode(batch)  # [B, C, H, W]
        latents.append(z.detach().cpu())

    return keys, torch.cat(latents, dim=0)


def save_cache(cache_path: str | Path, keys: List[str], latents: Tensor) -> None:
    cache = {
        'keys': keys,
        'latents': latents,  # [N, C, H, W] on CPU
    }
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)


def load_cache(cache_path: str | Path) -> Tuple[List[str], Tensor]:
    cache = torch.load(cache_path, map_location='cpu')
    return cache['keys'], cache['latents']


def make_image_proj_from_cache(
    keys: List[str],
    latents: Tensor,
    select_keys: List[str] | None = None,
    device: torch.device | str = 'cuda',
) -> Tensor:
    """
    Turn cached latents into sequence for cross-attention.

    - If select_keys is None, use all in order.
    - Output shape: [1, L_total, C], where L_total = sum_i (H_i * W_i)
    - You can repeat to batch if needed.
    """
    if select_keys is None:
        idxs = list(range(len(keys)))
    else:
        name_to_idx: Dict[str, int] = {k: i for i, k in enumerate(keys)}
        idxs = [name_to_idx[k] for k in select_keys if k in name_to_idx]
        if len(idxs) == 0:
            raise ValueError('select_keys not found in cache keys')

    z = latents[idxs]  # [M, C, H, W]
    seq_list = []
    for zi in z:  # [C,H,W]
        seq = rearrange(zi, 'c h w -> (h w) c')  # [L_i, C]
        seq_list.append(seq)
    image_proj = torch.cat(seq_list, dim=0)[None, ...].to(device)  # [1, L_total, C]
    return image_proj


def build_and_save(
    input_dir: str | Path,
    cache_path: str | Path,
    ae: AutoEncoder,
    device: torch.device | str = 'cuda',
    batch_size: int = 128,
    target_size: int = 256,
) -> str:
    keys, latents = build_cache_from_dir(
        input_dir=input_dir,
        ae=ae,
        device=device,
        batch_size=batch_size,
        target_size=target_size,
    )
    save_cache(cache_path, keys, latents)
    return str(cache_path)


if __name__ == "__main__":
    from src.flux.util import load_ae

    device = 'cuda'
    ae = load_ae('flux-dev', device)
    build_and_save(
        input_dir='/data/user/txu647/code/flux-calligraphy/cache_',
        cache_path='/data/user/txu647/code/flux-calligraphy/sourcehan_chars_128_latents.pt',
        ae=ae,
        device=device,
        batch_size=256,
        target_size=128,  # 与 AE 分辨率一致
    )