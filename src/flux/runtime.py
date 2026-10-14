"""Device helpers shared by the inference entry points."""

from collections.abc import Mapping

import torch


def resolve_device(device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    result = torch.device(device)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use an Apple Silicon Mac with an MPS-enabled PyTorch build.")
    return result


def validate_optimizations(device, use_deepspeed=False, use_4bit_quantization=False):
    if torch.device(device).type != "cuda" and (use_deepspeed or use_4bit_quantization):
        raise ValueError("DeepSpeed and the current 4-bit inference path require CUDA; disable them for MPS/CPU.")


def empty_device_cache(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def seeded_random(shape, device, dtype, seed, normal=True):
    """Use a CPU generator for MPS, preserving a reproducible local seed."""
    device = torch.device(device)
    random_device = torch.device("cpu") if device.type == "mps" else device
    generator = torch.Generator(device=random_device).manual_seed(seed)
    operation = torch.randn if normal else torch.rand
    return operation(*shape, device=random_device, dtype=dtype, generator=generator).to(device)


def assign_checkpoint(model, state_dict):
    """Materialize a meta model without allocating a second full parameter copy."""
    missing = set(model.state_dict()) - set(state_dict)
    if missing:
        raise ValueError(f"MPS loading requires a complete checkpoint; missing keys: {sorted(missing)}")
    model.load_state_dict(state_dict, strict=False, assign=True)
    return model


class MPSStagingWeights(Mapping):
    """Detach tensor views from giant flat checkpoint storages before Metal copies."""

    def __init__(self, weights):
        self.weights = weights

    def __getitem__(self, key):
        # The released .bin uses views into one 24 GB storage. Without staging,
        # Metal may map that entire storage when copying even a small parameter.
        return self.weights[key].clone(memory_format=torch.contiguous_format)

    def __iter__(self):
        return iter(self.weights)

    def __len__(self):
        return len(self.weights)


def enable_sequential_offload(model, device, preload_module_classes=None):
    """Keep CPU-backed weights and transfer only the current layer to MPS."""
    from accelerate import cpu_offload

    device = torch.device(device)
    state_dict = MPSStagingWeights(model.state_dict()) if device.type == "mps" else None
    model = cpu_offload(
        model, execution_device=torch.device(device), offload_buffers=True,
        state_dict=state_dict,
        preload_module_classes=preload_module_classes,
    )
    if device.type == "mps":
        # Wait for each streamed block before dropping its cached Metal buffers.
        # Otherwise queued work can keep previous blocks' allocations alive and
        # cause heavy unified-memory paging despite the meta parameters.
        def release_weights(module, args, output):
            torch.mps.synchronize()
            torch.mps.empty_cache()

        for module in model.modules():
            is_block = preload_module_classes and module.__class__.__name__ in preload_module_classes
            is_leaf = preload_module_classes is None and next(module.parameters(recurse=False), None) is not None
            if is_block or is_leaf:
                module.register_forward_hook(release_weights)
    return model
