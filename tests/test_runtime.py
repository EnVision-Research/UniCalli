import pytest
import torch

from src.flux.math import attention, rope
from src.flux.model import Flux, FluxParams
from src.flux.runtime import (
    assign_checkpoint, empty_device_cache, resolve_device,
    seeded_random, validate_optimizations,
    MPSStagingWeights,
)


def test_auto_device_priority(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device().type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device().type == "cpu"
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_device("mps")


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("option", ["use_deepspeed", "use_4bit_quantization"])
def test_unsupported_optimization_fails_early(device, option):
    with pytest.raises(ValueError, match="require CUDA"):
        validate_optimizations(device, **{option: True})
    validate_optimizations("cuda", **{option: True})


def test_cache_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    for device in ["cpu", "cuda", "mps"]:
        empty_device_cache(device)
    assert calls == ["cuda", "mps"]


def test_mmap_checkpoint_assignment(tmp_path):
    original = torch.nn.Linear(8, 4).to(torch.bfloat16)
    path = tmp_path / "weights.bin"
    torch.save(original.state_dict(), path)
    state = torch.load(path, mmap=True, weights_only=True)
    with torch.device("meta"):
        model = torch.nn.Linear(8, 4)
    assign_checkpoint(model, state)
    assert model.weight.data_ptr() == state["weight"].data_ptr()
    assert model.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.weight, original.weight)
    with torch.device("meta"):
        incomplete = torch.nn.Linear(8, 4)
    with pytest.raises(ValueError, match="missing keys"):
        assign_checkpoint(incomplete, {"weight": state["weight"]})


def test_cpu_seed_unchanged():
    actual = seeded_random((2, 16, 4, 4), "cpu", torch.float32, 42)
    expected = torch.randn(2, 16, 4, 4, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mps_staging_bounds_flat_storage(tmp_path):
    flat = torch.arange(1024, dtype=torch.float32)
    path = tmp_path / "flat.bin"
    torch.save({"weight": flat[256:272].view(4, 4)}, path)
    state = torch.load(path, mmap=True, weights_only=True)
    original = state["weight"]
    assert original.untyped_storage().nbytes() > original.numel() * original.element_size()
    weights = MPSStagingWeights(state)
    staged = weights["weight"]
    torch.testing.assert_close(staged, original)
    assert staged.storage_offset() == 0
    assert staged.untyped_storage().nbytes() == staged.numel() * staged.element_size()
    assert list(weights) == ["weight"]
    assert len(weights) == 1


mps_only = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


@mps_only
@pytest.mark.parametrize("normal", [True, False])
def test_mps_seed_matches_cpu_and_preserves_global_rng(normal):
    torch.manual_seed(100)
    before = torch.random.get_rng_state().clone()
    args = ((2, 16, 4, 4),)
    actual = seeded_random(*args, "mps", torch.bfloat16, 42, normal)
    expected = seeded_random(*args, "cpu", torch.bfloat16, 42, normal)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())


@mps_only
def test_mps_rope_and_attention():
    positions = torch.arange(512, dtype=torch.float32)[None]
    expected = rope(positions, 8, 10000)
    actual = rope(positions.to("mps"), 8, 10000)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    qkv = [seeded_random((1, 2, 512, 8), "cpu", torch.bfloat16, seed) for seed in range(3)]
    reference = attention(*qkv, expected[:, None])
    result = attention(*(x.to("mps") for x in qkv), actual[:, None])
    torch.testing.assert_close(result.cpu(), reference, rtol=0.03, atol=0.02)
    assert torch.isfinite(result).all()


@mps_only
def test_small_unicalli_transformer_on_mps():
    """Exercise real conditional double/single blocks without external weights."""
    torch.manual_seed(42)
    params = FluxParams(
        in_channels=16, vec_in_dim=8, context_in_dim=16,
        hidden_size=32, mlp_ratio=2, num_heads=4, depth=1,
        depth_single_blocks=1, axes_dim=[2, 2, 4], theta=10000,
        qkv_bias=True, guidance_embed=True,
    )
    model = Flux(params)
    model.init_module_embeddings(tokens_num=4, cond_txt_channel=8)
    model = model.to(dtype=torch.bfloat16).eval()
    inputs = dict(
        img=torch.randn(1, 8, 16), img_ids=torch.randn(1, 8, 3),
        txt=torch.randn(1, 512, 16), txt_ids=torch.zeros(1, 512, 3),
        y=torch.randn(1, 8), timesteps=torch.tensor([0.5]),
        timesteps2=torch.tensor([0.0]), cond_txt_latent=torch.randn(1, 5, 8),
        guidance=torch.tensor([3.5]),
    )
    inputs = {key: value.to(torch.bfloat16) for key, value in inputs.items()}
    with torch.inference_mode():
        expected = model(**{key: value.clone() for key, value in inputs.items()})
        model.to("mps")
        result = model(**{key: value.to("mps") for key, value in inputs.items()})
    assert result.shape == (1, 8, 16)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu(), expected, rtol=0.05, atol=0.05)
