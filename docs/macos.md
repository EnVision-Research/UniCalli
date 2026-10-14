# Experimental Apple Silicon inference

This patch adds an experimental MPS execution path. The full released checkpoint
completed an 8-step generation on an M5 Pro with 24 GB unified memory using
sequential offloading. Whole-model loading and quantization have different memory
requirements and are not covered by that result.

## Requirements

- Apple Silicon, macOS 14 or newer (the existing pipeline uses bfloat16).
- An MPS-enabled PyTorch build. Smoke tests were run with Python 3.12 and
  PyTorch 2.14.1 on an M5 Pro with 24 GB unified memory.
- Whole-model mode needs memory for the complete 24.3 GB diffusion checkpoint
  plus text encoders, the VAE, embeddings and activations.
  For smaller Macs, use `--sequential-offload` to transfer one block at a time
  instead of placing the full diffusion model on MPS. This trades speed for a
  smaller GPU working set; it does not quantize the weights.
- Enough disk space for the checkpoint, encoders, embeddings and environment.

CUDA's advertised 18 GB 4-bit requirement does not apply to this MPS path.
DeepSpeed and the current 4-bit path are rejected on MPS/CPU before model loading.
CPU offloading moves models between stages, but CPU and GPU share physical memory
on Apple Silicon. Stage offloading still places a complete model on the GPU.
Sequential offloading instead retains memory-mapped CPU weights and transfers
individual blocks for execution. It also streams the text encoders and VAE.

## Installation and experimental inference

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c 'import torch; print(torch.backends.mps.is_available())'
```

Download the official model package following the main README and accept its
access conditions. The original academic/non-commercial restrictions still apply.

```sh
python inference.py --device mps --sequential-offload \
  --checkpoint ./checkpoints/unicalli-base_cleaned.bin \
  --intern-vlm-path ./checkpoints/internvl_embedding \
  --text '生日快乐喵' --font 楷 --author 颜真卿 \
  --steps 25 --seed 42 --output output.png
```

The released model still requires exactly five characters and produces a column.
The Gradio entry point selects CUDA, then MPS, then CPU automatically; override
it with `UNICALLI_DEVICE=mps python app.py`. The 4-bit checkbox is disabled outside
CUDA. No CPU fallback for unsupported Metal operations is enabled implicitly.
The Gradio entry point defaults to sequential offloading on MPS; set
`UNICALLI_SEQUENTIAL_OFFLOAD=0` to use whole-model stage offloading on larger Macs.
On MPS the interface defaults to 8 preview steps; increasing steps takes longer.

## What changed

- MPS RoPE uses float32; CUDA/CPU retain the original float64 calculation.
- Seeded MPS noise is generated on CPU and transferred to Metal, without changing
  global RNG state. Cross-device full-image equivalence is not promised.
- MPS `.bin` checkpoint loading uses `mmap=True`, `weights_only=True`, and assigns
  parameters into a meta model rather than first allocating a second full copy.
  Incomplete checkpoints fail explicitly instead of leaving meta tensors unloaded.
  This path expects modern PyTorch ZIP-format tensor checkpoints. It does not
  convert or quantize the released model.
- The sampler receives the requested offload setting and moves the VAE encoder
  back to the device before encoding; cache clearing targets the selected backend.
- Optional sequential offloading uses Accelerate hooks to load only the current
  block. Text encoders explicitly send tokenizer inputs to their execution device,
  and load with `low_cpu_mem_usage=True` to avoid an initial duplicate allocation.
  The released checkpoint stores parameters as views into one large flat storage;
  each streamed parameter is staged in an independent contiguous CPU buffer before
  transfer so Metal does not map that entire storage for a small parameter. MPS
  execution is synchronized between blocks and unused Metal buffers are released.
- DeepSpeed is omitted on macOS; the Hugging Face Hub range is compatible with
  the existing Gradio 6.3.0 and Transformers 4.43.3 dependencies.

## Validation

```sh
pip install pytest
python -m pytest tests -q
```

Tests cover device selection, unsupported optimization options, checkpoint
assignment, RNG behavior, RoPE, attention, and real conditional double/single
Transformer blocks on MPS. A sampler integration test also runs encoding,
denoising and decoding twice with offloading, a small random Transformer and stub
encoders/VAE, checking that both runs finish and have identical output. Sequential
offloading is tested with the sampler and real small T5/CLIP architectures. A
regression test checks that a small tensor view from a flat checkpoint is staged
with storage bounded by its own tensor size. All 21
tests passed on the machine described above. MPS tests skip when no Apple GPU is
exposed. These tests alone do not establish pretrained-model quality or memory fit.

### Full-checkpoint smoke test

On the machine above, the official `unicalli-base_cleaned.bin` plus T5, CLIP,
VAE and the extracted InternVL embedding generated `生日快乐喵` with author
`颜真卿`, script `楷`, seed `42`, and 8 denoising steps. The output was 128 x 640,
with all five characters visually identifiable (the output uses the traditional
`樂`). Wall time including model loading was 419.7 seconds. The image has stray
white marks and this low-step smoke test is not a style/quality benchmark.

The released checkpoint's parameter names and shapes matched the code after
removing the training `module.` prefix. Without independent staging buffers,
the shared flat storage caused severe paging on this hardware.

Before calling the port production-ready, benchmark representative inputs at
normal sampling steps and repeat full-checkpoint generation to check long-running
offload behavior. CUDA end-to-end regression testing and MPS quantization remain
outstanding.
