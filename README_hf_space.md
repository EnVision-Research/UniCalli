---
title: UniCalli - Chinese Calligraphy Generator
emoji: 🖌️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: other
suggested_hardware: zero-a10g
models:
  - TSXu/UniCalli-base
  - OpenGVLab/InternVL3-1B
---

# 🖌️ UniCalli - Chinese Calligraphy Generator / 中国书法生成器

Generate beautiful Chinese calligraphy in various styles and by different historical masters.

用不同历史书法大师的风格生成精美的中国书法。

## Features

- **Multi-Master Styles**: Supports 70+ historical calligraphy masters including Wang Xizhi, Yan Zhenqing, Huang Tingjian, etc.
- **Multiple Font Styles**: Regular Script (楷), Running Script (行), Cursive Script (草)
- **4-bit Quantization**: Runs on ZeroGPU with only 18G VRAM

## Usage

1. Enter 5 Chinese characters
2. Select a calligrapher
3. Select a font style (based on the calligrapher's available styles)
4. Click "Generate"

## Citation

```bibtex
@article{xu2025unicalli,
  title={UniCalli: A Unified Diffusion Framework for Column-Level Generation and Recognition of Chinese Calligraphy},
  author={Xu, Tianshuo and Wang, Kai and Chen, Zhifei and Wu, Leyi and Wen, Tianshui and Chao, Fei and Chen, Ying-Cong},
  journal={arXiv preprint arXiv:2025.13745},
  year={2025}
}
```

## Links

- [Paper](https://arxiv.org/abs/2510.13745)
- [Project Page](https://envision-research.github.io/UniCalli/)
- [GitHub](https://github.com/EnVision-Research/UniCalli)
- [Model](https://huggingface.co/TSXu/UniCalli-base)
