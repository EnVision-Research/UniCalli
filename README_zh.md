# UniCalli: 中国书法列级生成与识别的统一扩散框架

[![arXiv](https://img.shields.io/badge/arXiv-2025.13745-b31b1b.svg)](https://arxiv.org/abs/2510.13745)
[![项目主页](https://img.shields.io/badge/Project-Page-green)](https://envision-research.github.io/UniCalli/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/TSXu/UniCalli-base)
[![魔搭社区](https://img.shields.io/badge/ModelScope-Model-blue)](https://www.modelscope.cn/models/tianshuo/UniCalli-base)
[![GitHub](https://img.shields.io/github/stars/EnVision-Research/UniCalli?style=social)](https://github.com/EnVision-Research/UniCalli)

[English](README.md) | 简体中文

<p align="center">
  <img src="docs/assets/demo.png" alt="UniCalli Demo" width="800">
</p>

## 概述

UniCalli 是一个突破性的统一扩散框架，解决了中国书法列级生成问题。与现有方法专注于孤立字符生成或在页面级合成中牺牲书法正确性不同，UniCalli 在单一模型中集成了识别和生成任务，在风格保真度和结构准确性方面都取得了卓越的成果。

### 主要特性

- **统一架构**: 首个统一列级书法生成与识别的框架
- **多大师风格**: 支持多样化的书法风格，包括王羲之、颜真卿、欧阳询等
- **密集标注数据**: 在大规模书法数据集上训练，包含详细标注，包括边界框、现代文字转录、作者信息和书体风格

## 许可证

仅供学术研究和非商业使用。商业使用请联系作者。

For academic research and non-commercial use only. For commercial use, please contact the authors. 

## TODO 列表

- [x] **模型发布** - 不带 pred_box 的基础版本
- [x] **推理代码**
- [x] **4-bit量化** - 仅需要18G显存！
- [x] **交互式演示**
- [ ] **数据集发布**
- [ ] **训练代码**

## 快速开始

### 安装

```bash
git clone https://github.com/EnVision-Research/UniCalli.git
cd UniCalli
pip install -r requirements.txt
```

### 下载模型

从 Hugging Face 下载预训练模型：

```bash
# 使用 huggingface-cli
huggingface-cli download TSXu/UniCalli-base unicalli-base_cleaned.bin --local-dir ./checkpoints
```

或从魔搭社区下载：

```bash
# 使用 modelscope
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('tianshuo/UniCalli-base', local_dir='./checkpoints')"
```

### 下载其他组件

请注意，您需要下载额外的组件以确保模型正常运行：

```bash
# InternVL3-1B:
https://huggingface.co/OpenGVLab/InternVL3-1B

# 方正字体 TTF:
https://www.fonts.net.cn/font-31659110985.html
MD5: 579e8932d773f5f58ebb2c643aa89ba9
```

## 使用方法

### 运行演示 (Gradio 界面)

```bash
python app.py
```

### 4-bit量化 (GPU Memory < 18G)

⚠️ **注意**: 4-bit量化会影响生成质量，如果对质量要求较高，可以运行deepspeed版。

```bash
pip install optimum-quanto
```

```python
from inference import CalligraphyGenerator

# 初始化生成器
generator = CalligraphyGenerator(
    model_name="flux-dev",
    device="cuda",
    offload=False,
    intern_vlm_path="path/to/InternVL3-1B",
    checkpoint_path="unicalli-base_cleaned.bin",
    font_descriptions_path='dataset/chirography.json',
    author_descriptions_path='dataset/calligraphy_styles_en.json'
    use_deepspeed=False,
    use_4bit_quantization=True,  # Enable 4-bit quantization
)

# 生成书法（必须是5个字符）
image, cond_img = generator.generate(
    text="生日快乐喵",  # Must be 5 characters
    font_style="草",    # 楷(Regular)/草(Cursive)/行(Running)
    author="黄庭坚",    # Or None to use synthetic style
    save_path="output.png",
    num_steps=25,
    seed=42,
)
```

### 使用 DeepSpeed 进行内存优化 (GPU Memory < 40G)

对于大型模型或有限的 GPU 内存，可以使用 DeepSpeed ZeRO：

```python
from inference import CalligraphyGenerator

generator = CalligraphyGenerator(
    model_name="flux-dev",
    device="cuda",
    offload=False,  # DeepSpeed 管理内存
    intern_vlm_path="path/to/InternVL3-1B",
    checkpoint_path="unicalli-base_cleaned.bin",
    font_descriptions_path='dataset/chirography.json',
    author_descriptions_path='dataset/calligraphy_styles_en.json',
    use_deepspeed=True,
    deepspeed_config="ds_config_zero2.json"
)

# 正常生成
image, cond_img = generator.generate(
    text="生日快乐喵",  # 必须是5个字符
    font_style="楷",    # 楷(楷书)/草(草书)/行(行书)
    author="赵佶",    # 或 None 使用合成风格
    save_path="output.png",
    num_steps=39,
    seed=1128293374,
)
```

### 支持的字体风格

- **楷 (楷书 / Kaishu)**: 标准的方块字体
- **行 (行书 / Xingshu)**: 半草书、流畅风格
- **草 (草书 / Caoshu)**: 高度草书、艺术风格

### 支持的书法大师

该模型支持各种历史书法大师，包括：
- 王羲之 (Wang Xizhi) - "书圣"
- 颜真卿 (Yan Zhenqing) - 唐代大师
- 欧阳询 (Ouyang Xun) - 四大楷书家之一
- 赵佶 (Emperor Huizong) - 宋代皇帝和书法家
- 以及更多...

您也可以使用 `author=None` 来生成合成的平均风格。

## 模型详情

- **基础架构**: FLUX 扩散模型
- **模型大小**: ~23GB
- **输入**: 文本（5个字符）、字体风格、作者风格
- **输出**: 列级书法图像
- **训练数据**: 带有密集标注的大规模中国书法数据集

## 引用

如果您觉得 UniCalli 对您的研究有用，请考虑引用：

```bibtex
@article{xu2025unicalli,
  title={UniCalli: A Unified Diffusion Framework for Column-Level Generation and Recognition of Chinese Calligraphy},
  author={Xu, Tianshuo and Wang, Kai and Chen, Zhifei and Wu, Leyi and Wen, Tianshui and Chao, Fei and Chen, Ying-Cong},
  journal={arXiv preprint arXiv:2025.13745},
  year={2025}
}
```

## 致谢

本工作基于 FLUX 架构，并受益于中国书法的丰富传统。我们感谢书法大师们的作品使这项研究成为可能。
