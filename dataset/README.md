# UniCalli Dataset

A large-scale Chinese calligraphy dataset with dense annotations, built for the [UniCalli](https://github.com/EnVision-Research/UniCalli) project.

## Overview

The dataset contains Chinese calligraphy images collected from historical works spanning multiple dynasties, covering **95+ calligraphy masters** and **5 major script styles**. Each sample is annotated with bounding boxes, modern text transcriptions, author attribution, and font style labels.

### Script Styles

| Style | Name | Description |
|:---:|:---:|:---|
| 楷 | Regular Script (Kaishu) | Symmetrical structure, stable center of gravity, clear gaps between strokes |
| 行 | Running Script (Xingshu) | Blends structure and freedom, fluid strokes with moderate simplification |
| 草 | Cursive Script (Caoshu) | Coherent and smooth strokes, free structure, expressive rhythm |
| 隶 | Clerical Script (Lishu) | Flat, wide strokes with distinctive flared brush endings |
| 篆 | Seal Script (Zhuanshu) | Uniform and symmetrical strokes, rounded or angular lines, highly decorative |

### Calligraphy Masters

The dataset covers works from 95+ historical calligraphy masters across Chinese history, including but not limited to:

- **王羲之** (Wang Xizhi) — "Sage of Calligraphy", Jin Dynasty
- **颜真卿** (Yan Zhenqing) — Tang Dynasty master
- **欧阳询** (Ouyang Xun) — One of the Four Great Masters of Tang
- **赵孟頫** (Zhao Mengfu) — Yuan Dynasty master
- **赵佶 / 宋徽宗** (Emperor Huizong) — Song Dynasty emperor and calligrapher
- **怀素** (Huai Su) — Tang Dynasty monk, master of wild cursive
- **张旭** (Zhang Xu) — "Crazy Zhang", Tang Dynasty kuangcao master
- **黄庭坚** (Huang Tingjian) — Song Dynasty master

The full list of supported authors and their font styles is available in [`author_fonts_summary.csv`](author_fonts_summary.csv).

## Files

| File | Description |
|:---|:---|
| `calligraphy_styles_en.json` | English descriptions of each calligraphy master's style and historical context |
| `chirography.json` | Descriptions of the 5 major script styles |
| `author_fonts_summary.csv` | Mapping of calligraphy masters to their supported script styles |

## Download

The dataset is available on HuggingFace:

- **HuggingFace Dataset**: [https://huggingface.co/datasets/TSXu/UniCalli_dataset](https://huggingface.co/datasets/TSXu/UniCalli_dataset)

## Webpage

- **Project Page**: [https://envision-research.github.io/UniCalli/](https://envision-research.github.io/UniCalli/)
- **GitHub**: [https://github.com/EnVision-Research/UniCalli](https://github.com/EnVision-Research/UniCalli)
- **HuggingFace Model**: [https://huggingface.co/TSXu/Unicalli_Pro](https://huggingface.co/TSXu/Unicalli_Pro)
- **HuggingFace Dataset**: [https://huggingface.co/datasets/TSXu/UniCalli_dataset](https://huggingface.co/datasets/TSXu/UniCalli_dataset)
- **HuggingFace Demo**: [https://huggingface.co/spaces/TSXu/UniCalli_Dev](https://huggingface.co/spaces/TSXu/UniCalli_Dev)
- **ModelScope**: [https://www.modelscope.cn/models/tianshuo/UniCalli-base](https://www.modelscope.cn/models/tianshuo/UniCalli-base)
- **arXiv**: [https://arxiv.org/abs/2510.13745](https://arxiv.org/abs/2510.13745)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{xu2025unicalli,
  title={UniCalli: A Unified Diffusion Framework for Column-Level Generation and Recognition of Chinese Calligraphy},
  author={Xu, Tianshuo and Wang, Kai and Chen, Zhifei and Wu, Leyi and Wen, Tianshui and Chao, Fei and Chen, Ying-Cong},
  journal={arXiv preprint arXiv:2025.13745},
  year={2025}
}
```

## License

cc-by-nc-nd-4.0

For academic research and non-commercial use only.
