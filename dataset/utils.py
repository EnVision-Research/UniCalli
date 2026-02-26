"""
Utility functions for image processing and text manipulation
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pypinyin import lazy_pinyin


def convert_to_pinyin(text, with_tone=False):
    """Convert Chinese text to Pinyin"""
    return ' '.join([item[0] if isinstance(item, list) else item for item in lazy_pinyin(text)])


def morph_clean(mask255, ksize=3, open_iters=1, close_iters=1, min_area=64):
    """Apply morphological operations to clean up a binary mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if open_iters > 0:
        mask255 = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    if close_iters > 0:
        mask255 = cv2.morphologyEx(mask255, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    if min_area and min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask255 > 0).astype(np.uint8), 8)
        keep = np.zeros_like(mask255, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        mask255 = keep
    return mask255


def score_mask(mask255, edge255, target_fg_range=(0.03, 0.35)):
    """Score a binary mask based on coverage and edge contrast"""
    r = mask255.mean() / 255.0
    lo, hi = target_fg_range
    mid = (lo + hi) / 2
    width = (hi - lo) / 2 if hi > lo else 0.5
    closeness = 1.0 - min(abs(r - mid) / (width + 1e-6), 1.0)

    inside = edge255[mask255 > 0].mean() / 255.0 if mask255.any() else 0.0
    outside = edge255[mask255 == 0].mean() / 255.0 if (mask255 == 0).any() else 0.0
    edge_contrast = max(inside - outside, 0.0)

    return 0.7 * closeness + 0.3 * edge_contrast


def binarize_auto_polarity(
    pil_img: Image.Image,
    box_img: Image.Image = None,
    bg_offset=10,
    close_iters=1,
    ksize=3,
):
    """
    利用 box 标注区分文字/背景，计算精准阈值做二值化。
    
    策略：
    - 从 box_img 提取文字区域掩码（非黑色像素 = 文字区域，box背景为黑色）
    - 计算 box 内（文字）和 box 外（背景）的灰度均值
    - 阈值 = 两者之间，偏向背景侧 bg_offset 个灰度值
    - 自动检测极性（亮底暗字 or 暗底亮字）
    
    如果没有 box_img，退化为 Otsu。
    
    Returns: (binary_mask_uint8, polarity_str)
        binary_mask: 白色前景(文字)在黑色背景上, uint8 [0,255]
        polarity: "white" (亮底) or "black" (暗底)
    """
    g = np.array(pil_img.convert("L"), dtype=np.uint8)

    # 轻微去噪
    g_blur = cv2.medianBlur(g, 3)

    # 检测极性
    border = np.concatenate([g[:5].flatten(), g[-5:].flatten(),
                             g[:, :5].flatten(), g[:, -5:].flatten()])
    polarity = "black" if border.mean() < 128 else "white"

    if box_img is not None and box_img.size == pil_img.size:
        # 从 box_img 提取文字区域掩码（box背景为黑色，非黑色像素 = 文字区域）
        box_arr = np.array(box_img.convert("RGB"))
        text_mask = ~((box_arr[:, :, 0] < 15) & (box_arr[:, :, 1] < 15) & (box_arr[:, :, 2] < 15))

        if text_mask.any() and (~text_mask).any():
            mean_text = float(g_blur[text_mask].mean())
            mean_bg = float(g_blur[~text_mask].mean())

            if mean_bg > mean_text:
                # 亮底暗字：阈值在背景下方一点，低于阈值 = 前景
                thresh = int(max(0, mean_bg - bg_offset))
                _, binary = cv2.threshold(g_blur, thresh, 255, cv2.THRESH_BINARY_INV)
            else:
                # 暗底亮字：阈值在背景上方一点，高于阈值 = 前景
                thresh = int(min(255, mean_bg + bg_offset))
                _, binary = cv2.threshold(g_blur, thresh, 255, cv2.THRESH_BINARY)
        else:
            # 掩码异常，退化到 Otsu
            _, binary = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        # 没有 box_img，退化到 Otsu
        _, binary = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if polarity == "black":
            binary = 255 - binary

    # 只做闭运算填充笔画内小空洞，不做开运算（保留细节）
    if close_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    return (binary, polarity)


def is_chinese_char(ch: str) -> bool:
    """Check if a character is Chinese"""
    cp = ord(ch)
    ranges = (
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # Extension B
        (0x2A700, 0x2B73F), # Extension C
        (0x2B740, 0x2B81F), # Extension D
        (0x2B820, 0x2CEAF), # Extension E
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    )
    return any(start <= cp <= end for start, end in ranges)


def normalize_tensor(img_array):
    """Normalize image array to tensor range [-1, 1]"""
    return (img_array / 127.5) - 1


def create_image_with_text(size, bg_color, text_color, font, texts, font_scale):
    """Create an image with text at specified positions"""
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    font_size = size[0]  # Assuming square cells

    for i, text in enumerate(texts):
        if text:  # Only draw non-empty text
            font_space = font_size * (1 - font_scale) // 2
            font_position = (font_space, font_size * i + font_space)
            draw.text(font_position, text, font=font, fill=text_color)

    return img
