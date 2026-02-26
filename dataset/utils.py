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
    use_adaptive: bool = False,
    target_fg_range=(0.03, 0.35),
    ksize=3, open_iters=1, close_iters=1,
    min_area=64
):
    """
    Binarize image with automatic polarity detection
    Always returns white text on black background
    """
    # Convert to grayscale and apply preprocessing
    g = np.array(pil_img.convert("L"), dtype=np.uint8)
    g = cv2.medianBlur(g, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_eq = clahe.apply(g)

    # Generate two candidates: inverted and normal binary threshold
    if use_adaptive:
        cand_A = cv2.adaptiveThreshold(
            g_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=25, C=10
        )
        cand_B = cv2.adaptiveThreshold(
            g_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=25, C=10
        )
    else:
        _, cand_A = cv2.threshold(g_eq, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, cand_B = cv2.threshold(g_eq, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Clean up noise
    cand_A = morph_clean(cand_A, ksize, open_iters, close_iters, min_area)
    cand_B = morph_clean(cand_B, ksize, open_iters, close_iters, min_area)

    # Score candidates
    edges = cv2.Canny(g_eq, 50, 150)
    score_A = score_mask(cand_A, edges, target_fg_range)
    score_B = score_mask(cand_B, edges, target_fg_range)

    # Return the better candidate (always black background with white text)
    return (cand_A, "black") if score_A >= score_B else (cand_B, "black")


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
