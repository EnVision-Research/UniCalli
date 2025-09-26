import os
import cv2
from pypinyin import lazy_pinyin
from PIL import Image
import numpy as np
import ast
import random


def _morph_clean(mask255, ksize=3, open_iters=1, close_iters=1, min_area=64):
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

def _score_mask(mask255, edge255, target_fg_range=(0.03, 0.35)):
    r = mask255.mean() / 255.0
    lo, hi = target_fg_range
    mid = (lo + hi) / 2
    width = (hi - lo) / 2 if hi > lo else 0.5
    closeness = 1.0 - min(abs(r - mid) / (width + 1e-6), 1.0)

    if mask255.any():
        inside = edge255[mask255 > 0].mean() / 255.0
    else:
        inside = 0.0
    if (mask255 == 0).any():
        outside = edge255[mask255 == 0].mean() / 255.0
    else:
        outside = 0.0
    edge_contrast = max(inside - outside, 0.0)

    return 0.7 * closeness + 0.3 * edge_contrast

def binarize_auto_polarity(
    pil_img: Image.Image,
    use_adaptive: bool = False,
    target_fg_range=(0.03, 0.35), 
    ksize=3, open_iters=1, close_iters=1,
    min_area=64
):
    g = np.array(pil_img.convert("L"), dtype=np.uint8)
    g = cv2.medianBlur(g, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_eq = clahe.apply(g)

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
        _, cand_B = cv2.threshold(g_eq, 0, 255, cv2.THRESH_BINARY     | cv2.THRESH_OTSU)

    cand_A = _morph_clean(cand_A, ksize, open_iters, close_iters, min_area)
    cand_B = _morph_clean(cand_B, ksize, open_iters, close_iters, min_area)

    edges = cv2.Canny(g_eq, 50, 150)
    score_A = _score_mask(cand_A, edges, target_fg_range)
    score_B = _score_mask(cand_B, edges, target_fg_range)

    if score_A >= score_B:  # all return black bg, white glyphs
        return cand_A, "black"
    else:
        return cand_B, "black"

def convert_to_pinyin(text, with_tone=False):
    return ' '.join([item[0] if isinstance(item, list) else item for item in lazy_pinyin(text)])

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img

def process_image_row(row, abs_path, required_chars=5, col_threshold=50, padding=5):
    img = Image.open(os.path.join(abs_path, row['img_path']))
    w, h = img.size

    locations = ast.literal_eval(row['location'])
    
    sorted_chars = sorted(locations, key=lambda x: x['p'][0])
    columns = []
    current_col = [sorted_chars[0]]
    
    for char in sorted_chars[1:]:
        if char['p'][0] - current_col[-1]['p'][0] > col_threshold:
            columns.append(current_col)
            current_col = [char]
        else:
            current_col.append(char)
    columns.append(current_col)

    valid_columns = [col for col in columns if len(col) >= required_chars]
    if not valid_columns:
        return None, None, None, None

    selected_col = random.choice(valid_columns)
    selected_index = valid_columns.index(selected_col)
    selected_col = sorted(selected_col, key=lambda x: x['p'][1])
    
    start_idx = random.randint(0, len(selected_col) - required_chars)
    selected_chars = selected_col[start_idx:start_idx+required_chars]
    
    # de-normalize
    for i in range(len(selected_chars)):
        selected_chars[i]['p'][0] = int(selected_chars[i]['p'][0] * w / 1000)
        selected_chars[i]['p'][1] = int(selected_chars[i]['p'][1] * h / 1000)
        selected_chars[i]['p'][2] = int(selected_chars[i]['p'][2] * w / 1000)
        selected_chars[i]['p'][3] = int(selected_chars[i]['p'][3] * h / 1000)

    x_coords = [c['p'][0] for c in selected_chars] + [c['p'][2] for c in selected_chars]
    y_coords = [c['p'][1] for c in selected_chars] + [c['p'][3] for c in selected_chars]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    crop_box = (
        max(0, x_min - padding),
        max(0, y_min - padding),
        min(img.width, x_max + padding),
        min(img.height, y_max + padding)
    )

    cropped_img = img.crop(crop_box)
    
    new_locations = []
    texts = ''
    for char in selected_chars:
        new_p = [
            char['p'][0] - crop_box[0],
            char['p'][1] - crop_box[1],
            char['p'][2] - crop_box[0],
            char['p'][3] - crop_box[1]
        ]
        new_locations.append({'c': char['c'], 'p': new_p})
        texts += char['c']

    return cropped_img, new_locations, row['chirography'], row['author']