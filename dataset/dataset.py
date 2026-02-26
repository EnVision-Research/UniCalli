import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import csv

from utils import (
    convert_to_pinyin,
    binarize_auto_polarity,
    is_chinese_char,
    normalize_tensor
)

# Page-level 尺寸配置: (cols, rows_per_col) -> (width, height)
# 总字数不超过20，支持多种极端比例
# 设计离散的尺寸选项，便于模型学习
PAGE_SIZE_OPTIONS = [
    # (cols, rows, total_chars, width_multiplier, height_multiplier)
    # === 1列（窄长） ===
    (1, 3, 3, 1, 3),      # 1列3行 = 3字
    (1, 4, 4, 1, 4),      # 1列4行 = 4字
    (1, 5, 5, 1, 5),      # 1列5行 = 5字
    (1, 6, 6, 1, 6),      # 1列6行 = 6字
    (1, 7, 7, 1, 7),      # 1列7行 = 7字
    (1, 8, 8, 1, 8),      # 1列8行 = 8字
    (1, 9, 9, 1, 9),      # 1列9行 = 9字
    (1, 10, 10, 1, 10),   # 1列10行 = 10字
    # === 2列 ===
    (2, 3, 6, 2, 3),      # 2列3行 = 6字
    (2, 4, 8, 2, 4),      # 2列4行 = 8字
    (2, 5, 10, 2, 5),     # 2列5行 = 10字
    (2, 6, 12, 2, 6),     # 2列6行 = 12字
    (2, 7, 14, 2, 7),     # 2列7行 = 14字
    (2, 8, 16, 2, 8),     # 2列8行 = 16字
    (2, 9, 18, 2, 9),     # 2列9行 = 18字 (极端窄长)
    (2, 10, 20, 2, 10),   # 2列10行 = 20字
    # === 3列 ===
    (3, 3, 9, 3, 3),      # 3列3行 = 9字
    (3, 4, 12, 3, 4),     # 3列4行 = 12字
    (3, 5, 15, 3, 5),     # 3列5行 = 15字
    (3, 6, 18, 3, 6),     # 3列6行 = 18字
    # === 4列 ===
    (4, 3, 12, 4, 3),     # 4列3行 = 12字
    (4, 4, 16, 4, 4),     # 4列4行 = 16字
    (4, 5, 20, 4, 5),     # 4列5行 = 20字
    # === 5列以上（横宽极端） ===
    (5, 2, 10, 5, 2),     # 5列2行 = 10字
    (5, 3, 15, 5, 3),     # 5列3行 = 15字
    (5, 4, 20, 5, 4),     # 5列4行 = 20字
    (6, 2, 12, 6, 2),     # 6列2行 = 12字
    (6, 3, 18, 6, 3),     # 6列3行 = 18字
    (7, 2, 14, 7, 2),     # 7列2行 = 14字
    (8, 2, 16, 8, 2),     # 8列2行 = 16字
    (9, 2, 18, 9, 2),     # 9列2行 = 18字 (极端横宽)
    (10, 2, 20, 10, 2),   # 10列2行 = 20字
]

def get_page_size(img_size, cols, rows):
    """根据列数和行数计算页面尺寸"""
    return (img_size * cols, img_size * rows)


def load_noise_scores(csv_path):
    """加载每个目录的平均噪声分数"""
    noise_scores = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dir_name = row['Directory']
                avg_noise = float(row['Avg_Noise'])
                noise_scores[dir_name] = avg_noise
        print(f"Loaded noise scores for {len(noise_scores)} directories")
    except Exception as e:
        print(f"Warning: Could not load noise scores: {e}")
        noise_scores = {}
    return noise_scores


def load_labelme_annotations(json_path):
    """
    读取labelme格式的json标注文件
    返回字符位置和文本信息
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        locations = []
        # 处理labelme格式的shapes
        if 'shapes' in data:
            for shape in data['shapes']:
                if shape['shape_type'] == 'rectangle' and len(shape['points']) == 2:
                    # 提取矩形的两个对角点
                    (x1, y1), (x2, y2) = shape['points']
                    # 确保坐标顺序正确
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)

                    locations.append({
                        'c': shape['label'],  # 字符内容
                        'p': [int(x_min), int(y_min), int(x_max), int(y_max)]  # 坐标
                    })

        # 去除重复坐标的字符（保留第一个）
        seen_coords = set()
        unique_locations = []
        for char in locations:
            coord_tuple = tuple(char['p'])
            if coord_tuple not in seen_coords:
                seen_coords.add(coord_tuple)
                unique_locations.append(char)

        return unique_locations

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading labelme file {json_path}: {e}")
        return []

def find_labelme_json(img_path, img_dir):
    """根据图片路径查找对应的labelme json文件"""
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_subdir = os.path.dirname(img_path)
    json_path = os.path.join(img_dir, img_subdir, f"{img_name}.json")
    return json_path if os.path.exists(json_path) else None


class CharBox:
    """
    统一的字符框标准，用于建立书法字和标准字的对应关系
    
    标准：
    - box 定义字符的外框位置和大小
    - 书法字图像填充整个 box
    - 标准字（condition）居中放置在 box 内，大小为 box 的 font_scale 倍
    - 支持 scale 变换（宽、高可以独立调整）
    """
    def __init__(self, x, y, w, h, scale_w=1.0, scale_h=1.0):
        """
        x, y: box 左上角位置
        w, h: 基础宽高（通常是 cell_size）
        scale_w, scale_h: 宽度和高度的缩放因子
        """
        self.x = x
        self.y = y
        self.base_w = w
        self.base_h = h
        self.scale_w = scale_w
        self.scale_h = scale_h
    
    @property
    def width(self):
        return int(self.base_w * self.scale_w)
    
    @property
    def height(self):
        return int(self.base_h * self.scale_h)
    
    @property
    def box(self):
        """返回 (x, y, x+w, y+h) 格式的框"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def center(self):
        """返回框的中心点"""
        return (self.x + self.width // 2, self.y + self.height // 2)


def render_char_in_box(canvas, char_box, char_img=None, char_text=None, 
                       font=None, font_scale=0.8, fill_color=(255, 255, 255),
                       is_calligraphy=True):
    """
    在指定的 CharBox 内渲染字符
    
    Args:
        canvas: PIL Image 对象（会被修改）
        char_box: CharBox 对象
        char_img: 书法字图像（PIL Image），仅当 is_calligraphy=True 时使用
        char_text: 字符文本，用于绘制标准字
        font: 字体对象
        font_scale: 字体相对于 box 的缩放比例
        fill_color: 文字颜色
        is_calligraphy: True 表示渲染书法字图像，False 表示渲染标准文字
    
    Returns:
        修改后的 canvas
    """
    if is_calligraphy and char_img is not None:
        # 渲染书法字：将图像缩放到 box 大小并粘贴
        resized_img = char_img.resize((char_box.width, char_box.height), Image.LANCZOS)
        canvas.paste(resized_img, (char_box.x, char_box.y))
    
    elif not is_calligraphy and char_text and font:
        # 渲染标准字：居中放置在 box 内
        draw = ImageDraw.Draw(canvas)
        
        # 计算字体大小：基于 box 的最小边长
        min_dim = min(char_box.width, char_box.height)
        target_font_size = int(min_dim * font_scale)
        
        # 尝试创建对应大小的字体
        try:
            # 如果 font 有 path 属性，创建新字体
            if hasattr(font, 'path'):
                sized_font = ImageFont.truetype(font.path, target_font_size)
            else:
                sized_font = font
        except:
            sized_font = font
        
        # 获取文字的 bounding box 来计算居中位置
        try:
            bbox = draw.textbbox((0, 0), char_text, font=sized_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except:
            text_w = target_font_size
            text_h = target_font_size
        
        # 居中放置
        text_x = char_box.x + (char_box.width - text_w) // 2
        text_y = char_box.y + (char_box.height - text_h) // 2
        
        draw.text((text_x, text_y), char_text, font=sized_font, fill=fill_color)
    
    return canvas


def generate_random_char_boxes(canvas_w, canvas_h, num_chars, base_cell_size,
                               scale_range=(0.8, 1.2), 
                               aspect_ratio_range=(0.85, 1.15),
                               layout_mode='grid', margin=5):
    """
    生成随机的字符框列表
    
    Args:
        canvas_w, canvas_h: 画布大小
        num_chars: 字符数量
        base_cell_size: 基础单元格大小
        scale_range: 整体缩放范围 (min, max)
        aspect_ratio_range: 宽高比变化范围 (min, max)
        layout_mode: 布局模式 ('grid', 'column', 'random', 'scatter')
        margin: 边距
    
    Returns:
        list of CharBox
    """
    char_boxes = []
    
    if layout_mode == 'grid':
        # 网格布局：规则排列，但每个格子可以有随机 scale
        cols = canvas_w // base_cell_size
        rows = canvas_h // base_cell_size
        num_chars = min(num_chars, cols * rows)
        
        # 随机选择 num_chars 个格子，消除位置偏向
        all_cells = [(c, r) for c in range(cols) for r in range(rows)]
        random.shuffle(all_cells)
        selected_cells = all_cells[:num_chars]
        # 按从右到左、从上到下排序（中文书法阅读顺序）
        selected_cells.sort(key=lambda cr: (cr[0], cr[1]))
        
        for col, row in selected_cells:
            # 随机 scale
            scale = random.uniform(*scale_range)
            aspect = random.uniform(*aspect_ratio_range)
            scale_w = scale * aspect
            scale_h = scale / aspect
            
            # 基础位置（从右到左）
            base_x = (cols - 1 - col) * base_cell_size
            base_y = row * base_cell_size
            
            # 计算居中偏移（让缩放后的字符仍然在格子中心）
            offset_x = int((base_cell_size - base_cell_size * scale_w) / 2)
            offset_y = int((base_cell_size - base_cell_size * scale_h) / 2)
            
            char_boxes.append(CharBox(
                x=base_x + offset_x,
                y=base_y + offset_y,
                w=base_cell_size,
                h=base_cell_size,
                scale_w=scale_w,
                scale_h=scale_h
            ))
                
    elif layout_mode == 'column':
        # 列布局：1-2 列，每列字符大小可以变化
        max_cols = max(1, canvas_w // base_cell_size)
        num_cols = random.choice([1, 2]) if max_cols >= 2 else 1
        chars_per_col = (num_chars + num_cols - 1) // num_cols
        max_rows = canvas_h // base_cell_size
        chars_per_col = min(chars_per_col, max_rows)
        
        # 随机选择列的起始位置，消除右侧偏向
        available_col_indices = list(range(max_cols))
        random.shuffle(available_col_indices)
        selected_col_indices = sorted(available_col_indices[:num_cols], reverse=True)  # 从右到左排列
        
        char_idx = 0
        for col_idx in selected_col_indices:
            col_x = col_idx * base_cell_size
            for row in range(chars_per_col):
                if char_idx >= num_chars:
                    break
                
                scale = random.uniform(*scale_range)
                aspect = random.uniform(*aspect_ratio_range)
                scale_w = scale * aspect
                scale_h = scale / aspect
                
                # 列内有轻微随机偏移
                x_jitter = random.randint(-5, 5)
                y_jitter = random.randint(-3, 3)
                
                char_boxes.append(CharBox(
                    x=col_x + x_jitter,
                    y=row * base_cell_size + y_jitter,
                    w=base_cell_size,
                    h=base_cell_size,
                    scale_w=scale_w,
                    scale_h=scale_h
                ))
                char_idx += 1
                
    elif layout_mode == 'random':
        # 随机位置，但保持阅读顺序
        positions = []
        
        for _ in range(num_chars * 3):
            if len(positions) >= num_chars:
                break
            
            scale = random.uniform(*scale_range)
            aspect = random.uniform(*aspect_ratio_range)
            scale_w = scale * aspect
            scale_h = scale / aspect
            
            char_w = int(base_cell_size * scale_w)
            char_h = int(base_cell_size * scale_h)
            
            x = random.randint(margin, max(margin + 1, canvas_w - char_w - margin))
            y = random.randint(margin, max(margin + 1, canvas_h - char_h - margin))
            
            # 检查重叠
            overlap = False
            for px, py, pw, ph, _, _ in positions:
                if (abs(x - px) < (char_w + pw) * 0.4 and 
                    abs(y - py) < (char_h + ph) * 0.4):
                    overlap = True
                    break
            
            if not overlap:
                positions.append((x, y, char_w, char_h, scale_w, scale_h))
        
        # 按从右到左、从上到下排序
        positions.sort(key=lambda p: (-p[0], p[1]))
        
        for x, y, _, _, sw, sh in positions:
            char_boxes.append(CharBox(
                x=x, y=y,
                w=base_cell_size, h=base_cell_size,
                scale_w=sw, scale_h=sh
            ))
            
    else:  # scatter
        # 散落式，更大的 scale 变化
        wide_scale_range = (0.6, 1.4)
        positions = []
        
        for _ in range(num_chars * 3):
            if len(positions) >= num_chars:
                break
            
            scale = random.uniform(*wide_scale_range)
            aspect = random.uniform(0.75, 1.25)
            scale_w = scale * aspect
            scale_h = scale / aspect
            
            char_w = int(base_cell_size * scale_w)
            char_h = int(base_cell_size * scale_h)
            
            x = random.randint(margin, max(margin + 1, canvas_w - char_w - margin))
            y = random.randint(margin, max(margin + 1, canvas_h - char_h - margin))
            
            overlap = False
            for px, py, pw, ph, _, _ in positions:
                if (abs(x - px) < (char_w + pw) * 0.35 and 
                    abs(y - py) < (char_h + ph) * 0.35):
                    overlap = True
                    break
            
            if not overlap:
                positions.append((x, y, char_w, char_h, scale_w, scale_h))
        
        for x, y, _, _, sw, sh in positions:
            char_boxes.append(CharBox(
                x=x, y=y,
                w=base_cell_size, h=base_cell_size,
                scale_w=sw, scale_h=sh
            ))
    
    return char_boxes[:num_chars]

def process_image_row(row, abs_path, required_chars=5, col_threshold=50, padding=5):
    """
    处理单个图片行，支持从labelme json文件读取标注（单列模式，向后兼容）
    row: 包含img_path, chirography, author等信息的数据行
    abs_path: 图片目录的绝对路径
    """
    return process_image_row_page(row, abs_path, target_cols=1, target_rows=required_chars, 
                                   col_threshold=col_threshold, padding=padding)


def process_image_row_page(row, abs_path, target_cols=1, target_rows=5, col_threshold=50, 
                           padding=5, cell_size=64, font=None, font_scale=0.8,
                           max_total_chars=None):
    """
    处理单个图片行，支持page-level多列多行选择
    同时生成 grid_img, cond_img 和 box_img，确保完全对齐
    
    流程：
    1. 加载所有字符标注
    2. 按列分组，每列内按y坐标排序
    3. 选择要 include 的字符（总数不超过 max_total_chars）
    4. 逐个 crop 字符，同时在对应位置绘制 condition 和 box
    
    返回: (grid_img, cond_img, box_img, texts, chirography, author, layout_info)
    """
    img = Image.open(os.path.join(abs_path, row['img_path']))

    # 查找对应的labelme json文件
    json_path = find_labelme_json(row['img_path'], abs_path)
    if json_path is None:
        return None, None, None, None, None, None, None

    # 从labelme json文件加载标注
    locations = load_labelme_annotations(json_path)
    if not locations:
        return None, None, None, None, None, None, None

    # === Step 1: 分列（根据x坐标聚类）===
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

    # === Step 2: 筛选有效列（检查字符间隔连续性）===
    valid_columns = []
    for col in columns:
        if len(col) >= 2:
            col_sorted = sorted(col, key=lambda x: x['p'][1])

            # 计算平均字符间隔
            gaps = []
            for i in range(1, len(col_sorted)):
                gap = col_sorted[i]['p'][1] - col_sorted[i-1]['p'][3]
                gaps.append(gap)
            avg_gap = sum(gaps) / len(gaps) if gaps else 0

            # 按间隔分割成连续序列
            valid_sequences = []
            current_sequence = [col_sorted[0]]

            for i in range(1, len(col_sorted)):
                gap = col_sorted[i]['p'][1] - col_sorted[i-1]['p'][3]
                if gap > avg_gap * 1.5:
                    if len(current_sequence) >= 2:
                        valid_sequences.append(current_sequence)
                    current_sequence = [col_sorted[i]]
                else:
                    current_sequence.append(col_sorted[i])

            if len(current_sequence) >= 2:
                valid_sequences.append(current_sequence)

            valid_columns.extend(valid_sequences)

    # === Step 3: 选择要 include 的字符 ===
    # 过滤出长度足够的列
    sufficient_columns = [col for col in valid_columns if len(col) >= target_rows]
    
    if not sufficient_columns:
        sufficient_columns = [col for col in valid_columns if len(col) >= 2]
        if not sufficient_columns:
            return None, None, None, None, None, None, None
        max_available_rows = max(len(col) for col in sufficient_columns)
        target_rows = min(target_rows, max_available_rows)

    # 按x坐标排序列（从右到左，符合中文书法阅读顺序）
    sufficient_columns_sorted = sorted(sufficient_columns, 
                                       key=lambda col: sum(c['p'][0] for c in col) / len(col),
                                       reverse=True)  # 从右到左
    
    # 选择相邻的 target_cols 列
    if len(sufficient_columns_sorted) >= target_cols:
        start_col_idx = random.randint(0, len(sufficient_columns_sorted) - target_cols)
        selected_columns = sufficient_columns_sorted[start_col_idx:start_col_idx + target_cols]
    else:
        selected_columns = sufficient_columns_sorted
        target_cols = len(selected_columns)

    # 如果设置了总字数上限，根据列数限制每列行数
    if max_total_chars is not None:
        max_rows_per_col = max(2, max_total_chars // target_cols)
        target_rows = min(target_rows, max_rows_per_col)

    # 从每列中选择连续的 target_rows 个字符
    # 所有列使用同一个 start_idx，确保选中的行在空间上对齐
    # 避免不同列选不同 y 范围导致 crop 过大、image 里有未标注的字
    col_lengths = [len(sorted(col, key=lambda x: x['p'][1])) for col in selected_columns]
    min_col_len = min(col_lengths)
    actual_target_rows = min(target_rows, min_col_len)
    
    if min_col_len > actual_target_rows:
        shared_start_idx = random.randint(0, min_col_len - actual_target_rows)
    else:
        shared_start_idx = 0
    
    selected_chars_2d = []
    for col in selected_columns:
        col_sorted = sorted(col, key=lambda x: x['p'][1])  # 按y坐标排序
        selected = col_sorted[shared_start_idx:shared_start_idx + actual_target_rows]
        selected_chars_2d.append(selected)
    
    actual_rows = actual_target_rows
    actual_cols = len(selected_chars_2d)

    # === Step 3.5: 检测列/行之间的大间距，只保留一个连续子区域 ===
    # 用最大字符尺寸作为间距判断基准（更保守，避免漏切）
    _tmp_w, _tmp_h = [], []
    for col_chars in selected_chars_2d:
        for c in col_chars:
            _tmp_w.append(c['p'][2] - c['p'][0])
            _tmp_h.append(c['p'][3] - c['p'][1])
    max_cw = max(_tmp_w) if _tmp_w else cell_size
    max_ch = max(_tmp_h) if _tmp_h else cell_size

    # 检测列间距：按 x 中心排序，间距 > 2× 最大字宽则切分，随机保留一组
    if len(selected_chars_2d) > 1:
        col_cx = [sum((c['p'][0] + c['p'][2]) / 2 for c in col) / len(col) 
                  for col in selected_chars_2d]
        order = sorted(range(len(col_cx)), key=lambda i: col_cx[i])
        sorted_cols = [selected_chars_2d[i] for i in order]
        sorted_cx = [col_cx[i] for i in order]

        col_groups = [[0]]
        for i in range(1, len(sorted_cx)):
            if sorted_cx[i] - sorted_cx[i - 1] > max_cw * 2:
                col_groups.append([i])
            else:
                col_groups[-1].append(i)

        if len(col_groups) > 1:
            grp = random.choice(col_groups)
            selected_chars_2d = [sorted_cols[i] for i in grp]
            actual_cols = len(selected_chars_2d)

    # 检测行间距：用第一列做参考，间距 > 2× 最大字高则切分，随机保留一组
    if selected_chars_2d and len(selected_chars_2d[0]) > 1:
        ref = selected_chars_2d[0]  # 已按 y 排序
        row_groups = [[0]]
        for i in range(1, len(ref)):
            gap = ref[i]['p'][1] - ref[i - 1]['p'][3]
            if gap > max_ch * 2:
                row_groups.append([i])
            else:
                row_groups[-1].append(i)

        if len(row_groups) > 1:
            grp = random.choice(row_groups)
            selected_chars_2d = [[col[i] for i in grp if i < len(col)]
                                 for col in selected_chars_2d]
            # 移除变空的列
            selected_chars_2d = [col for col in selected_chars_2d if col]
            actual_rows = len(grp)
            actual_cols = len(selected_chars_2d)

    if not selected_chars_2d or not selected_chars_2d[0]:
        return None, None, None, None, None, None, None

    # === Step 4: 直接从原图 crop 选中区域 ===
    # 计算所有选中字符的整体 bounding box
    all_x1, all_y1 = float('inf'), float('inf')
    all_x2, all_y2 = 0, 0
    _init_heights = []
    for col_chars in selected_chars_2d:
        for char_info in col_chars:
            box = char_info['p']
            all_x1 = min(all_x1, box[0])
            all_y1 = min(all_y1, box[1])
            all_x2 = max(all_x2, box[2])
            all_y2 = max(all_y2, box[3])
            _init_heights.append(box[3] - box[1])

    _init_avg_h = sum(_init_heights) / len(_init_heights) if _init_heights else cell_size

    # 随机 padding（字符平均高度的 5%~25%），让 crop 范围有变化
    pad_ratio = random.uniform(0.05, 0.25)
    pad = int(_init_avg_h * pad_ratio)
    crop_x1 = max(0, int(all_x1) - pad)
    crop_y1 = max(0, int(all_y1) - pad)
    crop_x2 = min(img.width, int(all_x2) + pad)
    crop_y2 = min(img.height, int(all_y2) + pad)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w <= 0 or crop_h <= 0:
        return None, None, None, None, None, None, None

    # 关键：收集 crop 区域内的所有标注字符，而不仅仅是之前选中的子集
    # 这样可以确保 image 里能看到的字全都有对应的 condition 和 box
    def _collect_chars_in_crop(locs, cx1, cy1, cx2, cy2):
        result = []
        for loc in locs:
            b = loc['p']
            mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if cx1 <= mx <= cx2 and cy1 <= my <= cy2:
                result.append(loc)
        result.sort(key=lambda c: (-c['p'][0], c['p'][1]))
        return result

    chars_in_crop = _collect_chars_in_crop(locations, crop_x1, crop_y1, crop_x2, crop_y2)

    # 如果 crop 内字符数超过上限，迭代缩小 crop 直到满足限制
    # 每轮把 pad 减半，避免 padding 反复把边缘字符拉回来导致不收敛
    if max_total_chars is not None:
        shrink_pad = pad
        for _ in range(5):
            if len(chars_in_crop) <= max_total_chars:
                break
            # 只保留前 max_total_chars 个字符，从它们重新算 crop
            chars_in_crop = chars_in_crop[:max_total_chars]
            _x1 = min(c['p'][0] for c in chars_in_crop)
            _y1 = min(c['p'][1] for c in chars_in_crop)
            _x2 = max(c['p'][2] for c in chars_in_crop)
            _y2 = max(c['p'][3] for c in chars_in_crop)
            shrink_pad = max(2, shrink_pad // 2)  # 逐步缩小 padding
            crop_x1 = max(0, int(_x1) - shrink_pad)
            crop_y1 = max(0, int(_y1) - shrink_pad)
            crop_x2 = min(img.width, int(_x2) + shrink_pad)
            crop_y2 = min(img.height, int(_y2) + shrink_pad)
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            chars_in_crop = _collect_chars_in_crop(locations, crop_x1, crop_y1, crop_x2, crop_y2)

        # 最终硬性截断：无论如何不超过 max_total_chars
        if len(chars_in_crop) > max_total_chars:
            chars_in_crop = chars_in_crop[:max_total_chars]

    if not chars_in_crop:
        return None, None, None, None, None, None, None

    # 用 crop 内最终字符集重新计算 resize 参数
    char_heights = [c['p'][3] - c['p'][1] for c in chars_in_crop]
    char_widths = [c['p'][2] - c['p'][0] for c in chars_in_crop]
    avg_char_h = sum(char_heights) / len(char_heights)
    avg_char_w = sum(char_widths) / len(char_widths)

    # 等比例 resize：用 min(avg_w, avg_h) 作为 scale 基准
    avg_char_dim = min(avg_char_w, avg_char_h)
    scale = cell_size / avg_char_dim if avg_char_dim > 0 else 1.0
    target_w = max(16, round(crop_w * scale / 16) * 16)
    target_h = max(16, round(crop_h * scale / 16) * 16)

    # 限制最大图片面积：width * height <= max_total_chars * cell_size^2
    # 例如 required_chars=20, cell_size=64 → 最大面积 = 20 * 64^2 = 81920
    if max_total_chars is not None:
        max_area = max_total_chars * cell_size * cell_size
        cur_area = target_w * target_h
        if cur_area > max_area:
            down_scale = (max_area / cur_area) ** 0.5
            target_w = max(16, round(target_w * down_scale / 16) * 16)
            target_h = max(16, round(target_h * down_scale / 16) * 16)
            scale = scale * down_scale  # 更新 scale 用于后续字体大小计算

    # crop 和 resize
    grid_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    grid_img = grid_img.resize((target_w, target_h), Image.LANCZOS)

    # 创建 cond_img 和 box_img
    # box_img 使用黑色背景，与书法图片背景一致，避免 attention 泄漏白色背景到生成图片中
    cond_img = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    box_img = Image.new('RGB', (target_w, target_h), (0, 0, 0))
    cond_draw = ImageDraw.Draw(cond_img)
    box_draw = ImageDraw.Draw(box_img)

    # condition 字体大小：始终基于 cell_size，不受画布缩放影响
    unified_cond_font_size = int(cell_size * font_scale)
    try:
        unified_font = ImageFont.truetype(font.path if hasattr(font, 'path') else str(font), unified_cond_font_size)
    except:
        unified_font = font

    texts = ''
    for char_info in chars_in_crop:
        char_text = char_info['c']
        box = char_info['p']  # [x1, y1, x2, y2]

        # 字符在 crop 中的相对位置 → 映射到 resize 后的尺寸
        new_x1 = int((box[0] - crop_x1) / crop_w * target_w)
        new_y1 = int((box[1] - crop_y1) / crop_h * target_h)
        new_x2 = int((box[2] - crop_x1) / crop_w * target_w)
        new_y2 = int((box[3] - crop_y1) / crop_h * target_h)
        char_box_w = new_x2 - new_x1
        char_box_h = new_y2 - new_y1

        # 绘制 box_img：随机亮色填充字符区域（黑色背景上需要保证颜色足够亮）
        random_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        box_draw.rectangle([(new_x1, new_y1), (new_x2, new_y2)], fill=random_color)

        # 绘制 cond_img：condition 字符居中在字符区域内
        if font and char_text:
            try:
                text_bbox = cond_draw.textbbox((0, 0), char_text, font=unified_font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except:
                text_w = unified_cond_font_size
                text_h = unified_cond_font_size

            cond_x = new_x1 + (char_box_w - text_w) // 2
            cond_y = new_y1 + (char_box_h - text_h) // 2
            cond_draw.text((cond_x, cond_y), char_text, font=unified_font, fill=(0, 0, 0))

        texts += char_text

    # 用 crop 内实际字符数重新估算 layout
    n_chars_in_crop = len(chars_in_crop)
    actual_cols = max(1, actual_cols)
    actual_rows = max(1, (n_chars_in_crop + actual_cols - 1) // actual_cols)
    layout_info = (actual_cols, actual_rows)
    return grid_img, cond_img, box_img, texts, row['chirography'], row['author'], layout_info

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False,
            author_descriptions=None, font_scale=0.8, font_size=None, required_chars=7,
            pred_box=False, to_english=True, txt_dir='./libs/text_clips', ttf_dir='./libs/font',
            synth_prob=0.5, data_aug=False, font_descriptions=None,
            noise_threshold=20.0, noise_csv_path=None, page_level=False):
        self.image_path = os.path.join(img_dir, 'images')

        # CSV文件现在只包含: img_path, chirography, author等基本信息
        # location信息从labelme json文件中读取
        csv_path = os.path.join(img_dir, 'data.csv')
        self.samples = pd.read_csv(csv_path)
        print(f"Loaded {len(self.samples)} samples from CSV")

        self.data_aug = data_aug
        self.to_english = to_english
        self.page_level = page_level  # 新增：是否启用page-level generation

        # 新增：噪声阈值参数
        self.noise_threshold = noise_threshold

        # 加载噪声分数
        if noise_csv_path is None:
            noise_csv_path = os.path.join(img_dir, "directory_noise_summary.csv")
        self.noise_scores = load_noise_scores(noise_csv_path)

        # 根据噪声阈值过滤样本
        if self.noise_scores:
            filtered_samples = []
            for idx, row in self.samples.iterrows():
                # 获取图片所在目录
                img_dir_name = os.path.dirname(row['img_path'])
                if not img_dir_name:  # 如果没有子目录，使用文件名的第一部分
                    img_dir_name = row['img_path'].split('/')[0] if '/' in row['img_path'] else ''

                # 检查噪声分数
                dir_noise_score = self.noise_scores.get(img_dir_name, 0)
                if dir_noise_score <= self.noise_threshold:
                    filtered_samples.append(row)
                else:
                    print(f"Filtered out {img_dir_name} (noise score: {dir_noise_score:.2f} > {self.noise_threshold})")

            if filtered_samples:
                self.samples = pd.DataFrame(filtered_samples)
                print(f"After noise filtering: {len(self.samples)} samples remain")
            else:
                print("Warning: No samples passed noise filtering, using all samples")

        assert author_descriptions is not None
        with open(author_descriptions, 'r', encoding='utf-8') as f:
            self.author_style = json.load(f)

        assert font_descriptions is not None
        with open(font_descriptions, 'r', encoding='utf-8') as f:
            self.font_style_des = json.load(f)

        print('Dataset length:', len(self.samples))
        print('Synth_prob:', synth_prob)
        print('author_nums:', len(self.author_style))
        print("data_aug:", self.data_aug)
        print("to_english:", self.to_english)
        print(f"Noise threshold: {self.noise_threshold}")
        print(f"Page-level mode: {self.page_level}")

        # 基础单元格尺寸
        self.cell_size = img_size
        self.img_size = (img_size, img_size * required_chars)  # 默认尺寸（兼容旧模式）

        self.font_path = "./FangZhengKaiTiFanTi-1.ttf"
        self.font_scale = font_scale
        self.font_size = img_size if font_size is None else font_size
        self.font = ImageFont.truetype(self.font_path, int(font_scale * self.font_size))

        self.pred_box = pred_box

        self.required_chars = required_chars
        print(f"Required chars: {self.required_chars}")

        self.bad_indices = []

        # 合成数据参数
        self.synth_prob = synth_prob  # 合成数据占比（概率）
        self.weights=[float(f'0.{i+1}')-0.01 for i in range(required_chars)]

        self.txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]
        if not self.txt_files:
            raise ValueError("未找到任何 .txt 文件，请检查目录。")

        ttf_files = [os.path.join(ttf_dir, f) for f in os.listdir(ttf_dir) if f.lower().endswith('.ttf')]
        if not ttf_files:
            raise ValueError("未找到任何 .ttf 字体文件，请检查目录。")
        self.ttf_fonts = [ImageFont.truetype(ttf_path, int(font_scale * self.font_size)) for ttf_path in ttf_files]
        self.ttf_style = [ttf_path.split('/')[-1].split('.')[0].split('_')[1] for ttf_path in ttf_files]
        
        # Page-level 配置
        if self.page_level:
            # 筛选适合当前数据集的尺寸选项
            # 根据 required_chars 来过滤（总字符数不超过 required_chars 的配置）
            self.page_options = [opt for opt in PAGE_SIZE_OPTIONS if opt[2] <= required_chars]
            if not self.page_options:
                # 如果没有合适的选项，使用单列模式
                self.page_options = [(1, required_chars, required_chars, 1, required_chars)]
            print(f"Page-level options: {self.page_options}")

    def __len__(self):
        return len(self.samples)
    
    def get_random_text_from_txt(self, num_chars=None):
        """获取随机文本，支持指定字符数"""
        max_chars = num_chars if num_chars else self.required_chars
        weights = [float(f'0.{i+1}')-0.01 for i in range(max_chars)]

        file_path = random.choice(self.txt_files)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()

        chinese_chars = [ch for ch in raw if is_chinese_char(ch)]
        if len(chinese_chars) < max_chars:
            return None

        chosen_len = random.choices(range(1, max_chars + 1), weights=weights, k=1)[0]
        start_index = random.randint(0, len(chinese_chars) - chosen_len)
        return ''.join(chinese_chars[start_index: start_index + chosen_len])

    def get_condition(self, locations, origin_img_size, actual_char_count=None, target_size=None):
        """
        原始单列condition生成（向后兼容）
        condition图像中的字符位置与resize后图像中字符的bounding box中心对齐
        
        Args:
            locations: 字符位置列表
            origin_img_size: crop后、resize前的原始图像尺寸 (width, height)
            actual_char_count: 实际字符数
            target_size: resize后的目标尺寸，默认为 self.img_size
        """
        background_color = (255, 255, 255)  # 白色背景
        text_color = (0, 0, 0)  # 黑色文字

        assert len(locations) == self.required_chars
        
        # 如果没有提供actual_char_count，则计算非空字符数
        actual_char_count = actual_char_count or sum(1 for loc in locations if loc['c'])
        
        # 目标尺寸：img实际resize到的尺寸（不含填充）
        if target_size is None:
            target_size = (self.img_size[0], self.img_size[0] * actual_char_count)
        
        # cond_img 尺寸与最终输出尺寸一致
        img = Image.new("RGB", self.img_size, background_color)
        box_img = Image.new("RGB", origin_img_size, background_color)
        draw = ImageDraw.Draw(img)
        draw_box = ImageDraw.Draw(box_img)

        # 计算从原始图像到目标尺寸的缩放比例
        scale_x = target_size[0] / origin_img_size[0]
        scale_y = target_size[1] / origin_img_size[1]
        
        # 获取字体的尺寸用于居中计算
        font_char_size = int(self.font_scale * self.font_size)

        texts = ''
        for i, loc in enumerate(locations):
            text = loc['c']
            if i < actual_char_count and text:  # 实际字符
                texts += text
                
                # 获取字符的bounding box
                box = loc['p']  # [x1, y1, x2, y2]
                
                # 计算box中心点（在原始图像坐标系中）
                center_x_orig = (box[0] + box[2]) / 2
                center_y_orig = (box[1] + box[3]) / 2
                
                # 将中心点坐标转换到resize后的目标尺寸坐标系
                center_x = center_x_orig * scale_x
                center_y = center_y_orig * scale_y
                
                # 计算字符绘制位置（使字符中心对齐到box中心）
                font_position = (
                    int(center_x - font_char_size / 2),
                    int(center_y - font_char_size / 2)
                )
                draw.text(font_position, text, font=self.font, fill=text_color)

                # 在原始图像上绘制box（用于可视化）
                draw_box.rectangle(
                    [(box[0], box[1]), (box[2], box[3])],
                    outline=text_color,
                    width=4
                )

        return img, box_img, texts

    def get_condition_page(self, char_texts_2d, layout_info):
        """
        Page-level condition生成（网格对齐版本）
        char_texts_2d: 二维列表 [[col1_texts], [col2_texts], ...]
        layout_info: (cols, rows)
        
        condition图像按网格生成，与grid_img中的字符位置一一对应
        """
        background_color = (255, 255, 255)  # 白色背景
        text_color = (0, 0, 0)  # 黑色文字
        
        cols, rows = layout_info
        page_size = get_page_size(self.cell_size, cols, rows)
        
        img = Image.new("RGB", page_size, background_color)
        draw = ImageDraw.Draw(img)

        texts = ''
        # 字符绘制在每个cell的中心
        font_char_size = int(self.font_scale * self.font_size)
        font_offset = (self.cell_size - font_char_size) // 2

        # 遍历每列每行，按网格位置绘制字符
        for col_idx, col_texts in enumerate(char_texts_2d):
            for row_idx, text in enumerate(col_texts):
                if text:
                    texts += text
                    
                    # 计算网格位置的中心
                    x_pos = col_idx * self.cell_size + font_offset
                    y_pos = row_idx * self.cell_size + font_offset
                    
                    draw.text((x_pos, y_pos), text, font=self.font, fill=text_color)

        return img, texts, page_size

    def get_real_img(self, idx):
        """获取真实图片（向后兼容单列模式）"""
        if self.page_level:
            return self.get_real_img_page(idx)
        
        img_path = self.samples.iloc[idx]['img_path']
        if img_path in self.bad_indices:
            return self.get_real_img(random.randint(0, len(self.samples) - 1))

        sample_row = self.samples.iloc[idx]
        result = process_image_row(
            sample_row,
            self.image_path,
            required_chars=self.required_chars,
            col_threshold=50,
            padding=5,
        )
        img, new_locs, chirography, author, actual_char_count = result

        # 检查无效情况
        if (not new_locs or
            chirography in ['隶', '篆']):
            self.bad_indices.append(img_path)
            return self.get_real_img(random.randint(0, len(self.samples) - 1))

        # 自动检测背景色（取边缘像素平均亮度）
        arr = np.array(img.convert('L'))
        border = np.concatenate([arr[:5].flatten(), arr[-5:].flatten(),
                                 arr[:, :5].flatten(), arr[:, -5:].flatten()])
        polarity = "black" if border.mean() < 128 else "white"

        # 构建prompt
        prompt = f"Traditional Chinese calligraphy works, background: {polarity}, font: {convert_to_pinyin(chirography)},"
        if chirography in self.font_style_des:
            prompt += ' ' + self.font_style_des[chirography]

        author_info = self.author_style.get(author, convert_to_pinyin(author))
        prompt += f' author: {author_info}' if author in self.author_style else f' author: {author_info}.'

        cond_img, box_img, texts = self.get_condition(new_locs, img.size, actual_char_count)

        # 调整图片尺寸
        img_size = self.img_size[0]
        img = img.resize((img_size, img_size * actual_char_count), Image.LANCZOS)
        if img.mode == 'L':
            img = img.convert('RGB')

        # 如果字符数不足5个，填充黑色
        if actual_char_count < 5:
            actual_img = Image.new('RGB', self.img_size, (0, 0, 0))
            actual_img.paste(img, (0, 0))
            img = actual_img

        box_img = box_img.resize(self.img_size, Image.LANCZOS)
        assert img.size == cond_img.size == self.img_size == box_img.size

        # 转换为tensor
        img = torch.from_numpy(normalize_tensor(np.array(img))).permute(2, 0, 1)
        cond_img = torch.from_numpy(normalize_tensor(np.array(cond_img))).permute(2, 0, 1)
        box_img = torch.from_numpy(normalize_tensor(np.array(box_img))).permute(2, 0, 1)

        if self.pred_box:
            img = torch.cat((img, box_img), dim=2)
            cond_img = torch.cat((cond_img, torch.zeros_like(cond_img)), dim=2)

        return img, prompt, cond_img, texts

    def get_real_img_page(self, idx):
        """获取真实图片（Page-level模式）- 网格对齐版本，包含 box_img"""
        img_path = self.samples.iloc[idx]['img_path']
        if img_path in self.bad_indices:
            return self.get_real_img_page(random.randint(0, len(self.samples) - 1))

        sample_row = self.samples.iloc[idx]
        
        # 用足够大的 target 探测数据实际的完整布局，避免截断
        # 通过 max_total_chars 限制总字数不超过 required_chars
        result = process_image_row_page(
            sample_row,
            self.image_path,
            target_cols=10,
            target_rows=20,
            col_threshold=50,
            padding=5,
            cell_size=self.cell_size,
            font=self.font,
            font_scale=self.font_scale,
            max_total_chars=self.required_chars,
        )
        grid_img, cond_img, box_img, texts, chirography, author, layout_info = result

        # 检查无效情况
        if (grid_img is None or not texts or
            chirography in ['隶', '篆']):
            self.bad_indices.append(img_path)
            return self.get_real_img_page(random.randint(0, len(self.samples) - 1))

        actual_cols, actual_rows = layout_info

        # 二值化：利用 box 标注区分文字/背景，精准计算阈值
        if self.data_aug:
            grid_img, polarity = binarize_auto_polarity(
                grid_img,
                box_img=box_img,
                bg_offset=10,
            )
            grid_img = Image.fromarray(grid_img, mode='L').convert('RGB')
        else:
            # 不二值化时仅检测背景色
            arr = np.array(grid_img.convert('L'))
            border = np.concatenate([arr[:5].flatten(), arr[-5:].flatten(),
                                     arr[:, :5].flatten(), arr[:, -5:].flatten()])
            polarity = "black" if border.mean() < 128 else "white"

        # 构建prompt
        prompt = f"Traditional Chinese calligraphy works, background: {polarity}, font: {convert_to_pinyin(chirography)},"
        if chirography in self.font_style_des:
            prompt += ' ' + self.font_style_des[chirography]

        author_info = self.author_style.get(author, convert_to_pinyin(author))
        prompt += f' author: {author_info}' if author in self.author_style else f' author: {author_info}.'

        # grid_img, cond_img 和 box_img 都是对齐的
        assert grid_img.size == cond_img.size == box_img.size

        # 转换为tensor
        img = torch.from_numpy(normalize_tensor(np.array(grid_img))).permute(2, 0, 1)
        cond_img = torch.from_numpy(normalize_tensor(np.array(cond_img))).permute(2, 0, 1)
        box_img = torch.from_numpy(normalize_tensor(np.array(box_img))).permute(2, 0, 1)

        return img, prompt, cond_img, box_img, texts, layout_info

    def get_syn_img(self):
        """获取合成图片（向后兼容单列模式）"""
        if self.page_level:
            return self.get_syn_img_page()
        
        # 始终使用黑底白字
        texts = self.get_random_text_from_txt()
        img = Image.new("RGB", self.img_size, (0, 0, 0))  # 黑色背景
        cond_img = Image.new("RGB", self.img_size, (255, 255, 255))  # 白色背景
        draw = ImageDraw.Draw(img)
        cond_draw = ImageDraw.Draw(cond_img)

        # 随机选择字体
        if len(self.ttf_fonts) == 0:
            return self.get_syn_img()  # 没有字体，重新获取
        font_index = random.randint(0, len(self.ttf_fonts) - 1)
        font_ttf = self.ttf_fonts[font_index]
        font_style = self.ttf_style[font_index]

        font_space = self.font_size * (1 - self.font_scale) // 2
        for i, text in enumerate(texts):
            font_position = (font_space, self.font_size * i + font_space)
            draw.text(font_position, text, font=font_ttf, fill=(255, 255, 255))
            cond_draw.text(font_position, text, font=self.font, fill=(0, 0, 0))

        # 转换为tensor
        img = torch.from_numpy(normalize_tensor(np.array(img))).permute(2, 0, 1)
        cond_img = torch.from_numpy(normalize_tensor(np.array(cond_img))).permute(2, 0, 1)

        # 构建prompt
        prompt = f'Synthetic calligraphy data, background: black, font: {convert_to_pinyin(font_style)}, '
        if font_style not in self.font_style_des:
            raise ValueError(f"Unsupported font style: {font_style}")
        prompt += self.font_style_des[font_style]

        return img, prompt, cond_img, texts

    def get_syn_img_page(self):
        """
        获取合成图片（Page-level模式）- 使用统一的 CharBox 标准
        
        核心思想：
        1. 生成随机的 CharBox 列表（位置、大小可变）
        2. 书法字大小随 CharBox 变化，标准字使用统一大小
        3. 额外输出 box_img 显示字符框位置和大小
        """
        # 先均匀采样字符数，再找匹配的画布，确保字数分布均衡
        num_chars = random.randint(3, self.required_chars)
        
        # 找能容纳 num_chars 个字符的画布尺寸
        matching_options = [opt for opt in self.page_options if opt[2] >= num_chars]
        if not matching_options:
            matching_options = [max(self.page_options, key=lambda x: x[2])]
            num_chars = matching_options[0][2]
        
        opt = random.choice(matching_options)
        canvas_w = opt[3] * self.cell_size
        canvas_h = opt[4] * self.cell_size
        canvas_size = (canvas_w, canvas_h)
        
        # 获取文本
        texts = self.get_random_text_from_txt(num_chars=num_chars)
        if texts is None or len(texts) < num_chars:
            return self.get_syn_img_page()
        texts = texts[:num_chars]
        
        # 随机选择布局模式
        layout_mode = random.choice(['grid', 'random', 'column', 'scatter'])
        
        # 使用统一的 CharBox 生成器
        char_boxes = generate_random_char_boxes(
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            num_chars=num_chars,
            base_cell_size=self.cell_size,
            scale_range=(0.85, 1.15),        # 整体 scale 变化
            aspect_ratio_range=(0.9, 1.1),   # 宽高比变化
            layout_mode=layout_mode,
            margin=5
        )
        
        # 如果生成的 box 数量不足，截断文本
        if len(char_boxes) < len(texts):
            texts = texts[:len(char_boxes)]
        
        # 创建图像
        img = Image.new("RGB", canvas_size, (0, 0, 0))  # 黑色背景
        cond_img = Image.new("RGB", canvas_size, (255, 255, 255))  # 白色背景
        # box_img 使用黑色背景，与书法图片背景一致，避免 attention 泄漏白色背景到生成图片中
        box_img = Image.new("RGB", canvas_size, (0, 0, 0))

        # 随机选择字体
        if len(self.ttf_fonts) == 0:
            return self.get_syn_img_page()  # 没有字体，重新获取
        font_index = random.randint(0, len(self.ttf_fonts) - 1)
        font_ttf = self.ttf_fonts[font_index]
        font_style = self.ttf_style[font_index]
        
        # 在每个 CharBox 内渲染字符
        draw = ImageDraw.Draw(img)
        cond_draw = ImageDraw.Draw(cond_img)
        box_draw = ImageDraw.Draw(box_img)
        
        # 统一的 condition 字体大小
        unified_cond_font_size = int(self.cell_size * self.font_scale)
        try:
            unified_cond_font = ImageFont.truetype(self.font_path, unified_cond_font_size)
        except:
            unified_cond_font = self.font
        
        for i, (char_box, text) in enumerate(zip(char_boxes, texts)):
            # 书法字大小随 CharBox 变化
            box_min_dim = min(char_box.width, char_box.height)
            calligraphy_font_size = int(box_min_dim * self.font_scale)
            
            # 创建对应大小的书法字体
            try:
                sized_ttf = ImageFont.truetype(
                    self.ttf_fonts[font_index].path if hasattr(self.ttf_fonts[font_index], 'path') 
                    else self.ttf_paths[font_index], 
                    calligraphy_font_size
                )
            except:
                sized_ttf = font_ttf
            
            # 计算书法字实际 bbox
            try:
                bbox = draw.textbbox((0, 0), text, font=sized_ttf)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except:
                text_w = calligraphy_font_size
                text_h = calligraphy_font_size
            
            # 书法字居中于 CharBox
            cal_x = char_box.x + (char_box.width - text_w) // 2
            cal_y = char_box.y + (char_box.height - text_h) // 2
            draw.text((cal_x, cal_y), text, font=sized_ttf, fill=(255, 255, 255))
            
            # --- 绘制 box_img：基于书法字实际 bbox（紧贴字符 + 小 padding），而非整个 CharBox ---
            # 这样与真实数据的标注框一致（真实数据 box 紧贴字符）
            padding = max(3, int(calligraphy_font_size * 0.05))
            box_x1 = max(0, cal_x - padding)
            box_y1 = max(0, cal_y - padding)
            box_x2 = min(canvas_w, cal_x + text_w + padding)
            box_y2 = min(canvas_h, cal_y + text_h + padding)
            
            random_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            box_draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], fill=random_color)
            
            # 标准字中心与书法字中心、box 几何中心重合
            center_x = cal_x + text_w // 2
            center_y = cal_y + text_h // 2
            try:
                cond_bbox = cond_draw.textbbox((0, 0), text, font=unified_cond_font)
                cond_text_w = cond_bbox[2] - cond_bbox[0]
                cond_text_h = cond_bbox[3] - cond_bbox[1]
            except:
                cond_text_w = unified_cond_font_size
                cond_text_h = unified_cond_font_size
            
            cond_x = center_x - cond_text_w // 2
            cond_y = center_y - cond_text_h // 2
            cond_draw.text((cond_x, cond_y), text, font=unified_cond_font, fill=(0, 0, 0))

        # 转换为tensor
        img = torch.from_numpy(normalize_tensor(np.array(img))).permute(2, 0, 1)
        cond_img = torch.from_numpy(normalize_tensor(np.array(cond_img))).permute(2, 0, 1)
        box_img = torch.from_numpy(normalize_tensor(np.array(box_img))).permute(2, 0, 1)

        # 构建prompt
        prompt = f'Synthetic calligraphy data, background: black, font: {convert_to_pinyin(font_style)}, '
        if font_style not in self.font_style_des:
            raise ValueError(f"Unsupported font style: {font_style}")
        prompt += self.font_style_des[font_style]

        # 计算实际的 cols 和 rows（近似值，用于兼容）
        actual_cols = max(1, canvas_w // self.cell_size)
        actual_rows = max(1, canvas_h // self.cell_size)
        layout_info = (actual_cols, actual_rows)
        
        return img, prompt, cond_img, box_img, texts, layout_info

    def __getitem__(self, idx):
        try:
            if self.page_level:
                # Page-level模式返回额外的 box_img 和 layout_info
                if random.random() < self.synth_prob:
                    img, prompt, cond_img, box_img, texts, layout_info = self.get_syn_img_page()
                else:
                    img, prompt, cond_img, box_img, texts, layout_info = self.get_real_img_page(idx)
                
                return img, prompt, cond_img, box_img, texts, layout_info
            else:
                # 原始单列模式
                if random.random() < self.synth_prob:
                    img, prompt, cond_img, texts = self.get_syn_img()
                else:
                    img, prompt, cond_img, texts = self.get_real_img(idx)

                return img, prompt, cond_img, texts

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def page_level_collate_fn(batch):
    """
    自定义collate函数用于page-level模式
    由于每个样本尺寸可能不同，batch_size必须为1
    """
    if len(batch) == 1:
        return batch[0] if isinstance(batch[0], tuple) else (batch[0],)
    
    # 如果batch_size > 1，需要特殊处理（目前不支持）
    raise ValueError("page_level mode only supports batch_size=1")


def loader(train_batch_size, num_workers, page_level=False, **args):
    dataset = CustomImageDataset(page_level=page_level, **args)
    
    if page_level:
        # page-level模式下使用自定义collate函数
        if train_batch_size != 1:
            print(f"Warning: page_level mode only supports batch_size=1, got {train_batch_size}")
            train_batch_size = 1
        return DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            num_workers=num_workers, 
            shuffle=True,
            collate_fn=page_level_collate_fn
        )
    else:
        return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)


if __name__ == '__main__':
    import sys
    
    def get_item(dataset, index, i, path, page_level=False):
        if page_level:
            clean_image, caption, condition_img, box_img, texts, layout_info = dataset[index]
            cols, rows = layout_info
            print(f"Layout: {cols}x{rows}")
            
            # 保存 box_img
            box_image = (box_img.permute(1, 2, 0) + 1).numpy() * 127.5
            box_image = Image.fromarray(box_image.astype(np.uint8))
            box_image.save(path+f'box_{i}.png')
        else:
            clean_image, caption, condition_img, texts = dataset[index]
        
        clean_image = (clean_image.permute(1, 2, 0) + 1).numpy() * 127.5
        condition_img = (condition_img.permute(1, 2, 0) + 1).numpy() * 127.5

        clean_image = Image.fromarray(clean_image.astype(np.uint8))
        condition_img = Image.fromarray(condition_img.astype(np.uint8))

        condition_img.save(path+f'cond_{i}.png')
        clean_image.save(path+f'clean_{i}.png')
        
        # Debug overlay: 在 grid_img 上叠加半透明 box 和 condition 文字
        if page_level:
            overlay = clean_image.copy().convert('RGBA')
            box_rgba = box_image.convert('RGBA')
            # box 白色区域变透明，彩色区域半透明叠加
            box_data = np.array(box_rgba)
            is_white = (box_data[:,:,0] > 200) & (box_data[:,:,1] > 200) & (box_data[:,:,2] > 200)
            box_data[is_white, 3] = 0       # 白色区域完全透明
            box_data[~is_white, 3] = 100    # 彩色区域半透明
            box_overlay = Image.fromarray(box_data)
            overlay = Image.alpha_composite(overlay, box_overlay)
            
            # 把 condition 文字也叠加（黑字区域）
            cond_rgba = condition_img.convert('RGBA')
            cond_data = np.array(cond_rgba)
            is_white_cond = (cond_data[:,:,0] > 200) & (cond_data[:,:,1] > 200) & (cond_data[:,:,2] > 200)
            cond_data[is_white_cond, 3] = 0
            cond_data[~is_white_cond] = [255, 0, 0, 180]  # 红色高亮 condition 文字
            cond_overlay = Image.fromarray(cond_data)
            overlay = Image.alpha_composite(overlay, cond_overlay)
            
            overlay.convert('RGB').save(path+f'overlay_{i}.png')
            
            # 横向拼接: clean | cond | box | overlay
            total_w = clean_image.width * 4
            compare = Image.new('RGB', (total_w, clean_image.height), (255, 255, 255))
            compare.paste(clean_image, (0, 0))
            compare.paste(condition_img, (clean_image.width, 0))
            compare.paste(box_image, (clean_image.width * 2, 0))
            compare.paste(overlay.convert('RGB'), (clean_image.width * 3, 0))
            compare.save(path+f'compare_{i}.png')

        print(caption)
        print(f"Texts: {texts}")
        return caption, texts

   
    # 默认：测试page-level模式
    print("\n" + "=" * 50)
    print("Testing page-level mode")
    print("=" * 50)
    dataset_page = CustomImageDataset(
        './word_dataset/optimized_data',
        img_size=64,
        required_chars=20,  # 最大字符数 (4x5=20)
        txt_dir='./libs/text_clips',
        ttf_dir='./libs/font',
        to_english=True,
        synth_prob=0.0,  # 只测试真实数据
        data_aug=False,   # 不二值化，直接看原图对齐
        author_descriptions="./word_dataset/calligraphy_styles_en.json",
        font_descriptions="./word_dataset/chirography.json",
        noise_threshold=20.0,
        page_level=True,
    )

    cond_page = {}
    path_page = "test_data/debug_page_level/"
    os.makedirs(path_page, exist_ok=True)

    for i in range(10):
        index = random.randint(0, len(dataset_page) - 1)
        caption, text = get_item(dataset_page, index, i, path_page, page_level=True)
        cond_page[i] = {'caption': caption, 'text': text}

    with open(path_page+"cond.json", "w", encoding="utf-8") as f:
        json.dump(cond_page, f, indent=4, ensure_ascii=False)
    
    print("\nDone! Check test_data/debug_page_level/")
    print("Each compare_*.png shows: Image | Condition | Box | Overlay")
    print("Overlay: semi-transparent box colors + red condition text on top of image")
    print("\nTo get 草书 samples, run: python dataset.py cao [num_samples]")
    