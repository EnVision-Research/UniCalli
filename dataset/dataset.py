import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
import csv

from utils import (
    convert_to_pinyin,
    binarize_auto_polarity,
    is_chinese_char,
    normalize_tensor
)

path = "/data/user/txu647/.cache/InternVL3-1B"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


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

def process_image_row(row, abs_path, required_chars=5, col_threshold=50, padding=5):
    """
    处理单个图片行，支持从labelme json文件读取标注
    row: 包含img_path, chirography, author等信息的数据行
    abs_path: 图片目录的绝对路径
    """
    img = Image.open(os.path.join(abs_path, row['img_path']))

    # 查找对应的labelme json文件
    json_path = find_labelme_json(row['img_path'], abs_path)
    if json_path is None:
        print(f"Warning: No labelme json found for {row['img_path']}")
        return None, None, None, None, None

    # 从labelme json文件加载标注
    locations = load_labelme_annotations(json_path)
    if not locations:
        print(f"Warning: No annotations found in {json_path}")
        return None, None, None, None, None

    # 分列逻辑（根据x坐标聚类）
    sorted_chars = sorted(locations, key=lambda x: x['p'][0])
    columns = []
    current_col = [sorted_chars[0]]

    # 动态分列（基于阈值）
    for char in sorted_chars[1:]:
        if char['p'][0] - current_col[-1]['p'][0] > col_threshold:
            columns.append(current_col)
            current_col = [char]
        else:
            current_col.append(char)
    columns.append(current_col)

    # 筛选有效列并检查字符间隔
    valid_columns = []
    for col in columns:
        if len(col) >= 2:  # 至少需要2个字符才能判断间隔
            col_sorted = sorted(col, key=lambda x: x['p'][1])

            # 计算平均字符间隔
            gaps = []
            for i in range(1, len(col_sorted)):
                prev_char = col_sorted[i-1]
                curr_char = col_sorted[i]
                gap = curr_char['p'][1] - prev_char['p'][3]  # 当前字符的y1 - 前一个字符的y2
                gaps.append(gap)

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
            else:
                avg_gap = 0

            # 检查连续字符间隔
            valid_sequences = []
            current_sequence = [col_sorted[0]]

            for i in range(1, len(col_sorted)):
                prev_char = col_sorted[i-1]
                curr_char = col_sorted[i]

                # 计算y坐标间隔（当前字符的y1 - 前一个字符的y2）
                gap = curr_char['p'][1] - prev_char['p'][3]

                # 如果间隔超过平均间隔的1.5倍，则开始新的序列
                if gap > avg_gap * 1.5:
                    if len(current_sequence) >= 2:  # 保存当前序列（如果长度>=2）
                        valid_sequences.append(current_sequence)
                    current_sequence = [curr_char]
                else:
                    current_sequence.append(curr_char)

            # 添加最后一个序列
            if len(current_sequence) >= 2:
                valid_sequences.append(current_sequence)

            # 添加所有有效序列到valid_columns
            valid_columns.extend(valid_sequences)

    # 过滤出长度足够的列
    sufficient_columns = [col for col in valid_columns if len(col) >= 2]
    if not sufficient_columns:
        return None, None, None, None, None

    # 找到最大可用字符数
    max_available_chars = max(len(col) for col in sufficient_columns)

    # 选择一个有效列
    selected_col = random.choice(sufficient_columns)
    selected_col = sorted(selected_col, key=lambda x: x['p'][1])

    # 根据列中字符数决定选择策略
    col_char_count = len(selected_col)

    if col_char_count > 5:
        num_to_select = 5
        start_idx = random.randint(0, col_char_count - num_to_select)
        selected_chars = selected_col[start_idx:start_idx+num_to_select]
    else:
        if max_available_chars < 5:
            max_selectable = min(max_available_chars, col_char_count)
            num_to_select = random.randint(2, max_selectable)
        else:
            # 否则可以选择当前列的所有字符
            num_to_select = col_char_count

        if num_to_select > col_char_count:
            num_to_select = col_char_count

        start_idx = random.randint(0, col_char_count - num_to_select) if col_char_count > num_to_select else 0
        selected_chars = selected_col[start_idx:start_idx+num_to_select]

    x_coords = [c['p'][0] for c in selected_chars] + [c['p'][2] for c in selected_chars]
    y_coords = [c['p'][1] for c in selected_chars] + [c['p'][3] for c in selected_chars]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    crop_box = (
        max(0, x_min - padding),
        max(0, y_min - padding - 10),
        min(img.width, x_max + padding),
        min(img.height, y_max + padding + 10)
    )

    cropped_img = img.crop(crop_box)
    new_locations = []
    texts = ''
    for char in selected_chars:
        # 使用调整后的坐标（如果有填充）或原始计算的坐标
        new_p = char.get('adjusted_p', [
            char['p'][0] - crop_box[0],
            char['p'][1] - crop_box[1],
            char['p'][2] - crop_box[0],
            char['p'][3] - crop_box[1]
        ])
        new_locations.append({'c': char['c'], 'p': new_p})
        texts += char['c']

    # 如果字符数不足required_chars个，用空字符填充（用于保持batch稳定）
    while len(new_locations) < required_chars:
        new_locations.append({'c': '', 'p': [0, 0, 0, 0]})

    # 实际选中的字符数（不包括填充的空字符）
    actual_char_count = len(selected_chars)

    return cropped_img, new_locations, row['chirography'], row['author'], actual_char_count

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False,
            author_descriptions=None, font_scale=0.8, font_size=None, required_chars=7,
            pred_box=False, to_english=True, txt_dir='./libs/text_clips', ttf_dir='./libs/font',
            synth_prob=0.5, data_aug=False, font_descriptions=None,
            noise_threshold=20.0, noise_csv_path=None):
        self.image_path = os.path.join(img_dir, 'images')

        # CSV文件现在只包含: img_path, chirography, author等基本信息
        # location信息从labelme json文件中读取
        csv_path = os.path.join(img_dir, 'data.csv')
        self.samples = pd.read_csv(csv_path)
        print(f"Loaded {len(self.samples)} samples from CSV")

        self.data_aug = data_aug
        self.to_english = to_english

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

        self.img_size = (img_size, img_size * required_chars)

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

    def __len__(self):
        return len(self.samples)
    
    def get_text_tokens(self, texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",   # pad 到固定长度
            max_length=self.required_chars
        )["input_ids"]

    def get_random_text_from_txt(self):
        max_chars = self.required_chars
        weights = self.weights
        assert max_chars == len(weights)

        file_path = random.choice(self.txt_files)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()

        chinese_chars = [ch for ch in raw if is_chinese_char(ch)]
        if len(chinese_chars) < max_chars:
            return None

        chosen_len = random.choices(range(1, max_chars + 1), weights=weights, k=1)[0]
        start_index = random.randint(0, len(chinese_chars) - chosen_len)
        return ''.join(chinese_chars[start_index: start_index + chosen_len])

    def get_condition(self, locations, origin_img_size, actual_char_count=None):
        background_color = (255, 255, 255)  # 白色背景
        text_color = (0, 0, 0)  # 黑色文字

        assert len(locations) == self.required_chars
        img = Image.new("RGB", self.img_size, background_color)
        box_img = Image.new("RGB", origin_img_size, background_color)
        draw = ImageDraw.Draw(img)
        draw_box = ImageDraw.Draw(box_img)

        # 如果没有提供actual_char_count，则计算非空字符数
        actual_char_count = actual_char_count or sum(1 for loc in locations if loc['c'])

        texts = ''
        font_space = self.font_size * (1 - self.font_scale) // 2

        for i, loc in enumerate(locations):
            text = loc['c']
            if i < actual_char_count and text:  # 实际字符
                texts += text
                font_position = (font_space, self.font_size * i + font_space)
                draw.text(font_position, text, font=self.font, fill=text_color)

                box = loc['p']
                draw_box.rectangle(
                    [(box[0], box[1]), (box[2], box[3])],
                    outline=text_color,
                    width=4
                )

        return img, box_img, texts

    def get_real_img(self, idx):
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

        if self.data_aug:
            img, polarity = binarize_auto_polarity(
                img,
                use_adaptive=False,
                target_fg_range=(0.03, 0.35),
                ksize=3, open_iters=1, close_iters=1, min_area=80
            )
            img = Image.fromarray(img, mode='L').convert('RGB')
        else:
            polarity = "white"  # 默认白色背景

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

    def get_syn_img(self):
        # 始终使用黑底白字
        texts = self.get_random_text_from_txt()
        img = Image.new("RGB", self.img_size, (0, 0, 0))  # 黑色背景
        cond_img = Image.new("RGB", self.img_size, (255, 255, 255))  # 白色背景
        draw = ImageDraw.Draw(img)
        cond_draw = ImageDraw.Draw(cond_img)

        # 随机选择字体
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

    def __getitem__(self, idx):
        try:
            if random.random() < self.synth_prob:  # get synthetic data
                img, prompt, cond_img, texts = self.get_syn_img()
            else:
                img, prompt, cond_img, texts = self.get_real_img(idx)

            text_token = self.get_text_tokens(texts).squeeze(0)
            return img, prompt, cond_img, text_token

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)


if __name__ == '__main__':
    def get_item(dataset, index, i, path):
        clean_image, caption, condition_img, texts_tokens = dataset[index]
        clean_image = (clean_image.permute(1, 2, 0) + 1).numpy() * 127.5
        condition_img = (condition_img.permute(1, 2, 0) + 1).numpy() * 127.5

        clean_image = Image.fromarray(clean_image.astype(np.uint8))
        condition_img = Image.fromarray(condition_img.astype(np.uint8))

        condition_img.save(path+f'cond_{i}.png')
        clean_image.save(path+f'clean_{i}.png')

        print(caption)
        text = tokenizer.decode(texts_tokens).split('<|')[0]
        return caption, text

    dataset = CustomImageDataset(
        './word_dataset/optimized_data',
        img_size=128,
        required_chars=7,
        txt_dir='./libs/text_clips',
        ttf_dir='./libs/font',
        to_english=True,
        synth_prob=0.5,
        data_aug=True,
        author_descriptions="./word_dataset/calligraphy_styles_en.json",
        font_descriptions="./word_dataset/chirography.json",
        noise_threshold=20.0,  # 噪声阈值
    )

    cond = {}
    path = "test_data/debug_chars7/"
    os.makedirs(path, exist_ok=True)

    # 测试10个样本
    for i in range(10):
        index = random.randint(0, len(dataset))
        caption, text = get_item(dataset, index, i, path)
        cond[i] = {'caption': caption, 'text': text}

    with open(path+"cond.json", "w", encoding="utf-8") as f:
        json.dump(cond, f, indent=4, ensure_ascii=False)
    