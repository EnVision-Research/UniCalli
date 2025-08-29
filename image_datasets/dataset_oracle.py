import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from PIL import Image, ImageDraw, ImageFont
import ast
from pypinyin import lazy_pinyin
import json
import cv2
from transformers import AutoModel, AutoTokenizer

path = "/data/user/txu647/.cache/InternVL3-1B"

embed_tokens = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float32,   # CPU 一般用 float32，更安全
    device_map="cpu",            # 强制全部放在 CPU
    trust_remote_code=True
).language_model.model.embed_tokens.eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


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
    use_adaptive: bool = False,        # 光照不均明显→True（自适应阈值）
    target_fg_range=(0.03, 0.35),      # 文字覆盖率期望范围（可按数据分布调）
    ksize=3, open_iters=1, close_iters=1,
    min_area=64
):
    # 1) 读灰度 + 预处理（中值滤波 + CLAHE 提升对比度）
    # g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    g = np.array(pil_img.convert("L"), dtype=np.uint8)
    g = cv2.medianBlur(g, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_eq = clahe.apply(g)

    # 2) 两个候选：A=“黑字为前景”（白底黑字场景），B=“白字为前景”（黑底白字场景）
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

    # 3) 去噪清理（开闭运算 + 小面积过滤）
    cand_A = _morph_clean(cand_A, ksize, open_iters, close_iters, min_area)
    cand_B = _morph_clean(cand_B, ksize, open_iters, close_iters, min_area)

    # 4) 评分：谁更像“笔画”
    edges = cv2.Canny(g_eq, 50, 150)
    score_A = _score_mask(cand_A, edges, target_fg_range)
    score_B = _score_mask(cand_B, edges, target_fg_range)

    if score_A >= score_B:
        return 255 - cand_A, "white"   # 背景为白
    else:
        return cand_B, "black"   # 背景为黑

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

    # 随机选择有效列
    valid_columns = [col for col in columns if len(col) >= required_chars]
    if not valid_columns:
        return None, None, None, None

    selected_col = random.choice(valid_columns)
    selected_index = valid_columns.index(selected_col)
    selected_col = sorted(selected_col, key=lambda x: x['p'][1])
    
    # 随机选择连续字符
    start_idx = random.randint(0, len(selected_col) - required_chars)
    selected_chars = selected_col[start_idx:start_idx+required_chars]
    
    # de-normalize
    for i in range(len(selected_chars)):
        selected_chars[i]['p'][0] = int(selected_chars[i]['p'][0] * w / 1000)
        selected_chars[i]['p'][1] = int(selected_chars[i]['p'][1] * h / 1000)
        selected_chars[i]['p'][2] = int(selected_chars[i]['p'][2] * w / 1000)
        selected_chars[i]['p'][3] = int(selected_chars[i]['p'][3] * h / 1000)

    # 计算裁剪区域
    x_coords = [c['p'][0] for c in selected_chars] + [c['p'][2] for c in selected_chars]
    y_coords = [c['p'][1] for c in selected_chars] + [c['p'][3] for c in selected_chars]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 扩展边界
    crop_box = (
        max(0, x_min - padding),
        max(0, y_min - padding),
        min(img.width, x_max + padding),
        min(img.height, y_max + padding)
    )

    # 裁剪图片
    cropped_img = img.crop(crop_box)
    
    # 调整坐标
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

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False, 
            author_descriptions=None, font_scale=0.8, font_size=None, required_chars=5, 
            pred_box=False, to_english=True, txt_dir='./libs/text_clips', ttf_dir='./libs/font', 
            synth_prob=0.5, data_aug=False, font_descriptions=None):        
        self.image_path = os.path.join(img_dir, 'images')
        df1 = pd.read_csv(os.path.join(img_dir, 'data1.csv'))
        df2 = pd.read_csv(os.path.join(img_dir, 'data2.csv'))
        self.samples = pd.concat([df1, df2], ignore_index=True)
        self.data_aug = data_aug
        self.to_english = to_english

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
        # self.ttf_style = [convert_to_pinyin(ttf_path.split('/')[-1].split('.')[0].split('_')[1]) for ttf_path in ttf_files]
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
        def is_chinese_char(ch: str) -> bool:
            """判断单字符是否为中文汉字（覆盖常见 CJK 区块）。"""
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
            for start, end in ranges:
                if start <= cp <= end:
                    return True
            return False
        
        max_chars = self.required_chars
        weights = self.weights
        assert max_chars == len(weights)

        # 随机选文件
        file_path = random.choice(self.txt_files)

        # 以容错方式读取文件（避免因编码问题崩溃）
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()

        # 只保留中文汉字
        chinese_chars = [ch for ch in raw if is_chinese_char(ch)]
        if len(chinese_chars) < max_chars:
            return None

        # 按权重随机选取长度（1..max_chars）
        chosen_len = random.choices(range(1, max_chars + 1), weights=weights, k=1)[0]

        # 随机选择起始位置并返回连续的中文片段
        start_index = random.randint(0, len(chinese_chars) - chosen_len)
        segment = ''.join(chinese_chars[start_index: start_index + chosen_len])

        return segment

    def get_condition(self, locations, origin_img_size):
        background_color = (255, 255, 255)  # 白色背景
        text_color = (0, 0, 0)  # 黑色文字

        assert len(locations) == self.required_chars
        img = Image.new("RGB", self.img_size, background_color)
        box_img = Image.new("RGB", origin_img_size, background_color)
        draw = ImageDraw.Draw(img)
        draw_box = ImageDraw.Draw(box_img)

        texts = ''
        for i, loc in enumerate(locations):
            text = loc['c']
            texts += text

            font_space = self.font_size * (1 - self.font_scale) // 2
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
        img, new_locs, chirography, author = process_image_row(
            sample_row, 
            self.image_path, 
            required_chars=self.required_chars, 
            col_threshold=50,
            padding=50,
        )

        if new_locs is None:
            # print(f"No valid characters found in the image: {img_path}.")
            self.bad_indices.append(img_path)
            return self.get_real_img(random.randint(0, len(self.samples) - 1))
        
        if self.data_aug:
            img, polarity = binarize_auto_polarity(
                img,
                use_adaptive=False,      # 黑白底差异大、光照均匀→False；光照不匀→True
                target_fg_range=(0.03, 0.35),
                ksize=3, open_iters=1, close_iters=1, min_area=80
            )
            img = Image.fromarray(img, mode='L')
            img = img.convert('RGB')  # 确保是RGB模式

        prompt = f"Traditional Chinese calligraphy works, background: {polarity}, font: {convert_to_pinyin(chirography)},"
        if chirography in self.font_style_des:
            prompt += ' ' + self.font_style_des[chirography]

        if author in self.author_style:
            prompt += f' author: {self.author_style[author]}'
        else:
            prompt += f' author: {convert_to_pinyin(author)}.'
        
        cond_img, box_img, texts = self.get_condition(new_locs, img.size)
        # cond_img.save('debug_cond.png'); img.save('debug.png')
        
        img = img.resize(self.img_size, Image.LANCZOS)
        box_img = box_img.resize(self.img_size, Image.LANCZOS)
        assert img.size == cond_img.size == self.img_size == box_img.size, "img and cond_img size should be equal"

        # grayscale img detect and convert
        if img.mode == 'L':
            img = img.convert('RGB')

        img = torch.from_numpy((np.array(img) / 127.5) - 1)
        cond_img = torch.from_numpy((np.array(cond_img) / 127.5) - 1)
        box_img = torch.from_numpy((np.array(box_img) / 127.5) - 1)

        assert img.ndim == 3 and cond_img.ndim == 3, "img and cond_img should be 3D tensors"
        img = img.permute(2, 0, 1)
        cond_img = cond_img.permute(2, 0, 1)
        box_img = box_img.permute(2, 0, 1)

        if self.pred_box:
            img = torch.cat((img, box_img), dim=2)
            cond_img = torch.cat((cond_img, torch.zeros_like(cond_img)), dim=2)

        return img, prompt, cond_img, texts

    def get_syn_img(self):
        if random.random() < 0.5:
            bg_color = "white"
            background_color = (255, 255, 255)  # 白色背景
            text_color = (0, 0, 0)  # 黑色文字
        else:
            bg_color = "black"
            background_color = (0, 0, 0)
            text_color = (255, 255, 255)

        texts = self.get_random_text_from_txt()
        img = Image.new("RGB", self.img_size, background_color)
        cond_img = Image.new("RGB", self.img_size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        cond_draw = ImageDraw.Draw(cond_img)

        font_index = random.randint(0, len(self.ttf_fonts) - 1)
        font_ttf = self.ttf_fonts[font_index]
        font_style = self.ttf_style[font_index]

        for i, text in enumerate(texts):
            font_space = self.font_size * (1 - self.font_scale) // 2
            font_position = (font_space, self.font_size * i + font_space)
            draw.text(font_position, text, font=font_ttf, fill=text_color)
            cond_draw.text(font_position, text, font=self.font, fill=(0, 0, 0))
        
        img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1)
        cond_img = torch.from_numpy((np.array(cond_img) / 127.5) - 1).permute(2, 0, 1)
        prompt = f'Synthetic calligraphy data, background: {bg_color}, font: {convert_to_pinyin(font_style)}, '

        if font_style in self.font_style_des:
            prompt += self.font_style_des[font_style]
        else:
            raise ValueError(f"Unsupported font style: {font_style}")

        return img, prompt, cond_img, texts

    def __getitem__(self, idx):
        try:
            if random.random() < self.synth_prob:  # get synthetic data
                img, prompt, cond_img, texts = self.get_syn_img()
            else:
                img, prompt, cond_img, texts = self.get_real_img(idx)

            text_tokens = self.get_text_tokens(texts)
            text_latent = embed_tokens(text_tokens)
            return img, prompt, cond_img, text_tokens, text_latent
        
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)


if __name__ == '__main__':
    def find_error_indices(dataset, save_path='error_indices.json'):
        error_indices = []
        
        for idx in range(len(dataset)):
            if idx % 100 == 0:
                print(idx)
            try:
                _ = dataset[idx]
                img_path = dataset.samples.iloc[idx]['img_path']
            except Exception as e:
                print(f"Error at img_path {img_path}: {str(e)}")
                if img_path not in error_indices:
                    error_indices.append(img_path)
        
        with open(save_path, 'w') as f:
            json.dump(error_indices, f)

    def get_item(dataset, index):
        image, caption, condition_img, texts_tokens, texts_latents = dataset[index]
        image = (image.permute(1, 2, 0) + 1).numpy() * 127.5
        condition_img = (condition_img.permute(1, 2, 0) + 1).numpy() * 127.5
        image = Image.fromarray(image.astype(np.uint8))
        condition_img = Image.fromarray(condition_img.astype(np.uint8))
        # image.save(f'ckpts/img_{index}.png'); condition_img.save(f'ckpts/cond_{index}.png')
        # image.save(f'test_data/debug/img_{index}.png'); condition_img.save(f'test_data/debug/cond_{index}.png')
        print(caption)
        print(texts_tokens.shape, texts_latents.shape)
        
    dataset = CustomImageDataset(
        './word_dataset/finalpage',
        img_size=128,
        required_chars=5,
        txt_dir='./libs/text_clips',
        ttf_dir='./libs/font',
        to_english=True,
        synth_prob=0.5,
        data_aug=True,
        author_descriptions="./word_dataset/calligraphy_styles_en.json",
        font_descriptions="./word_dataset/chirography.json",
        )

    # find_error_indices(dataset, 'error_indices_chars5.json')
    # breakpoint()
    
    # for i in range(len(dataset)):
    for i in range(8):
        # index = random.randint(0, len(dataset))
        get_item(dataset, i)
        # breakpoint()
    
