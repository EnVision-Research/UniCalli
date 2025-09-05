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
from transformers import AutoTokenizer
from pathlib import Path

PROMPT = 'Ancient Egyptian hieroglyphs are a pictographic writing system that blends phonetic signs, logograms, and determinatives to record language on stone, papyrus, and monuments for over three millennia.'
path = "/data/user/txu647/.cache/InternVL3-1B"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def convert_to_pinyin(text, with_tone=False):
    return ' '.join([item[0] if isinstance(item, list) else item for item in lazy_pinyin(text)])

def resize_and_pad(img: Image.Image, N: int, fill=(255, 255, 255)):
    w, h = img.size
    s = N / max(w, h)
    new_w, new_h = max(1, int(w * s)), max(1, int(h * s))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    mode = 'RGBA' if 'A' in img.getbands() else 'RGB'
    bg = Image.new(mode, (N, N), (0, 0, 0, 0) if mode == 'RGBA' else fill)
    if img.mode != mode:
        img = img.convert(mode)
    bg.paste(img, ((N - new_w) // 2, (N - new_h) // 2), img if mode == 'RGBA' else None)
    return bg

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=128, font_scale=0.8, use_undeciphered=False, font_size=None):        
        img_dir = Path(img_dir)
        self.samples = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg"]
        self.img_size = (img_size, img_size)
        print('Dataset length:', len(self.samples))
        print('Use undeciphered data:', use_undeciphered)
        
        json_path_1 = os.path.join(img_dir, 'mapping_ch.json')
        with open(json_path_1, 'r', encoding='utf-8') as f:        
            self.id2chinese = json.load(f)

        self.font_path = "./FangZhengKaiTiFanTi-1.ttf"  
        self.font_scale = font_scale
        self.font_size = img_size if font_size is None else font_size
        self.font = ImageFont.truetype(self.font_path, int(font_scale * self.font_size))

        self.bad_indices = []

    def __len__(self):
        return len(self.samples)
    
    def render_char(self, ch, bg="white", fg="black"):
        img = Image.new("RGB", self.img_size, bg)
        pad = self.font_size * (1 - self.font_scale) // 2
        ImageDraw.Draw(img).text((pad, pad), ch, font=self.font, fill=fg)
        return img

    def get_text_tokens(self, texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",   # pad 到固定长度
            max_length=5
        )["input_ids"]

    def get_real_img(self, idx):        
        img_path = self.samples[idx]
        if img_path in self.bad_indices:
            return self.get_real_img(random.randint(0, len(self.samples) - 1))
        
        # /data/user/txu647/code/flux-calligraphy/word_dataset/hieroglyphs_dataset/001/100 (1).jpg
        text_id = img_path.parent.name
        text = self.id2chinese[text_id]
        
        img = Image.open(img_path).convert('RGB')
        assert self.img_size[0] == self.img_size[1]
        img = resize_and_pad(img, self.img_size[0])
        img = torch.from_numpy((np.array(img) / 127.5) - 1)

        cond_img = self.render_char(text)
        assert cond_img is not None
        cond_img = torch.from_numpy((np.array(cond_img) / 127.5) - 1)

        assert img.ndim == 3 and cond_img.ndim == 3, "img and cond_img should be 3D tensors"
        img = img.permute(2, 0, 1)
        cond_img = cond_img.permute(2, 0, 1)
        prompt = PROMPT

        return img, prompt, cond_img, text

    def __getitem__(self, idx):
        try:
            img, prompt, cond_img, text = self.get_real_img(idx)
            text_token = self.get_text_tokens(text).squeeze(0)
            return img, prompt, cond_img, text_token
        
        except Exception as e:
            print(e)
            self.bad_indices.append(idx)
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def loader_eg(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)


if __name__ == '__main__':
    def get_item(dataset, i):
        index = random.randint(0, len(dataset))
        # index = i
        image, caption, condition_img, texts_tokens = dataset[index]
        image = (image.permute(1, 2, 0) + 1).numpy() * 127.5
        condition_img = (condition_img.permute(1, 2, 0) + 1).numpy() * 127.5
        image = Image.fromarray(image.astype(np.uint8))
        condition_img = Image.fromarray(condition_img.astype(np.uint8))
        image.save(f'test_data/eg/img_{i}.png'); condition_img.save(f'test_data/eg/cond_{i}.png')
        print(caption)
        print(texts_tokens)
        return tokenizer.decode(texts_tokens)[0]
        
    dataset = CustomImageDataset(
        './word_dataset/hieroglyphs_dataset',
        img_size=128,
        )

    
    # for i in range(len(dataset)):
    cond_txt = []
    for i in range(8):
        cond_txt.append(get_item(dataset, i))
        # breakpoint()
    print(cond_txt)
