import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from PIL import Image, ImageDraw, ImageFont
import json
from transformers import AutoTokenizer
from image_datasets.utils import process_image_row, binarize_auto_polarity, convert_to_pinyin


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False, 
            author_descriptions=None, font_scale=0.8, font_size=None, required_chars=5, 
            pred_box=False, to_english=True, txt_dir='./libs/text_clips', ttf_dir='./libs/font', 
            synth_prob=0.5, data_aug=False, font_descriptions=None, tokenizer_path=None):        
        self.image_path = os.path.join(img_dir, 'images')
        df1 = pd.read_csv(os.path.join(img_dir, 'data1.csv'))
        df2 = pd.read_csv(os.path.join(img_dir, 'data2.csv'))
        self.samples = pd.concat([df1, df2], ignore_index=True)
        self.data_aug = data_aug
        self.to_english = to_english

        assert tokenizer_path is not None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)

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

        self.synth_prob = synth_prob
        self.weights=[float(f'0.{i+1}')-0.01 for i in range(required_chars)]

        self.txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]
        if not self.txt_files:
            raise ValueError("Not find any .txt files, please check the path")

        ttf_files = [os.path.join(ttf_dir, f) for f in os.listdir(ttf_dir) if f.lower().endswith('.ttf')]
        if not ttf_files:
            raise ValueError("Not find any .ttf files, please check the path")
        self.ttf_fonts = [ImageFont.truetype(ttf_path, int(font_scale * self.font_size)) for ttf_path in ttf_files]
        self.ttf_style = [ttf_path.split('/')[-1].split('.')[0].split('_')[1] for ttf_path in ttf_files]

    def __len__(self):
        return len(self.samples)
    
    def get_text_tokens(self, texts):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length", 
            max_length=self.required_chars
        )["input_ids"]

    def get_random_text_from_txt(self):
        def is_chinese_char(ch: str) -> bool:
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

        file_path = random.choice(self.txt_files)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()

        chinese_chars = [ch for ch in raw if is_chinese_char(ch)]
        if len(chinese_chars) < max_chars:
            return None

        chosen_len = random.choices(range(1, max_chars + 1), weights=weights, k=1)[0]
        start_index = random.randint(0, len(chinese_chars) - chosen_len)
        segment = ''.join(chinese_chars[start_index: start_index + chosen_len])

        return segment

    def get_condition(self, locations, origin_img_size):
        background_color = (255, 255, 255)
        text_color = (0, 0, 0) 

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
        # if "王羲之" not in author:
        #     if chirography != "草":
        #         return self.get_real_img(random.randint(0, len(self.samples) - 1))
        
        if new_locs is None or chirography == '隶':
            # print(f"No valid characters found in the image: {img_path}.")
            self.bad_indices.append(img_path)
            return self.get_real_img(random.randint(0, len(self.samples) - 1))
        
        if self.data_aug:
            img, polarity = binarize_auto_polarity(
                img,
                use_adaptive=False, 
                target_fg_range=(0.03, 0.35),
                ksize=3, open_iters=1, close_iters=1, min_area=80
            )
            img = Image.fromarray(img, mode='L')
            img = img.convert('RGB')

        prompt = f"Traditional Chinese calligraphy works, background: {polarity}, font: {convert_to_pinyin(chirography)},"
        if chirography in self.font_style_des:
            prompt += ' ' + self.font_style_des[chirography]

        if author in self.author_style:
            prompt += f' author: {self.author_style[author]}'
        else:
            prompt += f' author: {convert_to_pinyin(author)}.'
        
        cond_img, box_img, texts = self.get_condition(new_locs, img.size)
        
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
            background_color = (255, 255, 255)
            text_color = (0, 0, 0) 
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
        image, caption, condition_img, texts_tokens = dataset[index]
        image = (image.permute(1, 2, 0) + 1).numpy() * 127.5
        condition_img = (condition_img.permute(1, 2, 0) + 1).numpy() * 127.5
        image = Image.fromarray(image.astype(np.uint8))
        condition_img = Image.fromarray(condition_img.astype(np.uint8))

        condition_img.save(path+f'cond_{i}.png')
        image.save(path+f'img_{i}.png')
        print(caption)
        text = dataset.tokenizer.decode(texts_tokens).split('<|')[0]
        return caption, text
        
    dataset = CustomImageDataset(
        './word_dataset/finalpage',
        img_size=128,
        required_chars=5,
        txt_dir='./libs/text_clips',
        ttf_dir='./libs/font',
        to_english=True,
        synth_prob=0,
        data_aug=True,
        author_descriptions="./word_dataset/calligraphy_styles_en.json",
        font_descriptions="./word_dataset/chirography.json",
        tokenizer_path=""
        )

    cond = {}
    path = "test_data/debug"
    os.mkdir(path)
    
    for i in range(4):
        index = random.randint(0, len(dataset))
        caption, text = get_item(dataset, index, i, path)
        cond[i] = {'caption': caption, 'text': text}

    with open(path+"cond.json", "w", encoding="utf-8") as f:
        json.dump(cond, f, indent=4, ensure_ascii=False)
    
