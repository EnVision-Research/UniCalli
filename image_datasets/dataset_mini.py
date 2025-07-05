import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from PIL import Image, ImageDraw, ImageFont

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


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False):
        caption_file = os.path.join(img_dir, 'data.txt')

        self.samples = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel_path, ann_str = line.split('\t')

                # for test mode:
                folder_num = eval(img_rel_path.split('/')[0])
                if folder_num > 15 or folder_num < 6:
                    continue

                full_img_path = os.path.join(img_dir, img_rel_path)
                annotations = json.loads(ann_str)
                caption = annotations[0]['transcription']
                self.samples.append({"img": full_img_path, "caption": caption})

        print(f"Loaded {len(self.samples)} samples from {caption_file}")
        self.img_size = img_size
        # self.random_ratio = random_ratio

    def __len__(self):
        return len(self.samples)

    def get_condition(self, text):
        background_color = (255, 255, 255)  # 白色背景
        text_color = (0, 0, 0)  # 黑色文字

        img = Image.new("RGB", (self.img_size, self.img_size), background_color)
        draw = ImageDraw.Draw(img)

        font_path = "/hpc2hdd/home/txu647/code/x-flux/FangZhengKaiTiFanTi-1.ttf"  
        font_size = int(0.8 * self.img_size)
        font = ImageFont.truetype(font_path, font_size)

        position = ((self.img_size - font_size) // 2, (self.img_size - font_size) // 2)

        draw.text(position, text, font=font, fill=text_color)

        return img

    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx]["img"]).convert('RGB')
            # if self.random_ratio:
            #     ratio = random.choice(["16:9", "default", "1:1", "4:3"])
            #     if ratio != "default":
            #         img = crop_to_aspect_ratio(img, ratio)
            # img = image_resize(img, self.img_size)
            # w, h = img.size
            # new_w = (w // 32) * 32
            # new_h = (h // 32) * 32
            img = img.resize((self.img_size, self.img_size))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            # json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            # if self.caption_type == "json":
            #     prompt = json.load(open(json_path))['caption']
            # else:
            #     prompt = open(json_path).read()
            prompt = self.samples[idx]["caption"]
            cond_img = self.get_condition(prompt)
            cond_img = torch.from_numpy((np.array(cond_img) / 127.5) - 1)
            cond_img = cond_img.permute(2, 0, 1)

            return img, prompt, cond_img
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)


if __name__ == '__main__':
    dataset = CustomImageDataset(
        '/hpc2hdd/home/txu647/code/word_dataset/word_mini',
        img_size=256,
        )
    image, caption, condition_img = dataset[2]
    image = (image.permute(1, 2, 0) + 1).numpy() * 127.5
    condition_img = (condition_img.permute(1, 2, 0) + 1).numpy() * 127.5
    image = Image.fromarray(image.astype(np.uint8))
    condition_img = Image.fromarray(condition_img.astype(np.uint8))
    breakpoint()
    image.save('men.png')
    condition_img.save('men_cond.png')
