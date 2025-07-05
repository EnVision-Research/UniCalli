from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14")

# url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)
# Squirtle, Bulbasaur, Charmander, Pikachu in English
texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

# compute image feature
# inputs = processor(images=image, return_tensors="pt")
# image_features = model.get_image_features(**inputs)
# image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# compute text features
inputs = processor(text=texts, padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
print(text_features.shape)
# compute image-text similarity scores
# inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # probs: [[1.1419e-02, 1.0478e-02, 5.2018e-04, 9.7758e-01]]
