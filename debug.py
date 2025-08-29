from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import time

texts = '仁义礼'  # assert None token: 1294
path = "/data/user/txu647/.cache/InternVL3-1B"
embed_tokens = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float32,   # CPU 一般用 float32，更安全
    device_map="cpu",            # 强制全部放在 CPU
    trust_remote_code=True
).language_model.model.embed_tokens.eval()
embed_tokens.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def get_text_tokens(texts):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",   # pad 到固定长度
        truncation=True,        # 太长就截断
        max_length=5            # 固定长度 5
    )["input_ids"]

breakpoint()
latents = embed_tokens(get_text_tokens(texts))

# text_decode = tokenizer.decode(tokens)
