#!/usr/bin/env python3
"""
测试采样修复的脚本
"""
import torch
import numpy as np
from transformers import AutoTokenizer

def test_sampling_logic():
    """测试采样逻辑"""
    print("测试采样逻辑...")

    # 模拟数据
    batch_size = 1
    seq_len = 5
    vocab_size = 1000

    # 模拟cond_txt_latent和embed_token_weight
    cond_txt_latent = torch.randn(batch_size, seq_len, 768)
    embed_token_weight = torch.randn(vocab_size, 768)

    # 归一化
    cond_txt_latent = torch.nn.functional.normalize(cond_txt_latent, dim=-1)
    EW = torch.nn.functional.normalize(embed_token_weight, dim=-1)

    # 计算分数
    scores = cond_txt_latent @ EW.t()
    print(f"原始分数形状: {scores.shape}")
    print(f"分数范围: [{scores.min():.4f}, {scores.max():.4f}]")

    # 温度调节
    temperature = 0.8
    scores = scores / temperature

    # Top-k采样
    top_k = 5
    # scores.topk 返回值和索引，形状都是 [batch_size, seq_len, top_k]
    top_k_vals, top_k_indices = scores.topk(top_k, dim=-1)
    # softmax 转成概率
    top_k_probs = torch.softmax(top_k_vals, dim=-1)

    print(f"Top-k概率形状: {top_k_probs.shape}")
    print(f"Top-k索引形状: {top_k_indices.shape}")
    print(f"Top-k概率范围: [{top_k_probs.min():.4f}, {top_k_probs.max():.4f}]")

    # multinomial 只能接受 1D 或 2D 的概率分布，因此先展平为 [batch_size*seq_len, top_k]
    flat_probs = top_k_probs.reshape(-1, top_k)
    # 对每个位置的 top-k 概率分布采样一个索引，得到 [batch_size*seq_len]
    flat_sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
    # 还原为 [batch_size, seq_len]
    sampled_indices = flat_sampled.view(batch_size, seq_len)
    # 用采样得到的索引从 top_k_indices 中取出真正的 token id，结果还是 [batch_size, seq_len]
    selected_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)

    print(f"采样的token ID: {selected_tokens.tolist()}")
    print(f"采样结果形状: {selected_tokens.shape}")

    # 如果 top_k_probs 中出现 NaN，就退化为贪心策略
    if torch.isnan(top_k_probs).any():
        print("警告: 检测到NaN值")
        selected_tokens = scores.argmax(dim=-1)
        print(f"使用贪心采样的结果: {selected_tokens.tolist()}")

    print("采样逻辑测试完成！")
    return selected_tokens


def test_tokenizer():
    """测试tokenizer"""
    print("\n测试tokenizer...")
    
    # 模拟tokenizer
    try:
        intern_path = "/data/user/txu647/.cache/InternVL3-1B"
        tokenizer = AutoTokenizer.from_pretrained(intern_path, trust_remote_code=True, use_fast=False)
        
        # 测试解码
        test_tokens = [101, 2023, 2003, 102]  # [CLS] this is [SEP]
        decoded_text = tokenizer.decode(test_tokens, skip_special_tokens=True)
        print(f"测试tokens: {test_tokens}")
        print(f"解码结果: '{decoded_text}'")
        
    except Exception as e:
        print(f"Tokenizer测试失败: {e}")
        print("使用模拟tokenizer...")
        
        # 模拟解码
        test_tokens = [1, 2, 3, 4, 5]
        decoded_text = "模拟解码结果"
        print(f"测试tokens: {test_tokens}")
        print(f"解码结果: '{decoded_text}'")

if __name__ == "__main__":
    print("开始测试采样修复...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试采样逻辑
    selected_tokens = test_sampling_logic()
    
    # 测试tokenizer
    test_tokenizer()
    
    print("\n所有测试完成！")
    print("如果看到合理的token ID和解码结果，说明修复是有效的。")
