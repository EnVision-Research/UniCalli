import argparse
import torch
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser(description="将Pytorch .bin文件转换成safetensor格式")
    parser.add_argument("input", type=str, help="输入.bin检查点文件路径")
    parser.add_argument("output", type=str, help="输出.safetensors文件路径")
    args = parser.parse_args()

    # 加载.bin检查点文件
    ckpt = torch.load(args.input, map_location="cpu")

    # 保存为safetensor文件
    save_file(ckpt, args.output)
    print(f"成功将 {args.input} 转换为 {args.output}")

if __name__ == "__main__":
    main()