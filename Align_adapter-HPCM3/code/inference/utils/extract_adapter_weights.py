"""
从保存的 ckpt 中提取 Adapter 权重并保存为 .pt 文件

功能：
从 PyTorch Lightning 训练保存的完整 Checkpoint (.ckpt) 中提取仅 Adapter 部分的权重，
并保存为推理所需的格式 (.pt)。HPCM2 的 Adapter 结构与 Align_adapter-HPCM2/model/adapter.py 一致，
权重键名以 "adapter." 开头。

使用方法：
python Align_adapter-HPCM2/inference/utils/extract_adapter_weights.py \
    --input_ckpt /path/to/checkpoint.ckpt \
    --output_path /path/to/adapter_weights.pt

或在代码中作为模块导入使用。
"""

import argparse
import sys
from pathlib import Path

import torch


def extract_adapter_weights(ckpt_path: str, output_path: str, verbose: bool = True) -> None:
    """
    从 PL Checkpoint 中提取 Adapter 权重（HPCM2：pre_neck + neck_siglip + neck_dino）。

    Args:
        ckpt_path: 输入的 .ckpt 文件路径
        output_path: 输出的 .pt 权重文件路径
        verbose: 是否打印详细信息
    """
    ckpt_path = Path(ckpt_path)
    output_path = Path(output_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 文件未找到: {ckpt_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"正在加载 Checkpoint: {ckpt_path} ...")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" not in checkpoint:
        state_dict = checkpoint
    else:
        state_dict = checkpoint["state_dict"]

    prefix = "adapter."
    adapter_weights = {}
    count = 0
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            adapter_weights[new_key] = value
            count += 1

    if count == 0:
        print("警告: 未找到以 'adapter.' 开头的权重，尝试模糊匹配 pre_neck / neck ...")
        for key, value in state_dict.items():
            if "pre_neck" in key or "neck_siglip" in key or "neck_dino" in key:
                adapter_weights[key] = value
                count += 1

    if count == 0:
        raise ValueError("在 Checkpoint 中未找到 Adapter 相关权重，请检查 ckpt 结构。")

    if verbose:
        print(f"提取了 {count} 个 Adapter 参数张量。")
        print(f"正在保存到: {output_path} ...")

    torch.save(adapter_weights, output_path)

    if verbose:
        print("完成。")


if __name__ == "__main__":
    DEFAULT_INPUT_CKPT = None  # 例如: "/path/to/Align_adapter_HPCM_ckpts/checkpoints/last.ckpt"
    DEFAULT_OUTPUT_PATH = None  # 例如: "Align_adapter-HPCM2/inference/utils/adapter_weights.pt"

    parser = argparse.ArgumentParser(description="从 HPCM2 训练 ckpt 中提取 Adapter 权重")
    parser.add_argument("--input_ckpt", type=str, default=None, help="输入的 .ckpt 文件路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出的 .pt 文件路径")

    args = parser.parse_args()

    input_ckpt = args.input_ckpt or DEFAULT_INPUT_CKPT
    output_path = args.output_path or DEFAULT_OUTPUT_PATH

    if input_ckpt is None or output_path is None:
        print("错误: 必须提供 input_ckpt 和 output_path")
        print("方式1: 在代码中设置 DEFAULT_INPUT_CKPT 和 DEFAULT_OUTPUT_PATH")
        print("方式2: 通过命令行 --input_ckpt 和 --output_path")
        sys.exit(1)

    extract_adapter_weights(input_ckpt, output_path)
