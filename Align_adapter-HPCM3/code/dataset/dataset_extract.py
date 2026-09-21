"""
从 Align_adapter/dataset 下四个 libero 数据集中读取图片，
用 HPCM 编码器提取量化特征 y_hat，按相同目录结构保存到 Align_adapter_HPCM/dataset。

数据集：libero_10_no_noops, libero_goal_no_noops, libero_object_no_noops, libero_spatial_no_noops
图片：*_image.png, *_wrist_image.png 等 .png
保存：同名 .pt 文件，内容为 torch.Tensor y_hat (1, 320, H', W')，float32
"""

import argparse
import os
import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# 保证从 Align_adapter_HPCM 根目录可 import dataset 包
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.hpcm_encoder import HPCMEncoder

# 四个 libero 数据集名称（与 train_config.yaml 一致）
LIBERO_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def main():
    parser = argparse.ArgumentParser(description="Extract HPCM y_hat for libero images")
    parser.add_argument(
        "--source_dir",
        type=str,
        default=os.path.join(_ROOT, "..", "Align_adapter", "dataset"),
        help="源数据集根目录（包含四个 libero_* 子目录）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/cjs/shared/linux/Align_adapter/HPCM_dataset",
        help="输出根目录（与 source 同结构，保存 .pt）",
    )
    parser.add_argument(
        "--hpcm_root",
        type=str,
        default=os.path.join(_ROOT, "..", "HPCM"),
        help="HPCM 项目根目录",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="HPCM 权重路径，默认 hpcm_root/ckpt/0.0018.pth.tar",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="批大小（1 则逐张）")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="若 .pt 已存在则跳过，默认开启，中断后重跑会从未完成的继续",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="关闭跳过已存在，全部重新提取",
    )
    parser.add_argument("--dataset", type=str, default=None, help="只处理指定数据集名，默认全部四个")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)
    hpcm_root = os.path.abspath(args.hpcm_root)
    ckpt = args.ckpt or os.path.join(hpcm_root, "ckpt", "0.0018.pth.tar")

    if not os.path.isdir(source_dir):
        print(f"源目录不存在: {source_dir}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    names = LIBERO_DATASET_NAMES if args.dataset is None else [args.dataset]
    if args.dataset is not None and args.dataset not in LIBERO_DATASET_NAMES:
        print(f"未知 dataset: {args.dataset}, 可选: {LIBERO_DATASET_NAMES}")
        sys.exit(1)

    # 收集指定数据集下的所有 .png
    all_pairs = []
    for name in names:
        d = os.path.join(source_dir, name)
        if not os.path.isdir(d):
            print(f"跳过（不存在）: {d}")
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".png"):
                    abs_path = os.path.join(root, f)
                    rel = os.path.relpath(abs_path, source_dir)
                    all_pairs.append((abs_path, rel))

    print(f"共 {len(all_pairs)} 张图片，source={source_dir}, output={output_dir}")
    if not all_pairs:
        print("没有找到任何 .png，退出")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = HPCMEncoder(
        hpcm_root=hpcm_root,
        checkpoint_path=ckpt,
        model_name="HPCM_Base",
        scale_table_levels=60,
        pad_multiple=256,
    ).to(device)
    encoder.eval()

    to_tensor = ToTensor()
    done = 0
    skipped = 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    for abs_path, rel_path in tqdm(all_pairs, desc="extract"):
        out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".pt")
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            img = Image.open(abs_path).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                y_hat = encoder(x)
            # 存为 (1, 320, H', W')，单张
            torch.save(y_hat.cpu().float(), out_path)
            done += 1
        except Exception as e:
            print(f"Error {abs_path}: {e}")

    print(f"完成: 写入 {done}, 跳过 {skipped}")


if __name__ == "__main__":
    main()
