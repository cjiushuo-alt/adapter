"""
Align Adapter (HPCM2) Dataset

从多个 LIBERO 数据集目录加载图像，供 HPCM2 模型训练。
仅需图片，HPCM 特征在模型内由 HPCMEncoder 实时从图像提取，无需预生成 .pt。
"""

import glob
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def center_crop(image: Image.Image) -> Image.Image:
    """中心裁剪为正方形。"""
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class LiberoImageDataset(Dataset):
    """
    LIBERO 图像数据集（仅图片，无 .pt）。

    每个样本：一张图。HPCM 特征在模型 forward 内由 HPCMEncoder 实时提取。
    """

    def __init__(
        self,
        image_dataset_dirs: List[str],
        image_size: int = 224,
        image_keys: Optional[List[str]] = None,
        train_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
        split_unit: str = "trajectory",
        split_manifest_path: Optional[str] = None,
    ):
        """
        Args:
            image_dataset_dirs: 图像数据集目录列表（每个为包含 episode_* 的路径）
            image_size: 目标图像尺寸（建议 224，与 Vision Backbone 一致）
            image_keys: 图像键名，如 ["image", "wrist_image"]，对应 step_*_{key}.png
            train_ratio: 训练集比例
            is_train: 是否训练集
            seed: 划分 train/val 的随机种子
            split_unit: trajectory（论文实验推荐）或 frame（仅兼容旧实验）
            split_manifest_path: trajectory 划分清单。首次运行原子生成，之后严格复用。
        """
        if image_keys is None:
            image_keys = ["image", "wrist_image"]
        self.image_size = image_size
        self.image_keys = image_keys
        self.is_train = is_train

        if isinstance(image_dataset_dirs, str):
            image_dataset_dirs = [image_dataset_dirs]

        if split_unit not in {"trajectory", "frame"}:
            raise ValueError(f"split_unit 必须是 trajectory 或 frame，得到: {split_unit}")

        # 收集所有图像路径。episode_* 就是从 RLDS 导出的一条 trajectory。
        all_image_paths: List[str] = []
        episode_dirs: List[str] = []
        for dataset_dir in image_dataset_dirs:
            dataset_path = Path(dataset_dir)
            if not dataset_path.is_absolute():
                base_dir = Path(__file__).parent
                dataset_path = base_dir / dataset_dir
            episode_dirs.extend(str(path.resolve()) for path in dataset_path.glob("episode_*") if path.is_dir())
            for image_key in image_keys:
                pattern = str(dataset_path / "episode_*" / f"step_*_{image_key}.png")
                all_image_paths.extend(glob.glob(pattern, recursive=False))

        all_image_paths = sorted(set(all_image_paths))
        episode_dirs = sorted(set(episode_dirs))

        if len(all_image_paths) == 0:
            raise ValueError(
                "未找到任何图像！请检查 image_dataset_dirs 与 image_keys。"
            )

        if split_unit == "trajectory":
            if not episode_dirs:
                raise ValueError("未找到 episode_* trajectory 目录")
            manifest_path = Path(split_manifest_path).expanduser() if split_manifest_path else None
            if manifest_path is not None and not manifest_path.is_absolute():
                manifest_path = Path.cwd() / manifest_path
            split = self._load_or_create_trajectory_split(
                episode_dirs=episode_dirs,
                train_ratio=train_ratio,
                seed=seed,
                manifest_path=manifest_path,
            )
            selected_episodes = set(split["train_episodes"] if is_train else split["val_episodes"])
            self.image_paths = [
                path for path in all_image_paths if str(Path(path).parent.resolve()) in selected_episodes
            ]
            self.indices = list(range(len(self.image_paths)))
            if not self.image_paths:
                raise ValueError(f"trajectory split 的 {'train' if is_train else 'val'} 集为空")
            overlap = set(split["train_episodes"]) & set(split["val_episodes"])
            if overlap:
                raise ValueError(f"trajectory split 存在 {len(overlap)} 个重叠 episode")
            split_desc = (
                f"trajectory; train={len(split['train_episodes'])}, "
                f"val={len(split['val_episodes'])}, overlap=0"
            )
        else:
            # 仅为复现历史 checkpoint 保留；正式实验不应使用 frame split。
            self.image_paths = all_image_paths
            indices = list(range(len(self.image_paths)))
            random.Random(seed).shuffle(indices)
            split_idx = int(len(indices) * train_ratio)
            self.indices = indices[:split_idx] if is_train else indices[split_idx:]
            split_desc = "frame (legacy)"

        print(f"LiberoImageDataset ({'训练' if is_train else '验证'}):")
        print(f"  总图像数: {len(all_image_paths)}")
        print(f"  使用样本数: {len(self.indices)}")
        print(f"  划分方式: {split_desc}")
        print(f"  图像目录: {image_dataset_dirs[:2]}...")
        print(f"  image_keys: {image_keys}, image_size: {image_size}")

    @staticmethod
    def _load_or_create_trajectory_split(
        episode_dirs: List[str],
        train_ratio: float,
        seed: int,
        manifest_path: Optional[Path],
    ) -> Dict[str, Any]:
        expected = set(episode_dirs)
        if manifest_path is not None and manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("split_unit") != "trajectory":
                raise ValueError(f"非 trajectory manifest: {manifest_path}")
            if int(manifest.get("seed")) != seed or float(manifest.get("train_ratio")) != train_ratio:
                raise ValueError(f"manifest 的 seed/train_ratio 与配置不一致: {manifest_path}")
            recorded = set(manifest.get("train_episodes", [])) | set(manifest.get("val_episodes", []))
            if recorded != expected:
                raise ValueError(
                    f"manifest 与当前数据集不一致: recorded={len(recorded)}, current={len(expected)}"
                )
            return manifest

        shuffled = list(episode_dirs)
        random.Random(seed).shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        manifest = {
            "version": 1,
            "split_unit": "trajectory",
            "seed": seed,
            "train_ratio": train_ratio,
            "train_episodes": sorted(shuffled[:split_idx]),
            "val_episodes": sorted(shuffled[split_idx:]),
        }
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, manifest_path)
            print(f"已写入 trajectory split manifest: {manifest_path}")
        return manifest

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict: "image" -> (C, H, W), float32, 值域 [-1, 1]
        """
        actual_idx = self.indices[idx]
        image_path = self.image_paths[actual_idx]

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = center_crop(image)
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32))
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor / 127.5 - 1.0

        return {"image": image_tensor}


def create_loader(
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    **dataset_params,
) -> DataLoader:
    """创建 DataLoader，batch 为 dict: image (B, C, H, W)。"""
    dataset = LiberoImageDataset(**dataset_params)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=False,
    )


class LightningDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule，仅图像，无 .pt。"""

    def __init__(self, config: Dict[str, Any]):
        """
        config 需包含:
            - dataset_dir: 图像数据集根目录（含各 libero_* 子目录）
            - dataset_names: 如 ["libero_10_no_noops", ...]
            - image_keys, image_size, batch_size, num_workers, train_ratio, seed 等
        """
        super().__init__()
        self.config = config

        dataset_dir = Path(config.get("dataset_dir", "Align_adapter/dataset"))
        if not dataset_dir.is_absolute():
            dataset_dir = Path(__file__).parent.parent.parent / dataset_dir

        dataset_names = config.get("dataset_names", [
            "libero_10_no_noops",
            "libero_goal_no_noops",
            "libero_object_no_noops",
            "libero_spatial_no_noops",
        ])

        self.image_dataset_dirs = [str(dataset_dir / name) for name in dataset_names]
        self.image_keys = config.get("image_keys", ["image", "wrist_image"])
        self.image_size = config.get("image_size", 224)
        self.batch_size = config.get("batch_size", 8)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)
        self.persistent_workers = config.get("persistent_workers", True)
        self.prefetch_factor = config.get("prefetch_factor", 2)
        self.train_ratio = config.get("train_ratio", 0.9)
        self.seed = config.get("seed", 42)
        self.split_unit = config.get("split_unit", "trajectory")
        self.split_manifest_path = config.get("split_manifest_path")

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        return create_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            image_dataset_dirs=self.image_dataset_dirs,
            image_size=self.image_size,
            image_keys=self.image_keys,
            train_ratio=self.train_ratio,
            is_train=True,
            seed=self.seed,
            split_unit=self.split_unit,
            split_manifest_path=self.split_manifest_path,
        )

    def val_dataloader(self) -> DataLoader:
        return create_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            image_dataset_dirs=self.image_dataset_dirs,
            image_size=self.image_size,
            image_keys=self.image_keys,
            train_ratio=self.train_ratio,
            is_train=False,
            seed=self.seed,
            split_unit=self.split_unit,
            split_manifest_path=self.split_manifest_path,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("LiberoImageDataset 测试")
    print("=" * 60)

    root = Path(__file__).parent.parent
    image_base = root.parent / "Align_adapter" / "dataset"
    if not image_base.exists():
        image_base = root / "dataset"

    dataset = LiberoImageDataset(
        image_dataset_dirs=[str(image_base / "libero_goal_no_noops")],
        image_size=224,
        image_keys=["image"],
        train_ratio=0.9,
        is_train=True,
    )
    print(f"数据集大小: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  image: {sample['image'].shape}")
    print("=" * 60)
