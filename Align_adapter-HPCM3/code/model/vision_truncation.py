"""
Vision Backbone 构建与截断工具
"""

from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch.nn as nn

from prismatic.models.materialize import get_vision_backbone_and_transform
from prismatic.models.backbones.vision.base_vision import unpack_tuple

from Align_adapter.vision_ckpt.load_partial_checkpoint import load_vision_backbone_from_partial


PathLike = Union[str, Path]


def build_truncated_vision_backbone(
    vision_backbone_id: str,
    image_resize_strategy: str,
    image_sequence_len: int,
    max_layer: Optional[int],
    *,
    partial_checkpoint_path: Optional[PathLike] = None,
    freeze_backbone: bool = True,
):
    """
    构建Prismatic Vision Backbone，并根据提供的checkpoint执行权重加载与层截断

    Args:
        vision_backbone_id: Backbone ID，例如"dinosiglip-vit-so-224px"
        image_resize_strategy: 图像resize策略
        image_sequence_len: 图像序列长度
        max_layer: 截断层索引（0-based，包含）；None 表示保留完整视觉塔
        partial_checkpoint_path: 部分权重checkpoint路径（可选）
        freeze_backbone: 是否冻结整条backbone

    Returns:
        Tuple[vision_backbone, image_transform]
    """
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        vision_backbone_id=vision_backbone_id,
        image_resize_strategy=image_resize_strategy,
        image_sequence_len=image_sequence_len,
    )

    # 先截断模型，再加载权重，这样可以避免报告缺失的键。
    # full-visual distillation 使用 None，保留全部 blocks。
    if max_layer is not None:
        truncate_vision_backbone_layers(vision_backbone, max_layer)

    if partial_checkpoint_path:
        checkpoint_path = Path(partial_checkpoint_path)
        load_vision_backbone_from_partial(
            checkpoint_path=checkpoint_path,
            vision_backbone=vision_backbone,
            strict=False,
            freeze_loaded=freeze_backbone,
            max_layer=max_layer,  # None 时从完整 VLA checkpoint 加载全视觉塔
        )

    if freeze_backbone:
        for param in vision_backbone.parameters():
            param.requires_grad_(False)
        vision_backbone.eval()

    return vision_backbone, image_transform


def truncate_vision_backbone_layers(vision_backbone: Any, max_layer: int) -> bool:
    """
    将Vision Backbone的Transformer层截断到指定层（包含max_layer）
    
    Args:
        vision_backbone: Vision Backbone实例
        max_layer: 截断层索引（0-based，包含），必需参数
    """
    if max_layer < 0:
        raise ValueError(f"max_layer必须 >= 0，但得到 {max_layer}")

    truncated = False

    if hasattr(vision_backbone, "dino_featurizer") and hasattr(vision_backbone, "siglip_featurizer"):
        truncated |= _truncate_vit_featurizer(vision_backbone.dino_featurizer, max_layer)
        truncated |= _truncate_vit_featurizer(vision_backbone.siglip_featurizer, max_layer)
    elif hasattr(vision_backbone, "featurizer"):
        truncated |= _truncate_vit_featurizer(vision_backbone.featurizer, max_layer)
    else:
        raise ValueError("无法在当前Vision Backbone上执行截断操作。")

    if truncated:
        print(f"✓ Vision Backbone已截断至Transformer层 {max_layer}（包含）")

    return truncated


def _truncate_vit_featurizer(featurizer: nn.Module, max_layer: int) -> bool:
    """
    将TIMM VisionTransformer的blocks截断到指定层
    """
    if not hasattr(featurizer, "blocks"):
        return False

    blocks = featurizer.blocks
    total_blocks = len(blocks)

    if max_layer >= total_blocks - 1:
        return False

    if isinstance(blocks, nn.Sequential):
        kept_blocks = nn.Sequential(*list(blocks.children())[: max_layer + 1])
    else:
        kept_blocks = nn.Sequential(*blocks[: max_layer + 1])

    featurizer.blocks = kept_blocks
    featurizer.forward = unpack_tuple(
        partial(featurizer.get_intermediate_layers, n={len(featurizer.blocks) - 1})
    )

    return True


