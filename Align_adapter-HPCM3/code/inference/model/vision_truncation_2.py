"""
Vision Backbone 构建与截断工具（仅支持：保留 min_layer 及之后的层）

截断语义（包含关系）：
- 参数 min_layer：0-based 层索引。
- 舍弃：层索引 [0, min_layer-1]（包含第 0 层，不包含第 min_layer 层）。
- 保留：层索引 >= min_layer 的所有层（包含第 min_layer 层及之后）。
- min_layer=0 时不截断，保留全部层。

与 vla-adapter 一致：DINO/SigLIP 返回的特征为倒数第二层的输出（不包含最后一层）。
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union
import types

import torch.nn as nn

from prismatic.models.materialize import get_vision_backbone_and_transform

PathLike = Union[str, Path]


def build_posterior_vision_backbone(
    vision_backbone_id: str,
    image_resize_strategy: str,
    image_sequence_len: int,
    min_layer: int,
    *,
    freeze_backbone: bool = True,
) -> Tuple[Any, Any]:
    """
    构建 Prismatic Vision Backbone，并做「保留 min_layer 及之后的层」式截断。

    截断语义（包含关系）：
    - 舍弃：层索引 [0, min_layer-1]（包含第 0 层，不包含第 min_layer 层），即前 min_layer 层。
    - 保留：层索引 >= min_layer 的所有层（包含第 min_layer 层及之后）。

    若 min_layer=0，则不截断，保留全部层。

    Args:
        vision_backbone_id: Backbone ID，例如 "dinosiglip-vit-so-224px"。
        image_resize_strategy: 图像 resize 策略。
        image_sequence_len: 图像序列长度。
        min_layer: 保留的起始层索引（0-based，包含）。舍弃索引 < min_layer 的层，保留索引 >= min_layer 的层。
        freeze_backbone: 是否冻结 backbone。

    Returns:
        (vision_backbone, image_transform)
    """
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        vision_backbone_id=vision_backbone_id,
        image_resize_strategy=image_resize_strategy,
        image_sequence_len=image_sequence_len,
    )

    if min_layer > 0:
        skip_vision_backbone_layers(vision_backbone, min_layer)

    if freeze_backbone:
        for param in vision_backbone.parameters():
            param.requires_grad_(False)
        vision_backbone.eval()

    return vision_backbone, image_transform


def skip_vision_backbone_layers(vision_backbone: Any, min_layer: int) -> bool:
    """
    对 Vision Backbone 做「保留 min_layer 及之后的层」式截断。

    截断语义（包含关系）：
    - 舍弃：层索引 [0, min_layer-1]（包含第 0 层，不包含第 min_layer 层）。
    - 保留：层索引 >= min_layer 的所有层（包含第 min_layer 层及之后）。

    Args:
        vision_backbone: Vision Backbone 实例。
        min_layer: 保留的起始层索引（0-based，包含）。为 0 时不截断，保留所有层。

    Returns:
        是否进行了截断。
    """
    if min_layer < 0:
        raise ValueError(f"min_layer必须 >= 0，但得到 {min_layer}")
    if min_layer == 0:
        return False

    skipped = False
    if hasattr(vision_backbone, "featurizer") and hasattr(vision_backbone, "fused_featurizer"):
        skipped |= _skip_vit_featurizer(vision_backbone.featurizer, min_layer)
        skipped |= _skip_vit_featurizer(vision_backbone.fused_featurizer, min_layer)
    elif hasattr(vision_backbone, "dino_featurizer") and hasattr(vision_backbone, "siglip_featurizer"):
        skipped |= _skip_vit_featurizer(vision_backbone.dino_featurizer, min_layer)
        skipped |= _skip_vit_featurizer(vision_backbone.siglip_featurizer, min_layer)
    elif hasattr(vision_backbone, "featurizer"):
        skipped |= _skip_vit_featurizer(vision_backbone.featurizer, min_layer)
    else:
        raise ValueError("无法在当前Vision Backbone上执行截断操作。")

    if skipped:
        print(f"✓ Vision Backbone已截断：舍弃层 [0, {min_layer - 1}]，保留层 [{min_layer}, end]（包含）")

    return skipped


def _skip_vit_featurizer(featurizer: nn.Module, min_layer: int) -> bool:
    """
    对单个 ViT featurizer 的 blocks 做「保留 min_layer 及之后的层」式截断。

    截断语义：
    - 舍弃：blocks[0 : min_layer]，即层索引 [0, min_layer-1]（不包含 min_layer）。
    - 保留：blocks[min_layer :]，即层索引 [min_layer, end]（包含 min_layer）。

    同时将 forward 改为接受特征输入 (B, N, C)，跳过 patch_embed 等，且输出为
    当前 blocks 的「倒数第二层」输出（与 Prismatic 的 get_intermediate_layers(n=L-2) 一致）。
    """
    if not hasattr(featurizer, "blocks"):
        return False

    blocks = featurizer.blocks
    total_blocks = len(blocks)
    if min_layer >= total_blocks:
        return False

    if isinstance(blocks, nn.Sequential):
        kept_layers = list(blocks.children())[min_layer:]
        kept_blocks = nn.Sequential(*kept_layers)
    else:
        kept_layers = blocks[min_layer:]
        kept_blocks = nn.Sequential(*kept_layers)

    featurizer.blocks = kept_blocks

    def forward_features(self, x):
        if len(self.blocks) == 0:
            return x
        if len(self.blocks) > 1:
            for i, block in enumerate(self.blocks):
                if i == len(self.blocks) - 1:
                    break
                x = block(x)
        return x

    featurizer.forward = types.MethodType(forward_features, featurizer)
    return True
