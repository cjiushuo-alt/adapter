"""
加载部分权重checkpoint

用于训练时加载预先提取的部分权重checkpoint
"""

import torch
from pathlib import Path
from typing import Dict
from safetensors import safe_open


def _is_vision_key(key: str) -> bool:
    """Return whether a key belongs to a standalone or full-VLA vision tower."""
    normalized = key.removeprefix("vision_backbone.")
    return normalized.startswith(
        ("featurizer.", "fused_featurizer.", "dino_featurizer.", "siglip_featurizer.")
    )


def load_partial_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """
    加载部分权重checkpoint

    Args:
        checkpoint_path: checkpoint文件路径（.safetensors或.pt）

    Returns:
        包含部分权重的state_dict
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        state_dict = {}
        with safe_open(str(checkpoint_path), framework="pt") as f:
            for key in f.keys():
                # Full VLA checkpoints are multi-gigabyte.  Stream only the
                # vision tensors instead of materializing the LLM/policy state.
                if _is_vision_key(key):
                    state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {key: value for key, value in state_dict.items() if _is_vision_key(key)}

    return state_dict


def load_vision_backbone_from_partial(
    checkpoint_path: Path,
    vision_backbone,
    strict: bool = False,
    freeze_loaded: bool = True,
    max_layer: int = None,
) -> None:
    """
    从部分权重checkpoint加载到vision backbone

    Args:
        checkpoint_path: 部分权重checkpoint路径
        vision_backbone: vision backbone模型实例
        strict: 是否严格匹配（默认False）
        freeze_loaded: 是否冻结已加载的权重（默认True）
        max_layer: 最大层索引（用于过滤不需要的层），如果为None则不过滤
    """
    print(f"从部分权重checkpoint加载: {checkpoint_path}")

    # 加载checkpoint
    partial_state_dict = load_partial_checkpoint(checkpoint_path)

    print(f"加载 {len(partial_state_dict)} 个权重...")

    # 处理键名：移除vision_backbone.前缀（如果存在）
    # checkpoint可能来自完整VLA模型，键名格式为：
    # - vision_backbone.featurizer.blocks.0... (融合格式)
    # - vision_backbone.dino_featurizer.blocks.0... (分离格式)
    # - vision_backbone.siglip_featurizer.blocks.0... (分离格式)
    # Prismatic Vision Backbone期望的键名格式为dino_featurizer.blocks.0...和siglip_featurizer.blocks.0...
    processed_state_dict = {}
    for key, value in partial_state_dict.items():
        # 移除vision_backbone.前缀
        if key.startswith("vision_backbone."):
            new_key = key.replace("vision_backbone.", "", 1)
        else:
            new_key = key

        # 如果指定了max_layer，过滤掉超过max_layer的层
        if max_layer is not None:
            # 检查是否是blocks层的权重
            if ".blocks." in new_key:
                # 提取层索引：featurizer.blocks.3.norm1.weight -> 3
                parts = new_key.split(".blocks.")
                if len(parts) == 2:
                    try:
                        layer_idx = int(parts[1].split(".")[0])
                        if layer_idx > max_layer:
                            # 跳过超过max_layer的层
                            continue
                    except ValueError:
                        # 无法解析层索引，保留
                        pass

        # 处理融合格式：
        # 注意：虽然变量名叫dino_featurizer和siglip_featurizer，但实际维度是：
        # - dino_featurizer.embed_dim = 1024 (实际是SigLIP的维度)
        # - siglip_featurizer.embed_dim = 1152 (实际是DINOv2的维度)
        # 根据维度匹配：
        # - featurizer.* (1024维) -> dino_featurizer.* (1024维)
        # - fused_featurizer.* (1152维) -> siglip_featurizer.* (1152维)
        if new_key.startswith("featurizer.") and "dino" not in new_key and "siglip" not in new_key and "fused" not in new_key:
            # 融合格式的featurizer是1024维，映射到dino_featurizer（1024维）
            dino_key = new_key.replace("featurizer.", "dino_featurizer.", 1)
            # 处理键名转换：scale_factor -> gamma (DINO使用gamma，checkpoint使用scale_factor)
            if ".scale_factor" in dino_key:
                dino_key = dino_key.replace(".scale_factor", ".gamma")
            processed_state_dict[dino_key] = value
        elif new_key.startswith("fused_featurizer."):
            # fused_featurizer是1152维，映射到siglip_featurizer（1152维）
            siglip_key = new_key.replace("fused_featurizer.", "siglip_featurizer.", 1)
            processed_state_dict[siglip_key] = value
        else:
            # 已经是分离格式（dino_featurizer或siglip_featurizer）或基础组件，直接使用
            # 如果是dino_featurizer的键，也需要转换scale_factor -> gamma
            final_key = new_key
            if "dino_featurizer" in final_key and ".scale_factor" in final_key:
                final_key = final_key.replace(".scale_factor", ".gamma")
            processed_state_dict[final_key] = value

    filter_info = "移除了vision_backbone.前缀"
    if max_layer is not None:
        filter_info += f"，过滤了超过max_layer={max_layer}的层"

    # 检查是否有融合格式的键
    featurizer_keys = [k for k in partial_state_dict.keys() if (k.startswith("vision_backbone.featurizer.") or (not k.startswith("vision_backbone.") and k.startswith("featurizer.") and "dino" not in k and "siglip" not in k and "fused" not in k))]
    fused_keys = [k for k in partial_state_dict.keys() if "fused_featurizer" in k]

    if featurizer_keys:
        filter_info += "，featurizer（1024维）已映射到dino_featurizer（1024维）"
    if fused_keys:
        filter_info += "，fused_featurizer（1152维）已映射到siglip_featurizer（1152维）"

    # 检查是否有scale_factor需要转换为gamma（在原始checkpoint中）
    scale_factor_keys = [k for k in partial_state_dict.keys() if ".scale_factor" in k]
    if scale_factor_keys:
        filter_info += "，scale_factor已转换为gamma"

    print(f"处理后 {len(processed_state_dict)} 个权重（{filter_info}）")

    # 加载到vision backbone
    missing_keys, unexpected_keys = vision_backbone.load_state_dict(
        processed_state_dict, strict=strict
    )

    if missing_keys:
        print(f"缺失的键 ({len(missing_keys)} 个，这是正常的，因为只加载了部分层):")
        # 只显示前几个作为示例
        for key in list(missing_keys)[:5]:
            print(f"  {key}")
        if len(missing_keys) > 5:
            print(f"  ... 还有 {len(missing_keys) - 5} 个")

    if unexpected_keys:
        print(f"意外的键 ({len(unexpected_keys)} 个):")
        for key in list(unexpected_keys)[:5]:
            print(f"  {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... 还有 {len(unexpected_keys) - 5} 个")

    # 冻结已加载的权重
    if freeze_loaded:
        frozen_count = 0
        for name, param in vision_backbone.named_parameters():
            # 直接使用处理后的键名（已移除vision_backbone.前缀）
            if name in processed_state_dict:
                param.requires_grad_(False)
                frozen_count += 1

        print(f"冻结了 {frozen_count} 个已加载的参数")

    print(f"✓ 成功加载部分权重到vision backbone")


if __name__ == "__main__":
    # 测试代码
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    checkpoint_path = project_root / "Align_adapter/ckpt/vision_backbone_partial_layer2.safetensors"

    if checkpoint_path.exists():
        print(f"测试加载: {checkpoint_path}")
        state_dict = load_partial_checkpoint(checkpoint_path)
        print(f"加载了 {len(state_dict)} 个权重")
        print(f"示例键:")
        for key in list(state_dict.keys())[:5]:
            print(f"  {key}")
    else:
        print(f"Checkpoint不存在: {checkpoint_path}")
        print("请先运行 extract_partial_weights.py 提取权重")
