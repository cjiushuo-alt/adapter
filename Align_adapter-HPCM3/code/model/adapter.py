"""
Adapter Module (HPCM 版本)

将 HPCM g_a 量化特征对齐到 SigLIP 和 DINOv2 的特征空间。

架构：
1. PreNeck: 将 HPCM 特征 (B, 320, 16, 16) 处理成序列格式 (B, 256, 320)（共享）
2. Neck_SigLIP: 将序列特征适配到 SigLIP 特征空间
3. Neck_DINO: 将序列特征适配到 DINOv2 特征空间

输入: (B, 320, 16, 16) - HPCM g_a 量化特征
输出:
    - siglip_features: (B, N, siglip_dim) - 适配到 SigLIP 的特征
    - dino_features: (B, N, dino_dim) - 适配到 DINOv2 的特征
    其中 N = 16*16 = 256
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .pre_neck import PreNeck
from .neck import Neck


class Adapter(nn.Module):
    """
    Adapter 模块（HPCM 版本）：将 HPCM g_a 量化特征对齐到 SigLIP 和 DINOv2 的特征空间。

    使用共享的 PreNeck 将 (B, 320, 16, 16) 转为 (B, 256, 320)，再分别通过两个 Neck 适配到不同特征空间。
    """

    def __init__(
        self,
        # PreNeck 参数（HPCM3：ResNet 版，与 pre_neck.py 一致）
        hpcm_input_dim: int = 320,
        pre_neck_output_dim: int = 320,
        pre_neck_spatial_size: int = 16,
        pre_neck_num_res_blocks: int = 2,
        pre_neck_dropout: float = 0.0,
        # Neck 参数
        siglip_dim: int = 1152,  # SigLIP SO400M 的 embed_dim
        dino_dim: int = 1024,    # DINOv2 Large 的 embed_dim
        neck_num_heads: int = 8,
        neck_ffn_hidden_dim: Optional[int] = None,
        neck_dropout: float = 0.0,
        neck_activation: str = "gelu",
    ):
        """
        Args:
            hpcm_input_dim: HPCM 特征通道数（默认 320）
            pre_neck_output_dim: PreNeck 输出通道数（enhanced_dim），与 Neck 的 in_dim 一致（默认 320）
            pre_neck_spatial_size: PreNeck 空间尺寸 H=W（默认 16），序列长度 = spatial_size ** 2
            pre_neck_num_res_blocks: PreNeck ResNet Block 数量（HPCM3 默认 2）
            pre_neck_dropout: PreNeck Dropout 概率
            siglip_dim: SigLIP 特征维度（默认 1152）
            dino_dim: DINOv2 特征维度（默认 1024）
            neck_num_heads: Neck 注意力头数
            neck_ffn_hidden_dim: Neck Feed Forward 隐藏层维度（None 表示 4 倍输出维度）
            neck_dropout: Neck Dropout 概率
            neck_activation: Neck 激活函数类型
        """
        super().__init__()

        self.hpcm_input_dim = hpcm_input_dim
        self.pre_neck_output_dim = pre_neck_output_dim
        self.siglip_dim = siglip_dim
        self.dino_dim = dino_dim

        # 共享的 PreNeck（HPCM3 ResNet 版）：(B, 320, 16, 16) -> (B, 256, 320)
        self.pre_neck = PreNeck(
            hpcm_feature_dim=hpcm_input_dim,
            enhanced_dim=pre_neck_output_dim,
            target_spatial_size=(pre_neck_spatial_size, pre_neck_spatial_size),
            num_res_blocks=pre_neck_num_res_blocks,
            dropout=pre_neck_dropout,
        )

        # Neck for SigLIP
        self.neck_siglip = Neck(
            in_dim=pre_neck_output_dim,
            out_dim=siglip_dim,
            num_heads=neck_num_heads,
            ffn_hidden_dim=neck_ffn_hidden_dim,
            dropout=neck_dropout,
            activation=neck_activation,
        )

        # Neck for DINOv2
        self.neck_dino = Neck(
            in_dim=pre_neck_output_dim,
            out_dim=dino_dim,
            num_heads=neck_num_heads,
            ffn_hidden_dim=neck_ffn_hidden_dim,
            dropout=neck_dropout,
            activation=neck_activation,
        )
    
    def forward(
        self,
        hpcm_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hpcm_features: HPCM g_a 量化特征 (B, 320, 16, 16)

        Returns:
            siglip_features: 适配到 SigLIP 的特征 (B, N, siglip_dim)
            dino_features: 适配到 DINOv2 的特征 (B, N, dino_dim)
            其中 N = 16*16 = 256（序列长度）
        """
        # Step 1: 通过共享的 PreNeck 处理 HPCM 特征
        # (B, 320, 16, 16) -> (B, 256, 320)
        pre_neck_output = self.pre_neck(hpcm_features)

        # Step 2: 分别通过两个 Neck 适配到不同特征空间
        # (B, 256, 320) -> (B, 256, siglip_dim)
        siglip_features = self.neck_siglip(pre_neck_output)
        # (B, 256, 320) -> (B, 256, dino_dim)
        dino_features = self.neck_dino(pre_neck_output)

        return siglip_features, dino_features
    
    def get_output_dims(self) -> Tuple[int, int]:
        """
        获取输出特征维度
        
        Returns:
            (siglip_dim, dino_dim)
        """
        return self.siglip_dim, self.dino_dim
    
    def get_sequence_length(self) -> int:
        """
        获取序列长度

        Returns:
            序列长度（默认 256 = 16*16）
        """
        return getattr(self.pre_neck, "sequence_length", 16 * 16)


if __name__ == "__main__":
    # 测试代码（HPCM 版本）
    print("=" * 60)
    print("Adapter 模块测试 (HPCM)")
    print("=" * 60)

    adapter = Adapter(
        hpcm_input_dim=320,
        pre_neck_output_dim=320,
        pre_neck_spatial_size=16,
        pre_neck_num_res_blocks=2,
        pre_neck_dropout=0.0,
        siglip_dim=1152,
        dino_dim=1024,
        neck_num_heads=8,
        neck_dropout=0.0,
    )

    # 模拟输入：HPCM g_a 量化特征 (B, 320, 16, 16)
    batch_size = 2
    hpcm_features = torch.randn(batch_size, 320, 16, 16)

    print(f"输入形状: {hpcm_features.shape} (HPCM g_a 量化特征)")
    print(f"  - Batch: {hpcm_features.shape[0]}, 通道: {hpcm_features.shape[1]}, 空间: {hpcm_features.shape[2]}×{hpcm_features.shape[3]}")

    siglip_features, dino_features = adapter(hpcm_features)

    print(f"\n输出形状:")
    print(f"  SigLIP特征: {siglip_features.shape}")
    print(f"  DINOv2特征: {dino_features.shape}")

    expected_seq_len = 16 * 16
    assert siglip_features.shape == (batch_size, expected_seq_len, 1152), f"SigLIP 应为 (B, 256, 1152), 得到 {siglip_features.shape}"
    assert dino_features.shape == (batch_size, expected_seq_len, 1024), f"DINO 应为 (B, 256, 1024), 得到 {dino_features.shape}"
    print("\n✓ 输出尺寸验证通过")

    total_params = sum(p.numel() for p in adapter.parameters())
    pre_neck_params = sum(p.numel() for p in adapter.pre_neck.parameters())
    print(f"\n参数: PreNeck {pre_neck_params:,}, 总 {total_params:,}")
    print(f"get_output_dims: {adapter.get_output_dims()}, get_sequence_length: {adapter.get_sequence_length()}")

    print("\n" + "=" * 60)
    print("✓ Adapter (HPCM) 模块测试通过")
    print("=" * 60)

