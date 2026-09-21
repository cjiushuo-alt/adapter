"""
Pre-Neck Module（HPCM3 版本）

将HPCM特征处理成适合neck输入的序列格式

将 HPCM g_a 量化特征 (B, 320, 16, 16) 转为 neck 输入的序列格式 (B, 256, 320)。

流程：
1. ResNet Blocks：（2个ResNet Block）
2. 序列化：(B, 320, 16, 16) -> (B, 256, 320)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    """Swish activation"""
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    """Group Normalization"""
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """
    ResNet Block for feature enhancement
    参考CGIC decoder中的ResnetBlock设计
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        # First normalization and convolution
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Second normalization and convolution
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Shortcut connection
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C_in, H, W), Returns: (B, C_out, H, W)"""
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        
        return x + h


class PreNeck(nn.Module):
    """
    将HPCM特征处理成适合neck输入的序列格式
    
    输入: (B, 320, 16, 16) - HPCM特征（对于256×256输入图像）
    输出: (B, 256, 320) - 序列特征（256个token，每个320维）
    """
    
    def __init__(
        self,
        hpcm_feature_dim: int = 320,
        enhanced_dim: int = 320,
        target_spatial_size: tuple = (16, 16),
        num_res_blocks: int = 2,  # 默认2个ResNet Block
        dropout: float = 0.0,
    ):
        """
        Args:
            dropout: Dropout概率
        """
        super().__init__()
        self.hpcm_feature_dim = hpcm_feature_dim
        self.enhanced_dim = enhanced_dim
        self.target_spatial_size = target_spatial_size
        self.sequence_length = target_spatial_size[0] * target_spatial_size[1]
        
        # ResNet Blocks
        intermediate_channels = [320, 320, 320]
        self.blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            block_in = intermediate_channels[min(i, len(intermediate_channels) - 1)]
            block_out = intermediate_channels[min(i + 1, len(intermediate_channels) - 1)]
            self.blocks.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=dropout,
                )
            )
    
    def forward(self, hpcm_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hpcm_features: HPCM特征 (B, 320, H, W)
                - 对于256×256输入，H=W=16
        
        Returns:
            序列特征 (B, N, 320)，其中N = 16 * 16 = 256
        """

        h = hpcm_features
        # Step 1: 通过2个ResNet Blocks
        for block in self.blocks:
            h = block(h)

        # Step 2: 转换为序列 (B, 320, 16, 16) -> (B, 256, 320)
        B, C, H, W = h.shape
        N = H * W
        h = h.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        h = h.view(B, N, C)  # (B, N, C)
        
        return h  # (B, 256, 320)


if __name__ == "__main__":
    # 测试代码
    model = PreNeck(
        hpcm_feature_dim=320,
        enhanced_dim=320,
        target_spatial_size=(16, 16),
        num_res_blocks=2,  # 64 -> 64 -> 128 -> 192 -> 256，然后conv_out: 256 -> 320
        dropout=0.0,
    )
    
    # 模拟输入：CGIC特征 (B, 4, 64, 64) - 对应256×256输入图像
    x = torch.randn(2, 320, 16, 16)
    
    print("=" * 60)
    print("PreNeck 测试")
    print("=" * 60)
    print(f"输入形状: {x.shape} (CGIC特征，对应256×256输入图像)")
    
    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"序列长度: {output.shape[1]}, 特征维度: {output.shape[2]}")
    
    # 验证输出尺寸
    expected_seq_len = 16 * 16
    assert output.shape[1] == expected_seq_len, f"序列长度应该是{expected_seq_len}，实际是{output.shape[1]}"
    assert output.shape[2] == 320, f"特征维度应该是320，实际是{output.shape[2]}"
    print("✓ 输出尺寸验证通过")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数: {total_params:,}")

