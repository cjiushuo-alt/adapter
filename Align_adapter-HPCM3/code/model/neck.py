"""
Neck Module (Transform-Neck)

将PreNeck输出的序列特征通过Transformer结构进行特征变换

数据流：
1. Linear Projection: 调整特征维度
2. Residual Block 1 (Self-Attention): LayerNorm -> Self-Attention -> Residual
3. Residual Block 2 (Feed Forward): LayerNorm -> MLP -> Residual

输入: (B, N, C_in) - PreNeck输出的序列特征
输出: (B, N, C_out) - 变换后的序列特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelfAttention(nn.Module):
    """
    Self-Attention层
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim: 特征维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V投影
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B, num_heads, N, head_dim)
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力到V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        
        # 输出投影
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class FeedForward(nn.Module):
    """
    Feed Forward Network (MLP)
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        """
        Args:
            dim: 输入/输出维度
            hidden_dim: 隐藏层维度（默认4倍dim）
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        
        Returns:
            (B, N, C)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ResidualBlock1(nn.Module):
    """
    第一个残差块：Self-Attention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, num_heads, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        
        Returns:
            (B, N, C)
        """
        # 残差连接：输入直接跳过
        residual = x
        
        # 处理路径：LayerNorm -> Self-Attention
        x = self.norm(x)
        x = self.attention(x)
        
        # 残差连接
        x = x + residual
        
        return x


class ResidualBlock2(nn.Module):
    """
    第二个残差块：Feed Forward
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, dropout, activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        
        Returns:
            (B, N, C)
        """
        # 残差连接：输入直接跳过
        residual = x
        
        # 处理路径：LayerNorm -> Feed Forward
        x = self.norm(x)
        x = self.ffn(x)
        
        # 残差连接
        x = x + residual
        
        return x


class Neck(nn.Module):
    """
    Transform-Neck模块
    
    输入: (B, N, C_in) - PreNeck输出的序列特征
    输出: (B, N, C_out) - 变换后的序列特征
    """
    
    def __init__(
        self,
        in_dim: int = 320,
        out_dim: int = 256,
        num_heads: int = 8,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        """
        Args:
            in_dim: 输入特征维度（默认320，对应PreNeck输出）
            out_dim: 输出特征维度（默认256）
            num_heads: 注意力头数
            ffn_hidden_dim: Feed Forward隐藏层维度（默认4倍out_dim）
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 1. Linear Projection: 调整特征维度
        self.linear_proj = nn.Linear(in_dim, out_dim)
        
        # 2. Residual Block 1: Self-Attention
        self.residual_block1 = ResidualBlock1(
            dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # 3. Residual Block 2: Feed Forward
        self.residual_block2 = ResidualBlock2(
            dim=out_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C_in) - PreNeck输出的序列特征
        
        Returns:
            (B, N, C_out) - 变换后的序列特征
        """
        # Step 1: Linear Projection
        x = self.linear_proj(x)  # (B, N, C_in) -> (B, N, C_out)
        
        # Step 2: Residual Block 1 (Self-Attention)
        x = self.residual_block1(x)  # (B, N, C_out)
        
        # Step 3: Residual Block 2 (Feed Forward)
        x = self.residual_block2(x)  # (B, N, C_out)
        
        return x


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Neck 模块测试")
    print("=" * 60)
    
    # 创建模型
    neck = Neck(
        in_dim=320,
        out_dim=256,
        num_heads=8,
        dropout=0.0,
    )
    
    # 模拟输入：PreNeck输出的序列特征 (B, N, C)
    batch_size = 2
    seq_length = 256  # 16×16
    feature_dim = 320  # PreNeck输出维度
    
    x = torch.randn(batch_size, seq_length, feature_dim)
    
    print(f"输入形状: {x.shape}")
    print(f"  - Batch: {x.shape[0]}")
    print(f"  - 序列长度: {x.shape[1]}")
    print(f"  - 特征维度: {x.shape[2]}")
    
    # 前向传播
    output = neck(x)
    
    print(f"\n输出形状: {output.shape}")
    print(f"  - Batch: {output.shape[0]}")
    print(f"  - 序列长度: {output.shape[1]}")
    print(f"  - 特征维度: {output.shape[2]}")
    
    # 验证输出尺寸
    assert output.shape[0] == batch_size, "Batch size should match"
    assert output.shape[1] == seq_length, "Sequence length should match"
    assert output.shape[2] == 256, "Output feature dimension should be 256"
    print("\n✓ 输出尺寸验证通过")
    
    # 统计参数
    total_params = sum(p.numel() for p in neck.parameters())
    trainable_params = sum(p.numel() for p in neck.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试与PreNeck的连接
    print("\n" + "=" * 60)
    print("PreNeck + Neck 联合测试")
    print("=" * 60)
    
    try:
        from .pre_neck import PreNeck
    except ImportError:
        print("无法导入PreNeck，跳过联合测试")
        PreNeck = None
    
    if PreNeck is not None:
        pre_neck = PreNeck(
            hpcm_feature_dim=320,
            enhanced_dim=320,
            target_spatial_size=(16, 16),
            num_res_blocks=2,
            dropout=0.0,
        )
        
        # HPCM g_a 量化特征输入 (B, 320, 16, 16)
        hpcm_features = torch.randn(batch_size, 320, 16, 16)
        print(f"HPCM特征输入: {hpcm_features.shape}")
        
        # PreNeck处理
        pre_neck_output = pre_neck(hpcm_features)
        print(f"PreNeck输出: {pre_neck_output.shape}")
        
        # Neck处理
        neck_output = neck(pre_neck_output)
        print(f"Neck输出: {neck_output.shape}")
        
        print("\n✓ PreNeck + Neck 联合测试通过")

