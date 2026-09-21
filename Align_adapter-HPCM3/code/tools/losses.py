"""
损失函数模块

定义用于特征对齐的各种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Dict, Optional


class DistLoss(nn.Module):
    """
    距离损失（MSE）
    
    计算对齐后的特征与视觉编码器特征之间的距离损失
    """
    
    def __init__(
        self,
        reduction: str = "mean",
    ):
        """
        Args:
            reduction: 损失归约方式（"mean", "sum", "none"）
        """
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        aligned_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算距离损失（MSE）
        
        Args:
            aligned_features: 对齐后的特征 (B, N, D)
            vision_features: 视觉编码器特征 (B, N, D)
        
        Returns:
            损失值（标量或tensor，取决于reduction）
        """
        loss = self.mse_loss(aligned_features, vision_features)
        return loss


class CombinedDistLoss(nn.Module):
    """
    组合距离损失
    
    同时计算SigLIP和DINOv2的距离损失
    """
    
    def __init__(
        self,
        siglip_loss_weight: float = 1.0,
        dino_loss_weight: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Args:
            siglip_loss_weight: SigLIP损失权重
            dino_loss_weight: DINOv2损失权重
            reduction: 损失归约方式
        """
        super().__init__()
        self.siglip_loss_weight = siglip_loss_weight
        self.dino_loss_weight = dino_loss_weight
        
        self.dist_loss = DistLoss(reduction=reduction)
    
    def forward(
        self,
        aligned_siglip: torch.Tensor,
        aligned_dino: torch.Tensor,
        vision_siglip: torch.Tensor,
        vision_dino: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合距离损失
        
        Args:
            aligned_siglip: 对齐后的SigLIP特征 (B, N, siglip_dim)
            aligned_dino: 对齐后的DINOv2特征 (B, N, dino_dim)
            vision_siglip: 视觉编码器SigLIP特征 (B, N, siglip_dim)
            vision_dino: 视觉编码器DINOv2特征 (B, N, dino_dim)
        
        Returns:
            Dict包含:
                - total_loss: 总损失
                - siglip_loss: SigLIP损失
                - dino_loss: DINOv2损失
        """
        # 计算SigLIP距离损失
        siglip_loss = self.dist_loss(aligned_siglip, vision_siglip)
        
        # 计算DINOv2距离损失
        dino_loss = self.dist_loss(aligned_dino, vision_dino)
        
        # 组合损失
        total_loss = (
            self.siglip_loss_weight * siglip_loss +
            self.dino_loss_weight * dino_loss
        )
        
        return {
            "total_loss": total_loss,
            "siglip_loss": siglip_loss,
            "dino_loss": dino_loss,
        }
