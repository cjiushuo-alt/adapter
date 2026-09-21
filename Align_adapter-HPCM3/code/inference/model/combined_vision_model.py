"""
Combined Vision: HPCM decoder 输入(y_hat) + Align Adapter + Vision Backbone（推理）

HPCM3：图像 -> HPCM(y_hat) -> Adapter(ResNet PreNeck + Neck) -> 对齐特征 -> Vision Backbone 后段。
Adapter 与 Align_adapter-HPCM3/model/adapter.py 一致（PreNeck 为 ResNet，num_res_blocks=2）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# 项目根与 Align_adapter-HPCM3（dataset.hpcm_encoder）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HPCM3_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJECT_ROOT, _HPCM3_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model.adapter import Adapter
from .load_adapter import load_adapter_weights, default_adapter_weights_path
from dataset.hpcm_encoder import HPCMEncoder
from .vision_truncation_2 import build_posterior_vision_backbone, skip_vision_backbone_layers

logger = logging.getLogger(__name__)

# 默认 Adapter 配置（与 train_config / inference_config 一致；可被传入的 adapter_config 覆盖）
DEFAULT_ADAPTER_CONFIG: Dict[str, Any] = {
    "hpcm_input_dim": 320,
    "pre_neck_output_dim": 320,
    "pre_neck_spatial_size": 16,
    "pre_neck_num_res_blocks": 2,
    "pre_neck_dropout": 0.0,
    "siglip_dim": 1152,
    "dino_dim": 1024,
}


class CombinedVisionModel(nn.Module):
    """
    组合模型：HPCM(y_hat) + Adapter + Vision Backbone（推理）

    架构（与 Align_adapter/inference 风格一致）：
    1. HPCM：图像 -> 完整前向 -> hook 取 decoder 输入 y_hat (B, 320, H', W')，插值到 16×16
    2. Adapter：y_hat -> 对齐 SigLIP / DINO 特征空间
    3. Vision Backbone：图像 -> vision 特征（由 vision_truncation_2 构建）
    4. 融合：对齐特征 + vision 特征
    """

    def __init__(
        self,
        *,
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        vision_backbone_checkpoint: Optional[Path] = None,
        vision_backbone_max_layer: int = 2,
        image_resize_strategy: str = "letterbox",
        image_sequence_len: int = 1,
        device: Optional[torch.device] = None,
        # HPCM：完整模型前向 + hook 取 decoder 输入 y_hat
        hpcm_root: Optional[str] = None,
        hpcm_checkpoint: Optional[str] = None,
        hpcm_model_name: str = "HPCM_Base",
        hpcm_scale_table_levels: int = 60,
        freeze_hpcm_encoder: bool = True,
        # Adapter
        adapter_config: Optional[Dict[str, Any]] = None,
        adapter_weights_path: Optional[str] = None,
        # 可选：从 get_vla 加载的 OpenVLA 的 vision_backbone，注入后作为截断 vision 分支（与 checkpoint 二选一）
        external_vision_backbone: Optional[nn.Module] = None,
        freeze_vision_backbone: bool = True,
        train_adapter: bool = True,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== 1. HPCM：完整前向，hook 取 decoder 输入 y_hat ==========
        _hpcm_root = Path(hpcm_root) if hpcm_root else (_HPCM3_ROOT.parent / "HPCM")
        hpcm_ckpt = hpcm_checkpoint or str(_hpcm_root / "ckpt" / "0.0018.pth.tar")
        if hpcm_checkpoint and not Path(hpcm_checkpoint).is_absolute():
            hpcm_ckpt = str(_hpcm_root / hpcm_checkpoint)
        self.hpcm_encoder = HPCMEncoder(
            hpcm_root=str(_hpcm_root),
            checkpoint_path=hpcm_ckpt,
            model_name=hpcm_model_name,
            scale_table_levels=hpcm_scale_table_levels,
            pad_multiple=256,
        )
        if freeze_hpcm_encoder:
            for p in self.hpcm_encoder.parameters():
                p.requires_grad_(False)
            self.hpcm_encoder.eval()

        # ========== 2. Adapter（HPCM3：ResNet PreNeck，y_hat 320→Align）==========
        # 配置与代码解耦：以 DEFAULT_ADAPTER_CONFIG 为底，inference_config 传入项覆盖
        merged_adapter_config = {**DEFAULT_ADAPTER_CONFIG, **(adapter_config or {})}
        self.adapter = Adapter(**merged_adapter_config)
        # init 阶段从 .pt 加载 Adapter 权重（如 inference/utils/00300.pt）
        _adapter_pt = adapter_weights_path or default_adapter_weights_path()
        if Path(_adapter_pt).exists():
            load_adapter_weights(self.adapter, _adapter_pt)
            logger.info("Adapter 权重已加载: %s", _adapter_pt)
        elif adapter_weights_path is not None:
            load_adapter_weights(self.adapter, adapter_weights_path)
        else:
            logger.warning("未找到默认 Adapter 权重 %s，使用随机初始化", _adapter_pt)
        if not train_adapter:
            for p in self.adapter.parameters():
                p.requires_grad_(False)
            self.adapter.eval()

        # ========== 3. Vision Backbone（vision_truncation_2 或外部注入的 VLA vision_backbone，并截断）==========
        if external_vision_backbone is not None:
            self.vision_backbone = external_vision_backbone
            # 与 Align_adapter 一致：舍弃前 min_layer 层，保留第 min_layer 层及之后（min_layer = max_layer+1）
            min_layer = vision_backbone_max_layer + 1
            skip_vision_backbone_layers(self.vision_backbone, min_layer)
            # 打印截断后层数，便于核对
            if hasattr(self.vision_backbone, "featurizer") and hasattr(self.vision_backbone.featurizer, "blocks"):
                n_dino = len(self.vision_backbone.featurizer.blocks)
                logger.info("  DINO featurizer 截断后层数: %s (保留第 %s 层及之后)", n_dino, min_layer)
            if hasattr(self.vision_backbone, "fused_featurizer") and hasattr(self.vision_backbone.fused_featurizer, "blocks"):
                n_siglip = len(self.vision_backbone.fused_featurizer.blocks)
                logger.info("  SigLIP featurizer 截断后层数: %s (保留第 %s 层及之后)", n_siglip, min_layer)
            # 仅需 image_transform，用 min_layer=0 构建一次以获取 transform
            _, self.image_transform = build_posterior_vision_backbone(
                vision_backbone_id=vision_backbone_id,
                image_resize_strategy=image_resize_strategy,
                image_sequence_len=image_sequence_len,
                min_layer=0,
                freeze_backbone=True,
            )
            if freeze_vision_backbone:
                for p in self.vision_backbone.parameters():
                    p.requires_grad_(False)
                self.vision_backbone.eval()
            logger.info(
                "Vision backbone 使用外部注入（如 OpenVLA 提取），舍弃前 %s 层，保留第 %s 层及之后",
                min_layer,
                min_layer,
            )
        else:
            # 无外部 backbone 时：构建并做「保留 min_layer 及之后」截断（min_layer = max_layer+1）
            min_layer = vision_backbone_max_layer + 1
            self.vision_backbone, self.image_transform = build_posterior_vision_backbone(
                vision_backbone_id=vision_backbone_id,
                image_resize_strategy=image_resize_strategy,
                image_sequence_len=image_sequence_len,
                min_layer=min_layer,
                freeze_backbone=freeze_vision_backbone,
            )

        # ========== 4. 融合层 ==========
        dino_dim = getattr(self.vision_backbone, "dino_dim", 1024)
        siglip_dim = getattr(self.vision_backbone, "siglip_dim", 1152)
        fused_dim = siglip_dim + dino_dim + siglip_dim + dino_dim
        self.fusion = nn.Linear(fused_dim, fused_dim)
        # 实际送入 OpenVLA projector 的特征维度（与 CGIC 一致：DINO 在前 + SigLIP 在后）
        output_dim = dino_dim + siglip_dim
        logger.info(
            "CombinedVisionModel (HPCM y_hat + Adapter + Vision): 输出到 projector dim=%s (DINO %s + SigLIP %s)",
            output_dim,
            dino_dim,
            siglip_dim,
        )

    def forward(
        self,
        image: torch.Tensor,
        task_instruction: Optional[str] = None,
        image_pad_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向: HPCM(y_hat) + Adapter + Vision Truncated.

        Args:
            image: (B, C, H, W) 或 (B, N, C, H, W)，值域 [-1, 1]

        Returns:
            Dict: aligned_siglip_features, aligned_dino_features,
            vision_siglip_features, vision_dino_features, fused_features
        """
        return self._forward_with_hpcm(image, **kwargs)

    def _forward_with_hpcm(self, image: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        if image.dim() == 5:
            B, N, C, H, W = image.shape
            image = image.view(B * N, C, H, W)
            need_reshape = True
        else:
            B = image.shape[0]
            N = 1
            need_reshape = False

        # 1. HPCM 提取量化特征：输入需为 [0, 1]，且 FP32 以保证计算稳定性
        images_01 = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                y_hat = self.hpcm_encoder(images_01.float())
        if y_hat.shape[-2] != 16 or y_hat.shape[-1] != 16:
            y_hat = F.interpolate(y_hat, size=(16, 16), mode="bilinear", align_corners=False)
        device = y_hat.device

        # 2. Adapter 对齐特征空间（adapter 已转 float16，输入需匹配）
        align_siglip, align_dino = self.adapter(y_hat.to(dtype=torch.float16))

        # 3. 将 Adapter 特征直接喂给后半截 Vision Backbone（替代图像）
        # 兼容 DinoSigLIP (dino/siglip_featurizer) 与 HF PrismaticVisionBackbone
        # HF timm_model_ids=[DINO,SigLIP] => featurizer=DINO(1024), fused_featurizer=SigLIP(1152)
        if hasattr(self.vision_backbone, "dino_featurizer"):
            dino_feat = self.vision_backbone.dino_featurizer
            siglip_feat = self.vision_backbone.siglip_featurizer
        elif hasattr(self.vision_backbone, "fused_featurizer"):
            dino_feat = self.vision_backbone.featurizer         # DINO (1024)
            siglip_feat = self.vision_backbone.fused_featurizer  # SigLIP (1152)
        else:
            raise AttributeError(
                "vision_backbone must have dino_featurizer/siglip_featurizer or featurizer/fused_featurizer"
            )
        # Vision featurizer 来自 VLA，LayerNorm 对 fp16/float32 混用敏感。
        # 临时将 featurizer 转为 float32 运行，再转回原 dtype，输出转 float16 供 projector。
        def _run_featurizer_in_fp32(feat: nn.Module, x: torch.Tensor) -> torch.Tensor:
            orig_dtype = next(feat.parameters()).dtype
            feat.to(torch.float32)
            try:
                return feat(x.float())
            finally:
                feat.to(orig_dtype)

        with torch.cuda.amp.autocast(enabled=False):
            vision_dino = _run_featurizer_in_fp32(dino_feat, align_dino)
            vision_siglip = _run_featurizer_in_fp32(siglip_feat, align_siglip)
        vision_dino = vision_dino.to(dtype=torch.bfloat16)
        vision_siglip = vision_siglip.to(dtype=torch.bfloat16)

        # 4. 形状整理：恢复多图维度 (B, N, Sequence_Length, Dim)
        # 注意：这里千万不能写成 view(B, N, -1)，必须把 256 (Seq) 保留下来！
        if need_reshape:
            vision_dino = vision_dino.view(B, N, 256, vision_dino.shape[-1])
            vision_siglip = vision_siglip.view(B, N, 256, vision_siglip.shape[-1])
            
            align_dino = align_dino.view(B, N, 256, align_dino.shape[-1])
            align_siglip = align_siglip.view(B, N, 256, align_siglip.shape[-1])
        else:
            vision_dino = vision_dino.unsqueeze(1)
            vision_siglip = vision_siglip.unsqueeze(1)
            
            align_dino = align_dino.unsqueeze(1)
            align_siglip = align_siglip.unsqueeze(1)

        # 5. 最终特征融合：只拼接经过 posterior 提取后的特征
        # 顺序必须与 Align_adapter (CGIC) 一致：DINO 在前、SigLIP 在后，否则 OpenVLA projector 会错乱导致成功率为 0
        # 维度 1024 + 1152 = 2176，与 OpenVLA 的 Projector 匹配
        fused_features = torch.cat([vision_dino, vision_siglip], dim=-1)

        return {
            "aligned_siglip_features": align_siglip,
            "aligned_dino_features": align_dino,
            "vision_siglip_features": vision_siglip,
            "vision_dino_features": vision_dino,
            "fused_features": fused_features,
        }

    def _resize_images_to_224(self, images: torch.Tensor) -> torch.Tensor:
        """与 Align_adapter/inference 一致：letterbox + pad 到 224。"""
        B, C, H, W = images.shape
        target_size = 224
        if H == target_size and W == target_size:
            return images
        scale = min(target_size / H, target_size / W)
        new_h = int(H * scale)
        new_w = int(W * scale)
        resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_top > 0 or pad_left > 0:
            resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        return resized

    def load_adapter_from_checkpoint(self, checkpoint_path: Path) -> None:
        """从 checkpoint 加载 Adapter 权重（与 Align_adapter/inference 一致）。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        adapter_state = {
            k.replace("adapter.", ""): v
            for k, v in state_dict.items()
            if k.startswith("adapter.")
        }
        self.adapter.load_state_dict(adapter_state, strict=True)

    def load_vision_backbone_from_checkpoint(self, checkpoint_path: Path) -> None:
        """从 checkpoint 加载 Vision backbone 权重。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        vision_state = {
            k.replace("vision_backbone.", ""): v
            for k, v in state_dict.items()
            if k.startswith("vision_backbone.")
        }
        self.vision_backbone.load_state_dict(vision_state, strict=False)

    def to(self, device: torch.device) -> "CombinedVisionModel":
        super().to(device)
        self.device = device
        return self

    def eval(self) -> "CombinedVisionModel":
        super().eval()
        self.hpcm_encoder.eval()
        self.adapter.eval()
        self.vision_backbone.eval()
        return self

    def train(self, mode: bool = True) -> "CombinedVisionModel":
        super().train(mode)
        if mode:
            self.hpcm_encoder.eval()
            self.vision_backbone.eval()
        return self
