"""
HPCM3 推理：用 VisionBackboneWrapper 包装 CombinedVisionModel，替换 VLA 的 vision_backbone，
直接返回 vla_model，由官方 predict_action / processor 处理。Adapter 使用 ResNet PreNeck。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.robot.openvla_utils import get_vla
from .combined_vision_model import CombinedVisionModel

logger = logging.getLogger(__name__)


class VisionBackboneWrapper(nn.Module):
    """
    将 CombinedVisionModel 包装成与 Prismatic/OpenVLA vision_backbone 同接口的模块。
    输入：pixel_values (B, 6*num_images, H, W)，与 OpenVLA fused backbone 一致。
    输出：(B, num_images*256, embed_dim)，供 VLA 后续使用。
    """

    def __init__(self, combined_model: CombinedVisionModel, original_backbone: nn.Module):
        super().__init__()
        self.model = combined_model
        self.embed_dim = getattr(original_backbone, "embed_dim", 2176)
        self.default_image_resolution = getattr(
            original_backbone, "default_image_resolution", (3, 224, 224)
        )
        if hasattr(original_backbone, "dtype"):
            self.dtype = original_backbone.dtype
        self.identifier = getattr(
            original_backbone, "identifier", "hpcm3-combined-vision"
        )
        self.image_transform = getattr(
            original_backbone, "get_image_transform", lambda: None
        )()
        if self.image_transform is None and hasattr(original_backbone, "image_transform"):
            self.image_transform = original_backbone.image_transform

        self.num_images_in_input = getattr(
            original_backbone, "num_images_in_input",
            getattr(original_backbone, "get_num_images_in_input", lambda: 1)(),
        )
        if callable(self.num_images_in_input):
            self.num_images_in_input = 1
        self._num_patches = 256

    def get_image_transform(self):
        return self.image_transform

    def get_fsdp_wrapping_policy(self):
        return lambda x: False

    def get_num_patches(self) -> int:
        return self._num_patches

    def get_num_images_in_input(self) -> int:
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images: int):
        self.num_images_in_input = num_images
        if hasattr(self.model, "set_num_images_in_input"):
            self.model.set_num_images_in_input(num_images)

    def forward(self, pixel_values: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        pixel_values: (B, 6*num_images, H, W)，与 OpenVLA fused backbone 一致（前3 ch DINO，后3 ch SigLIP）。
        与 Align_adapter (CGIC) 一致：取 SigLIP 通道，resize 到 256x256 再送入 CombinedVisionModel，返回 (B, num_images*256, embed_dim)。
        """
        B, C, H, W = pixel_values.shape
        channels_per_image = 6
        num_images = C // channels_per_image
        x_reshaped = pixel_values.view(B, num_images, channels_per_image, H, W)
        x_rgb = x_reshaped[:, :, 3:, :, :]
        x_flat = x_rgb.reshape(B * num_images, 3, H, W)
        # 与 CGIC 一致：resize 到 256x256，避免 HPCM/Adapter 输入分辨率与训练不一致导致成功率为 0
        if H != 256 or W != 256:
            x_flat = torch.nn.functional.interpolate(
                x_flat, size=(256, 256), mode="bicubic", align_corners=False
            )

        with torch.no_grad():
            results = self.model(image=x_flat)
        fused = results["fused_features"]
        if fused.dim() == 4 and fused.shape[1] == 1:
            fused = fused.squeeze(1)
        _, N, D = fused.shape
        output = fused.view(B, num_images * N, D)
        return output.to(dtype=torch.bfloat16)  # VLA projector 为 bfloat16，输出需匹配


def get_custom_vla_model(
    openvla_path: Union[str, Path],
    *,
    vision_backbone_id: str = "dinosiglip-vit-so-224px",
    vision_backbone_checkpoint: Optional[Union[str, Path]] = None,
    vision_backbone_max_layer: int = 2,
    hpcm_root: Optional[str] = None,
    hpcm_checkpoint: Optional[str] = None,
    adapter_weights_path: Optional[Union[str, Path]] = None,
    adapter_config: Optional[Dict[str, Any]] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_film: bool = False,
    num_images_in_input: int = 1,
    device: Union[str, torch.device] = "cuda",
) -> nn.Module:
    """
    加载 OpenVLA，用 VisionBackboneWrapper(CombinedVisionModel) 替换 vision_backbone，直接返回 vla_model。
    文本与 predict_action 由官方 processor / predict_action 处理。
    """
    cfg = SimpleNamespace(
        pretrained_checkpoint=str(openvla_path),
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        use_film=use_film,
        num_images_in_input=num_images_in_input,
        preserve_checkpoint_files=True,
    )
    logger.info("Loading OpenVLA from %s ...", openvla_path)
    vla_model = get_vla(cfg)
    original_backbone = vla_model.vision_backbone

    hpcm_root = hpcm_root or str(_PROJECT_ROOT / "HPCM")
    if hpcm_checkpoint and not Path(hpcm_checkpoint).is_absolute():
        hpcm_checkpoint = str(Path(hpcm_root) / hpcm_checkpoint)

    combined_vision_model = CombinedVisionModel(
        vision_backbone_id=vision_backbone_id,
        vision_backbone_checkpoint=Path(vision_backbone_checkpoint) if vision_backbone_checkpoint else None,
        vision_backbone_max_layer=vision_backbone_max_layer,
        image_resize_strategy="letterbox",
        image_sequence_len=1,
        device=torch.device(device) if isinstance(device, str) else device,
        hpcm_root=hpcm_root,
        hpcm_checkpoint=hpcm_checkpoint or None,
        adapter_weights_path=str(adapter_weights_path) if adapter_weights_path else None,
        adapter_config=adapter_config,
        external_vision_backbone=original_backbone,
        freeze_vision_backbone=True,
        train_adapter=False,
    )
    combined_vision_model = combined_vision_model.to(device=device)

    def convert_to_float16(module: nn.Module, skip_hpcm: bool = True):
        for name, param in module.named_parameters():
            if skip_hpcm and "hpcm_encoder" in name:
                continue
            if param.dtype != torch.float16 and param.is_floating_point():
                param.data = param.data.to(dtype=torch.float16)
        for name, buf in module.named_buffers():
            if skip_hpcm and "hpcm_encoder" in name:
                continue
            if buf.is_floating_point() and buf.dtype != torch.float16:
                buf.data = buf.data.to(dtype=torch.float16)
    convert_to_float16(combined_vision_model)

    wrapper = VisionBackboneWrapper(combined_vision_model, original_backbone)
    vla_model.vision_backbone = wrapper
    wrapper.set_num_images_in_input(num_images_in_input)
    if not load_in_8bit and not load_in_4bit:
        vla_model = vla_model.to(device)
    logger.info("HPCM2 vision_backbone replaced; returning vla_model.")
    return vla_model
