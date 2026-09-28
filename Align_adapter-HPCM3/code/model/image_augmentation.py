"""GPU image augmentation compatible with the VLA-Adapter RLDS recipe."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as tvf


class VLAImageAugmentation(nn.Module):
    """Apply the VLA crop and photometric recipe once to a batched RGB tensor.

    Inputs and outputs use the HPCM training range ``[-1, 1]``.  The same
    returned tensor must be shared by the HPCM student and vision teacher.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.crop_scale = float(cfg.get("crop_scale", 0.9))
        self.brightness_delta = float(cfg.get("brightness_delta", 0.2))
        self.contrast_min = float(cfg.get("contrast_min", 0.8))
        self.contrast_max = float(cfg.get("contrast_max", 1.2))
        self.saturation_min = float(cfg.get("saturation_min", 0.8))
        self.saturation_max = float(cfg.get("saturation_max", 1.2))
        self.hue_delta = float(cfg.get("hue_delta", 0.05))
        self.validation_enabled = bool(cfg.get("validation_enabled", True))
        self.validation_seed = int(cfg.get("validation_seed", 4242))

        if not 0.0 < self.crop_scale <= 1.0:
            raise ValueError(f"crop_scale 必须在 (0, 1]，得到 {self.crop_scale}")
        if not 0.0 <= self.hue_delta <= 0.5:
            raise ValueError(f"hue_delta 必须在 [0, 0.5]，得到 {self.hue_delta}")

    @staticmethod
    def _uniform(
        batch: int,
        low: float,
        high: float,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.rand(batch, device=device, generator=generator) * (high - low) + low

    def _random_resized_crop(
        self, images: torch.Tensor, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        batch = images.shape[0]
        side = math.sqrt(self.crop_scale)
        max_offset = 1.0 - side
        top = self._uniform(batch, 0.0, max_offset, images.device, generator)
        left = self._uniform(batch, 0.0, max_offset, images.device, generator)

        # affine_grid uses coordinates in [-1, 1].  This is the vectorized
        # equivalent of crop-and-resize with a square normalized crop box.
        theta = torch.zeros(batch, 2, 3, device=images.device, dtype=images.dtype)
        theta[:, 0, 0] = side
        theta[:, 1, 1] = side
        theta[:, 0, 2] = 2.0 * left + side - 1.0
        theta[:, 1, 2] = 2.0 * top + side - 1.0
        grid = F.affine_grid(theta, images.shape, align_corners=True)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, images: torch.Tensor, *, seed: Optional[int] = None) -> torch.Tensor:
        if not self.enabled:
            return images
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"期望 RGB batch (B,3,H,W)，得到 {tuple(images.shape)}")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=images.device)
            generator.manual_seed(int(seed))

        output = images.float().add(1.0).mul(0.5).clamp_(0.0, 1.0)
        batch = output.shape[0]

        output = self._random_resized_crop(output, generator).clamp_(0.0, 1.0)

        brightness = self._uniform(
            batch, -self.brightness_delta, self.brightness_delta, output.device, generator
        ).view(batch, 1, 1, 1)
        output = output.add(brightness).clamp_(0.0, 1.0)

        contrast = self._uniform(
            batch, self.contrast_min, self.contrast_max, output.device, generator
        ).view(batch, 1, 1, 1)
        channel_means = output.mean(dim=(-2, -1), keepdim=True)
        output = ((output - channel_means) * contrast + channel_means).clamp_(0.0, 1.0)

        saturation = self._uniform(
            batch, self.saturation_min, self.saturation_max, output.device, generator
        )
        hue = self._uniform(batch, -self.hue_delta, self.hue_delta, output.device, generator)
        adjusted = []
        for index in range(batch):
            sample = tvf.adjust_saturation(output[index], float(saturation[index]))
            sample = tvf.adjust_hue(sample, float(hue[index]))
            adjusted.append(sample.clamp_(0.0, 1.0))
        output = torch.stack(adjusted)

        return output.mul(2.0).sub(1.0)
