"""
HPCM 特征提取：完整模型前向 + hook 截取 decoder 输入

- 不是只跑 HPCM 的“编码器”子图，而是把整个 HPCM 模型跑一遍（g_a → 量化 → … → g_s）。
- 通过 decoder g_s 的 forward hook 截取「输入到 g_s 的 y_hat」（即量化后的 latent）作为返回值。
- 输入：图像 (B, 3, H, W)，[0, 1]
- 输出：y_hat (B, 320, H', W')，即送入 decoder 前的量化表示。

参考：HPCM/base_show.py、HPCM/src/models/HPCM_Base.py
"""

import math
import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_hpcm_in_path(hpcm_root: str) -> None:
    hpcm_root = os.path.abspath(hpcm_root)
    if hpcm_root not in sys.path:
        sys.path.insert(0, hpcm_root)


def _get_scale_table(min_val: float = 0.12, max_val: float = 64.0, levels: int = 60) -> torch.Tensor:
    """与 HPCM/test.py、base_show.py 一致"""
    return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))


def pad_to_multiple(x: torch.Tensor, p: int = 256) -> torch.Tensor:
    """将特征图 pad 为 H、W 均为 p 的整数倍（与 HPCM/test.py 一致）"""
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)


def crop_to_size(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """从中心裁回指定 (h, w)（与 HPCM/test.py 一致）"""
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom), mode="constant", value=0)


class HPCMEncoder(nn.Module):
    """
    跑完整 HPCM 模型，用 g_s(decoder) 的 forward hook 截取「decoder 的输入 y_hat」作为输出。

    - 内部：整图 pad 为 256 倍数 → 完整 HPCM 前向 model(x) → 用 hook 取 g_s 的输入
    - 输入: image (B, 3, H, W)，[0, 1]
    - 输出: y_hat (B, 320, H', W')，即送入 decoder 前的量化 latent（H'=h_pad/16, W'=w_pad/16）
    """

    def __init__(
        self,
        hpcm_root: str,
        checkpoint_path: str,
        model_name: str = "HPCM_Base",
        scale_table_levels: int = 60,
        scale_table_min: float = 0.12,
        scale_table_max: float = 64.0,
        pad_multiple: int = 256,
    ):
        """
        Args:
            hpcm_root: HPCM 项目根目录（包含 src/、ckpt/ 等）
            checkpoint_path: 权重路径，如 ckpt/0.0018.pth.tar
            model_name: 模型模块名，如 HPCM_Base（对应 src.models.HPCM_Base）
            scale_table_levels: update(scale_table) 的 levels
            scale_table_min / scale_table_max: scale_table 范围
            pad_multiple: 输入 pad 的倍数（与 test 一致用 256）
        """
        super().__init__()
        self.hpcm_root = os.path.abspath(hpcm_root)
        self.checkpoint_path = (
            os.path.join(self.hpcm_root, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
        )
        self.model_name = model_name
        self.pad_multiple = pad_multiple
        self._scale_table_levels = scale_table_levels
        self._scale_table_min = scale_table_min
        self._scale_table_max = scale_table_max

        _ensure_hpcm_in_path(self.hpcm_root)
        # 与 CGIC 一致：在 __init__ 里立即构建并加载权重，恢复训练时可直接从 ckpt 恢复 HPCM
        self._hpcm_model: Optional[nn.Module] = None
        self._build_model(torch.device("cpu"))

    def _build_model(self, device: torch.device) -> nn.Module:
        if self._hpcm_model is not None:
            return self._hpcm_model.to(device)
        import importlib
        mod = importlib.import_module(f".{self.model_name}", "src.models")
        model = mod.HPCM()
        raw = torch.load(self.checkpoint_path, map_location=device)
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        model.load_state_dict(state_dict, strict=False)
        scale_table = _get_scale_table(
            self._scale_table_min,
            self._scale_table_max,
            self._scale_table_levels,
        )
        model.update(scale_table)
        model = model.to(device)
        model.eval()
        self._hpcm_model = model
        return model

    def forward(
        self,
        image: torch.Tensor,
        return_padded_shape: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int, int, int]]]:
        """
        Args:
            image: (B, 3, H, W)，[0, 1]
            return_padded_shape: 若 True，额外返回 (h_orig, w_orig, h_pad, w_pad) 便于后续 crop

        Returns:
            y_hat: (B, 320, H', W')，H'=h_pad/16, W'=w_pad/16
            若 return_padded_shape 为 True，则返回 (y_hat, (h_orig, w_orig, h_pad, w_pad))
        """
        device = image.device
        B, _, h_orig, w_orig = image.shape
        x = pad_to_multiple(image, self.pad_multiple)
        h_pad, w_pad = x.size(2), x.size(3)

        model = self._build_model(device)
        y_hat_list: list = []

        def hook_fn(_module, inp, _out):
            y_hat_list.append(inp[0].detach())

        # 完整 HPCM 前向；hook 在 decoder g_s 的输入处截取 y_hat（量化后、送入 g_s 前的 latent）
        handle = model.g_s.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = model(x, training=False)
        finally:
            handle.remove()

        y_hat = y_hat_list[0]
        if return_padded_shape:
            return y_hat, (h_orig, w_orig, h_pad, w_pad)
        return y_hat

    def get_output_spatial_size(self, h: int, w: int) -> Tuple[int, int]:
        """根据输入高宽返回 y_hat 的空间尺寸 (H', W')"""
        H = (h + self.pad_multiple - 1) // self.pad_multiple * self.pad_multiple
        W = (w + self.pad_multiple - 1) // self.pad_multiple * self.pad_multiple
        return H // 16, W // 16


if __name__ == "__main__":
    import argparse
    from torchvision.transforms import ToTensor
    from PIL import Image

    parser = argparse.ArgumentParser(description="HPCM Encoder: image -> y_hat")
    parser.add_argument("--hpcm_root", type=str, default=None, help="HPCM 项目根目录，默认使用 ../HPCM")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint 路径，默认 hpcm_root/ckpt/0.0018.pth.tar")
    parser.add_argument("--image", type=str, default=None, help="测试图片路径（可选）")
    args = parser.parse_args()

    # 默认 hpcm_root 为 Align_adapter_HPCM 同级目录下的 HPCM
    this_dir = os.path.dirname(os.path.abspath(__file__))
    align_hpcm_root = os.path.dirname(this_dir)  # Align_adapter_HPCM
    default_hpcm_root = os.path.join(os.path.dirname(align_hpcm_root), "HPCM")
    hpcm_root = args.hpcm_root or default_hpcm_root
    ckpt = args.ckpt or os.path.join(hpcm_root, "ckpt", "0.0018.pth.tar")

    print("HPCM root:", hpcm_root)
    print("Checkpoint:", ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = HPCMEncoder(
        hpcm_root=hpcm_root,
        checkpoint_path=ckpt,
        model_name="HPCM_Base",
        scale_table_levels=60,
        pad_multiple=256,
    ).to(device)
    encoder.eval()

    if args.image and os.path.isfile(args.image):
        img = Image.open(args.image).convert("RGB")
        x = ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y_hat = encoder(x)
        print(f"输入图像: {x.shape}")
        print(f"y_hat: {y_hat.shape}")
    else:
        # 随机输入
        x = torch.rand(2, 3, 256, 256, device=device)
        with torch.no_grad():
            y_hat = encoder(x)
        print(f"随机输入: {x.shape}")
        print(f"y_hat: {y_hat.shape}")
        assert y_hat.shape == (2, 320, 16, 16), f"期望 (2, 320, 16, 16), 得到 {y_hat.shape}"
        print("✓ 输出形状 (2, 320, 16, 16) 符合预期")
    print("done.")
