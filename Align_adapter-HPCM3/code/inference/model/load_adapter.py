"""
从 .pt 文件加载 Adapter 权重到 Adapter 模块。

.pt 可为 extract_adapter_weights.py 提取的纯 adapter state_dict（pre_neck.*, neck_siglip.*, neck_dino.*），
或 Lightning ckpt 风格（键为 adapter.*），本函数会自动处理。
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


def load_adapter_weights(
    adapter: nn.Module,
    weights_path: Union[str, Path],
    strict: bool = True,
    map_location: str = "cpu",
) -> None:
    """
    从 .pt 加载权重到 adapter 模块。

    Args:
        adapter: 推理用 Adapter 实例（与 model/adapter.py 结构一致）
        weights_path: .pt 路径，如 inference/utils/00300.pt；若为相对路径，相对于项目根或 inference/utils
        strict: 是否 strict=True load_state_dict
        map_location: torch.load 的 map_location
    """
    weights_path = Path(weights_path)
    if not weights_path.is_absolute():
        # 相对路径：先试 inference/utils 下，再当前工作目录
        _here = Path(__file__).resolve().parent
        _utils = _here.parent / "utils"
        for base in (_utils, _here.parent.parent, Path.cwd()):
            candidate = base / weights_path
            if candidate.exists():
                weights_path = candidate
                break
        else:
            weights_path = Path.cwd() / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Adapter 权重文件不存在: {weights_path}")

    state = torch.load(weights_path, map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # 若来自完整 ckpt，键为 adapter.pre_neck.*，去掉前缀
    if state and next(iter(state.keys())).startswith("adapter."):
        state = {k.replace("adapter.", "", 1): v for k, v in state.items()}
    adapter.load_state_dict(state, strict=strict)


def default_adapter_weights_path() -> Path:
    """默认路径：inference/utils/00300.pt（相对本文件所在 inference/model）。"""
    return Path(__file__).resolve().parent.parent / "utils" / "00300.pt"
