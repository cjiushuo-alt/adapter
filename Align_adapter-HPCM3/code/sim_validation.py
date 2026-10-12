"""Diagnostic-only Object simulator boundary-feature validation callback.

This callback never controls ModelCheckpoint or EarlyStopping; only the
trajectory-held-out RLDS clean val_loss selects a checkpoint.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

from dataset.dataset import object_pro_rollout_pixels


class SimulatorBoundaryValidation(pl.Callback):
    def __init__(self, manifest_path: str, checkpoint_dir: str, output_jsonl: str):
        self.manifest_path = Path(manifest_path)
        self.checkpoint_dir = str(checkpoint_dir)
        self.output_jsonl = Path(output_jsonl)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("source") != "fresh_env_reset_not_default_pruned_init" or manifest.get("protected_state_collisions") != 0:
            raise ValueError("Simulator validation manifest lacks protected-state exclusion")
        self.samples = []
        for row in manifest["samples"]:
            with Image.open(self.manifest_path.parent / row["path"]) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
            import hashlib
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
                raise ValueError(f"Simulator validation RGB changed: {row['path']}")
            raw = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
            teacher = object_pro_rollout_pixels(rgb, self.checkpoint_dir)
            self.samples.append((row, raw, teacher))
        if not self.samples:
            raise ValueError("Simulator validation must not be empty")
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        values = defaultdict(lambda: defaultdict(list))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for meta, raw, teacher in self.samples:
                result = pl_module.forward(
                    raw.unsqueeze(0).to(pl_module.device),
                    teacher_pixels=teacher.unsqueeze(0).to(pl_module.device),
                )
                dino = (result["aligned_dino_features"].float() -
                        result["vision_dino_features"].float()).square().mean().item()
                siglip = (result["aligned_siglip_features"].float() -
                          result["vision_siglip_features"].float()).square().mean().item()
                for view in (meta["view"], "all"):
                    values[view]["dino_mse"].append(dino)
                    values[view]["siglip_mse"].append(siglip)
                    values[view]["loss"].append(siglip + 5.0 * dino)
        summary = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "source": "fresh_reset_simulator_not_formal_eval_states",
            "selection_metric": False,
            "views": {
                view: {"n": len(metrics["loss"]), **{
                    name: float(np.mean(series)) for name, series in metrics.items()
                }} for view, metrics in values.items()
            },
        }
        with self.output_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary) + "\n")
        for view, metrics in summary["views"].items():
            for name in ("loss", "dino_mse", "siglip_mse"):
                pl_module.log(f"sim/{view}_{name}", metrics[name], on_step=False,
                              on_epoch=True, logger=True, prog_bar=False)
