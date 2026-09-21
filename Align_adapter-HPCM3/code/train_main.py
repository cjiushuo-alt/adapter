"""
Align Adapter (HPCM2) 训练主函数

图像 -> HPCM 编码器(实时) -> Adapter -> 对齐特征；仅需图像数据，无需预提取 .pt。
"""

import sys
from pathlib import Path

# 保证可 import model、dataset（运行目录为 Align_adapter-HPCM2 或项目根）
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytorch_lightning as pl
from omegaconf import OmegaConf
from argparse import ArgumentParser
import torch
import numpy as np
import os
import json
from typing import Optional

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from dataset.dataset import LightningDataModule
from model.model import AlignModel


class MetricsLogger(Callback):
    """训练/验证指标日志回调"""

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 50 != 0:
            return
        if hasattr(pl_module, "last_train_losses") and pl_module.last_train_losses:
            L = pl_module.last_train_losses
            print(
                f"Step {trainer.global_step:05d} | "
                f"Total: {L.get('total_loss', 0):.4f} | "
                f"SigLIP: {L.get('siglip_loss', 0):.4f} | "
                f"DINO: {L.get('dino_loss', 0):.4f}"
            )

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        total = m.get("train/total_loss")
        if total is not None:
            t = float(total) if isinstance(total, torch.Tensor) else total
            s = float(m.get("train/siglip_loss", 0) or 0) if isinstance(m.get("train/siglip_loss"), torch.Tensor) else (m.get("train/siglip_loss") or 0)
            d = float(m.get("train/dino_loss", 0) or 0) if isinstance(m.get("train/dino_loss"), torch.Tensor) else (m.get("train/dino_loss") or 0)
            print(f"\nEpoch {trainer.current_epoch:03d} 训练: Total={t:.4f}, SigLIP={s:.4f}, DINO={d:.4f}")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        total = m.get("val/total_loss")
        if total is not None:
            t = float(total) if isinstance(total, torch.Tensor) else total
            s = float(m.get("val/siglip_loss", 0) or 0) if isinstance(m.get("val/siglip_loss"), torch.Tensor) else (m.get("val/siglip_loss") or 0)
            d = float(m.get("val/dino_loss", 0) or 0) if isinstance(m.get("val/dino_loss"), torch.Tensor) else (m.get("val/dino_loss") or 0)
            print(f"\nEpoch {trainer.current_epoch:03d} 验证: Total={t:.4f}, SigLIP={s:.4f}, DINO={d:.4f}")


def resolve_ckpt_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root / p


def run_train(config):
    if "seed" in config:
        pl.seed_everything(config.seed, workers=True)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # 项目根 = VLA-Adapter（vision_ckpt 等路径相对于此）
    project_root = Path(__file__).parent.parent

    # ========== 1. 加载模型 ==========
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)

    model_cfg = config.model
    training_cfg = config.training
    hpcm_cfg = getattr(model_cfg, "hpcm", None) or getattr(model_cfg, "hpcm_encoder", None) or {}
    if hasattr(hpcm_cfg, "items"):
        hpcm_cfg = dict(hpcm_cfg)
    else:
        hpcm_cfg = OmegaConf.to_container(hpcm_cfg, resolve=True) if hpcm_cfg else {}
    vision_cfg = model_cfg.vision
    adapter_cfg = model_cfg.adapter

    vision_checkpoint = resolve_ckpt_path(vision_cfg.checkpoint, project_root)
    adapter_dict = OmegaConf.to_container(adapter_cfg, resolve=True) if adapter_cfg else None
    train_adapter = adapter_dict.pop("train_adapter", True) if adapter_dict else True

    model = AlignModel(
        hpcm_root=hpcm_cfg.get("hpcm_root", ""),
        hpcm_checkpoint=hpcm_cfg.get("checkpoint") or None,
        hpcm_model_name=hpcm_cfg.get("model_name", "HPCM_Base"),
        hpcm_scale_table_levels=hpcm_cfg.get("scale_table_levels", 60),
        freeze_hpcm_encoder=hpcm_cfg.get("freeze", True),
        vision_backbone_id=vision_cfg.backbone_id,
        vision_backbone_checkpoint=vision_checkpoint,
        vision_backbone_max_layer=getattr(vision_cfg, "max_layer", 2),
        image_resize_strategy=vision_cfg.image_resize_strategy,
        image_sequence_len=vision_cfg.image_sequence_len,
        adapter_config=adapter_dict,
        freeze_vision_backbone=vision_cfg.freeze_backbone,
        train_adapter=train_adapter,
        loss_config=dict(config.loss) if hasattr(config, "loss") else None,
        optimizer_config=dict(training_cfg.optimizer) if hasattr(training_cfg, "optimizer") else None,
        lr_scheduler_config=dict(training_cfg.lr_scheduler) if hasattr(training_cfg, "lr_scheduler") else None,
        learning_rate=training_cfg.optimizer.get("lr") if hasattr(training_cfg, "optimizer") else None,
    )

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {n_total:,}, 可训练: {n_train:,}, Adapter: {sum(p.numel() for p in model.adapter.parameters()):,}")

    # ========== 2. 数据（仅图像，HPCM 在模型内实时提取）==========

    print("\n" + "=" * 60)
    print("加载数据（仅图像）...")
    print("=" * 60)
    data_module = LightningDataModule(OmegaConf.to_container(config.data, resolve=True))
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    # ========== 3. Callbacks ==========
    ckpt_cfg = training_cfg.checkpoint
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch-{epoch:02d}-step-{step:05d}-{val_loss:.4f}",
            monitor=ckpt_cfg.get("monitor", "val_loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            save_last=ckpt_cfg.get("save_last", True),
            save_weights_only=False,
        )
    callbacks = [
        best_checkpoint,
        LearningRateMonitor(logging_interval="step"),
        MetricsLogger(),
    ]
    early_cfg = training_cfg.get("early_stopping", {})
    if early_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg.get("monitor", ckpt_cfg.get("monitor", "val_loss")),
                mode=early_cfg.get("mode", ckpt_cfg.get("mode", "min")),
                min_delta=early_cfg.get("min_delta", 0.0),
                patience=early_cfg.get("patience", 5),
                check_finite=early_cfg.get("check_finite", True),
                verbose=True,
            )
        )
    if ckpt_cfg.get("train_every_n_steps"):
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="train-step-{step:05d}",
                every_n_train_steps=ckpt_cfg.train_every_n_steps,
                save_top_k=-1,
                save_weights_only=False,
            )
        )

    # ========== 4. Resume ==========
    resume_cfg = training_cfg.get("resume", {})
    resume_enabled = resume_cfg.get("enabled", False)
    resume_ckpt_path: Optional[Path] = None
    resume_strategy = None
    adapter_resume_path: Optional[Path] = None
    adapter_path_cfg = resume_cfg.get("adapter_only_path") or config.get("resume_from_checkpoint")

    if resume_enabled and resume_cfg.get("state_ckpt_path"):
        resume_ckpt_path = resolve_ckpt_path(resume_cfg.state_ckpt_path, project_root)
        if resume_ckpt_path.exists():
            resume_strategy = "state"
            print(f"将从完整 checkpoint 恢复: {resume_ckpt_path}")
    if adapter_path_cfg:
        adapter_resume_path = resolve_ckpt_path(adapter_path_cfg, project_root)
        if adapter_resume_path.exists() and resume_strategy != "state":
            resume_strategy = "adapter"
            ckpt = torch.load(adapter_resume_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("adapter_state_dict", ckpt)
            model.adapter.load_state_dict(sd, strict=resume_cfg.get("strict", True))
            print(f"已加载 Adapter 权重: {adapter_resume_path}（新 run，step 从 0 开始）")

    # ========== 5. Logger & Trainer ==========
    wb_cfg = config.get("wandb", {})
    wb_kw = {
        "project": wb_cfg.get("project", "align-adapter-hpcm2"),
        "name": wb_cfg.get("experiment_name", "hpcm2_train"),
        "save_dir": str(output_dir),
    }
    if wb_cfg.get("resume", {}).get("enabled") and wb_cfg["resume"].get("run_id"):
        wb_kw["id"] = wb_cfg["resume"]["run_id"]
        wb_kw["resume"] = wb_cfg["resume"].get("resume_mode", "allow")
    else:
        wb_kw["resume"] = False

    wandb_logger = WandbLogger(**wb_kw)

    val_interval = training_cfg.get("validation", {}).get("every_n_steps")
    trainer_kwargs = dict(
        max_epochs=training_cfg.max_epochs,
        max_steps=training_cfg.get("max_steps"),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=training_cfg.get("gradient_clip_algorithm", "norm"),
        precision=training_cfg.get("precision", "16"),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 50),
        enable_progress_bar=training_cfg.get("enable_progress_bar", True),
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=config.get("device", {}).get("accelerator", "gpu"),
        devices=config.get("device", {}).get("devices", 1),
        strategy=config.get("device", {}).get("strategy", "auto"),
    )
    if val_interval is None:
        trainer_kwargs["check_val_every_n_epoch"] = training_cfg.get("validation", {}).get("every_n_epochs", 1)
    else:
        trainer_kwargs["val_check_interval"] = val_interval
        trainer_kwargs["check_val_every_n_epoch"] = None
    trainer = pl.Trainer(**trainer_kwargs)

    # ========== 6. 训练 ==========
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    fit_kw = {}
    if resume_strategy == "state" and resume_ckpt_path and resume_ckpt_path.exists():
        fit_kw["ckpt_path"] = str(resume_ckpt_path)

    trainer.fit(model, datamodule=data_module, **fit_kw)

    if not best_checkpoint.best_model_path:
        raise RuntimeError("训练结束后未找到 best checkpoint")
    best = torch.load(best_checkpoint.best_model_path, map_location="cpu", weights_only=False)
    state_dict = best.get("state_dict", best)
    adapter_state = {
        key.replace("adapter.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("adapter.")
    }
    if not adapter_state:
        raise RuntimeError(f"best checkpoint 中没有 adapter 权重: {best_checkpoint.best_model_path}")
    adapter_output = output_dir / "adapter-best.pt"
    torch.save(adapter_state, adapter_output)
    summary = {
        "best_checkpoint": best_checkpoint.best_model_path,
        "best_val_loss": float(best_checkpoint.best_model_score),
        "adapter_checkpoint": str(adapter_output),
        "stopped_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "wandb_run_id": getattr(wandb_logger.experiment, "id", None),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\n训练完成，best adapter: {adapter_output}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Align Adapter (HPCM2) 训练：仅图像，HPCM 实时编码")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_MODE"] = args.wandb_mode

    config = OmegaConf.load(args.config)
    run_train(config)
