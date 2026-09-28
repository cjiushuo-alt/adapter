"""
组合模型：完整 HPCM(含 hook 取 decoder 输入) + Vision Backbone + Align Adapter

与 Align_adapter 类似，用 HPCM 替代 CGIC：图像 -> 完整 HPCM 前向(hook 截取 decoder 输入 y_hat)
-> Adapter -> 对齐特征；图像 -> Vision Backbone -> vision 特征。HPCM 部分见 dataset/hpcm_encoder.py。
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import sys
import yaml

# 项目路径：VLA-Adapter（Align_adapter）、Align_adapter-HPCM2（dataset）
project_root = Path(__file__).parent.parent.parent
_hpcm2_root = Path(__file__).parent.parent
for _p in (project_root, _hpcm2_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from .adapter import Adapter
from .image_augmentation import VLAImageAugmentation
from .vision_truncation import build_truncated_vision_backbone
from Align_adapter.tools.losses import CombinedDistLoss
from dataset.hpcm_encoder import HPCMEncoder

class AlignModel(pl.LightningModule):
    """
    组合模型：完整 HPCM(hook 取 decoder 输入) + Vision Backbone + Align Adapter

    架构（与 Align_adapter 类似，CGIC 换为 HPCM）：
    1. HPCM: 图像 -> 完整模型前向 -> hook 截取 decoder 输入 y_hat (B, 320, H', W')，冻结
    2. Vision Backbone: 图像 -> vision 特征，冻结
    3. Align Adapter: y_hat -> 对齐特征，可训练
    """

    def __init__(
        self,
        # HPCM Encoder 配置
        hpcm_root: str = "",
        hpcm_checkpoint: Optional[str] = None,
        hpcm_model_name: str = "HPCM_Base",
        hpcm_scale_table_levels: int = 60,
        freeze_hpcm_encoder: bool = True,
        # Vision Backbone 配置
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        vision_backbone_checkpoint: Optional[Path] = None,
        vision_backbone_max_layer: Optional[int] = 2,
        image_resize_strategy: str = "letterbox",
        image_sequence_len: int = 1,
        # Adapter 配置
        adapter_config: Optional[Dict[str, Any]] = None,
        # 训练配置
        freeze_vision_backbone: bool = True,
        train_adapter: bool = True,
        # 损失与优化
        loss_config: Optional[Dict[str, Any]] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        learning_rate: Optional[float] = None,
    ):
        super().__init__()
        self.last_train_losses = {}

        # ========== 1. HPCM：完整模型前向 + hook 取 decoder 输入 y_hat ==========
        print("=" * 80)
        print("初始化 HPCM（完整模型，hook 截取 decoder 输入）")
        print("=" * 80)
        _hpcm_root = Path(hpcm_root) if hpcm_root else (_hpcm2_root.parent / "HPCM")
        hpcm_ckpt = hpcm_checkpoint or str(_hpcm_root / "ckpt" / "0.0018.pth.tar")
        if hpcm_checkpoint and not Path(hpcm_checkpoint).is_absolute():
            hpcm_ckpt = str(_hpcm_root / hpcm_checkpoint)
        # 与 CGIC 一致：在 __init__ 里即构建并加载 HPCM 权重，恢复训练时可直接从 ckpt 恢复
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
            print("✓ HPCM（完整模型）已冻结")

        # ========== 2. Vision Backbone ==========
        print("\n" + "=" * 80)
        print("初始化 Vision Backbone")
        print("=" * 80)
        checkpoint_path = None
        if vision_backbone_checkpoint is not None:
            checkpoint_path = Path(vision_backbone_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root / vision_backbone_checkpoint
        self.vision_backbone, self.image_transform = build_truncated_vision_backbone(
            vision_backbone_id=vision_backbone_id,
            image_resize_strategy=image_resize_strategy,
            image_sequence_len=image_sequence_len,
            max_layer=vision_backbone_max_layer,
            partial_checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_vision_backbone,
        )

        # ========== 3. Align Adapter ==========
        print("\n" + "=" * 80)
        print("初始化 Align Adapter")
        print("=" * 80)
        
        if adapter_config is None:
            # 默认配置（HPCM3：PreNeck 为 ResNet 版；(B, 320, 16, 16) -> 序列 256；输出维与 DinoSigLIP 一致）
            adapter_config = {
                "hpcm_input_dim": 320,
                "pre_neck_output_dim": 320,
                "pre_neck_spatial_size": 16,
                "pre_neck_num_res_blocks": 2,
                "pre_neck_dropout": 0.0,
                "siglip_dim": 1152,  # 与 vision backbone siglip_featurizer.embed_dim 一致
                "dino_dim": 1024,    # 与 vision backbone dino_featurizer.embed_dim 一致
                "neck_num_heads": 8,
                "neck_ffn_hidden_dim": None,
                "neck_dropout": 0.0,
                "neck_activation": "gelu",
            }
        
        self.adapter = Adapter(**adapter_config)
        
        # 设置Adapter训练状态
        if train_adapter:
            for param in self.adapter.parameters():
                param.requires_grad_(True)
            self.adapter.train()
            print("✓ Adapter设置为可训练")
        else:
            for param in self.adapter.parameters():
                param.requires_grad_(False)
            self.adapter.eval()
            print("✓ Adapter已冻结")
        
        # ========== 4. 损失函数 ==========
        if loss_config is None:
            loss_config = {
                "type": "dist",
                "siglip_loss_weight": 1.0,
                "dino_loss_weight": 1.0,
                "reduction": "mean",
            }

        self.loss_scope = loss_config.get("scope", "boundary")
        if self.loss_scope not in {"boundary", "full_visual"}:
            raise ValueError(f"Unsupported loss scope: {self.loss_scope}")
        self.posterior_start_layer = int(loss_config.get("posterior_start_layer", 3))
        self.posterior_gradient_checkpointing = bool(
            loss_config.get("posterior_gradient_checkpointing", False)
        )
        if self.loss_scope == "full_visual" and vision_backbone_max_layer is not None:
            raise ValueError("full_visual loss 必须设置 model.vision.max_layer=null，加载完整视觉塔")
        self.image_augmentation = VLAImageAugmentation(augmentation_config)
        
        if loss_config.get("type", "dist") == "dist":
            self.loss_fn = CombinedDistLoss(
                siglip_loss_weight=loss_config.get("siglip_loss_weight", 1.0),
                dino_loss_weight=loss_config.get("dino_loss_weight", 1.0),
                reduction=loss_config.get("reduction", "mean"),
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_config.get('type')}")
        
        # ========== 5. 优化器和学习率配置 ==========
        # 保存配置用于configure_optimizers
        self.optimizer_config = optimizer_config or {}
        self.lr_scheduler_config = lr_scheduler_config or {}
        self.learning_rate = learning_rate or self.optimizer_config.get("lr", 1.0e-4)
        
        # 保存配置
        self.vision_backbone_id = vision_backbone_id
        self.freeze_hpcm_encoder = freeze_hpcm_encoder
        self.freeze_vision_backbone = freeze_vision_backbone
        self.train_adapter = train_adapter

        print(
            f"✓ Loss scope: {self.loss_scope}; posterior start layer: "
            f"{self.posterior_start_layer}; gradient checkpointing: "
            f"{self.posterior_gradient_checkpointing}"
        )
        print(f"✓ VLA-compatible train augmentation: {self.image_augmentation.enabled}")
        
        print("\n" + "=" * 80)
        print("✓ 模型初始化完成")
        print("=" * 80)

    def train(self, mode: bool = True):
        """
        重写train方法，确保冻结的模块始终保持eval模式
        这是解决训练速度慢和GPU使用率低的关键！
        默认的model.train()会将所有子模块设为train模式。
        """
        super().train(mode)
        
        # 如果处于训练模式，强制将冻结的模块设置回 eval 模式
        if mode:
            if self.freeze_hpcm_encoder:
                self.hpcm_encoder.eval()
            if self.freeze_vision_backbone:
                self.vision_backbone.eval()
            if not self.train_adapter:
                self.adapter.eval()
        return self

    def extract_hpcm_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        从图像提取 y_hat：跑完整 HPCM 前向，用 hook 截取 decoder 的输入。
        输入需为 [0,1] 或 [-1,1]（内部会转为 [0,1]）。

        Args:
            images: (B, C, H, W) 或 (B, N, C, H, W)

        Returns:
            y_hat: (B, 320, 16, 16) 或 (B*N, 320, 16, 16)，与 adapter 输入一致
        """
        if len(images.shape) == 5:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
        # 转为 [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        # HPCM 内部用 float32 做 index put，混合精度下需在 float32 下运行避免 Half/Float 不一致
        images = images.float()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                y_hat = self.hpcm_encoder(images)
        # Adapter 默认期望 (B, 320, 16, 16)，若非 16x16 则插值
        if y_hat.shape[-2] != 16 or y_hat.shape[-1] != 16:
            y_hat = torch.nn.functional.interpolate(y_hat, size=(16, 16), mode="bilinear", align_corners=False)
        return y_hat
    
    def _prepare_vision_inputs(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim == 5:
            batch, views, channels, height, width = images.shape
            images = images.view(batch * views, channels, height, width)
        if images.shape[-2:] != (224, 224):
            images = torch.nn.functional.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )
        images_01 = images.add(1.0).mul(0.5)
        dino_mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        dino_std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return {
            "dino": (images_01 - dino_mean) / dino_std,
            "siglip": images_01.mul(2.0).sub(1.0),
        }

    def _run_frozen_posterior(self, featurizer: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        """Run rollout-equivalent blocks 3..penultimate with gradients to tokens."""
        blocks = list(featurizer.blocks)
        if self.posterior_start_layer >= len(blocks) - 1:
            raise ValueError(
                f"posterior_start_layer={self.posterior_start_layer} 与 {len(blocks)} 层视觉塔不兼容"
            )
        output = tokens
        # Prismatic uses the second-to-last full-tower block as its visual
        # output.  The final block is intentionally excluded, matching rollout.
        for block in blocks[self.posterior_start_layer : -1]:
            if self.training and self.posterior_gradient_checkpointing and output.requires_grad:
                output = activation_checkpoint(block, output, use_reentrant=False)
            else:
                output = block(output)
        return output

    def forward(
        self,
        images: torch.Tensor,
        extract_hpcm_features: bool = True,
        extract_vision_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Extract boundary or full-visual student/teacher feature pairs."""
        results: Dict[str, torch.Tensor] = {}

        boundary_siglip = boundary_dino = None
        if extract_hpcm_features:
            hpcm_features = self.extract_hpcm_features(images)
            boundary_siglip, boundary_dino = self.adapter(hpcm_features)
            results["boundary_siglip_features"] = boundary_siglip
            results["boundary_dino_features"] = boundary_dino

        if extract_vision_features:
            if not self.vision_backbone_id.startswith("dinosiglip"):
                raise ValueError("full/boundary distillation currently requires a DinoSigLIP backbone")
            processed_images = self._prepare_vision_inputs(images)
            # Teacher targets are constants.  The same frozen modules are then
            # reused below for the student posterior with autograd enabled.
            with torch.no_grad():
                vision_features = self.vision_backbone(processed_images)
            dino_dim = self.vision_backbone.dino_featurizer.embed_dim
            vision_dino = vision_features[..., :dino_dim]
            vision_siglip = vision_features[..., dino_dim:]
            results["vision_dino_features"] = vision_dino
            results["vision_siglip_features"] = vision_siglip

        if extract_hpcm_features:
            if self.loss_scope == "full_visual":
                if not extract_vision_features:
                    raise ValueError("full_visual loss requires the full vision backbone")
                results["aligned_dino_features"] = self._run_frozen_posterior(
                    self.vision_backbone.dino_featurizer, boundary_dino
                )
                results["aligned_siglip_features"] = self._run_frozen_posterior(
                    self.vision_backbone.siglip_featurizer, boundary_siglip
                )
            else:
                results["aligned_dino_features"] = boundary_dino
                results["aligned_siglip_features"] = boundary_siglip

        return results

    def get_trainable_parameters(self) -> list:
        """
        获取可训练参数
        
        Returns:
            可训练参数的列表
        """
        trainable_params = []
        if self.train_adapter:
            trainable_params.extend(list(self.adapter.parameters()))
        return trainable_params
    
    def get_frozen_parameters(self) -> list:
        """
        获取冻结参数

        Returns:
            冻结参数的列表
        """
        frozen_params = []
        if self.freeze_hpcm_encoder:
            frozen_params.extend(list(self.hpcm_encoder.parameters()))
        if self.freeze_vision_backbone:
            frozen_params.extend(list(self.vision_backbone.parameters()))
        if not self.train_adapter:
            frozen_params.extend(list(self.adapter.parameters()))
        return frozen_params
    
    def training_step(self, batch, batch_idx):
        """
        训练步骤。batch 为 dict：image 或 images（原图），HPCM 特征在 forward 内由 HPCM 编码器实时提取。
        """
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
        else:
            raise ValueError("batch 需为 dict，包含 image 或 images")
        if images is None:
            raise ValueError("batch 中未找到 image 或 images")

        # One sampled augmentation is shared by the HPCM student and vision
        # teacher.  Applying two independent transforms would make the target
        # itself inconsistent and turn the objective into cross-view matching.
        images = self.image_augmentation(images)
        results = self.forward(images, extract_hpcm_features=True, extract_vision_features=True)
        
        aligned_siglip = results["aligned_siglip_features"]
        aligned_dino = results["aligned_dino_features"]
        vision_siglip = results["vision_siglip_features"]
        vision_dino = results["vision_dino_features"]
        
        # 计算损失
        loss_dict = self.loss_fn(
            aligned_siglip=aligned_siglip,
            aligned_dino=aligned_dino,
            vision_siglip=vision_siglip,
            vision_dino=vision_dino,
        )
        
        # 保存loss值供MetricsLogger实时打印使用
        self.last_train_losses = {
            "total_loss": float(loss_dict["total_loss"].detach().cpu()),
            "siglip_loss": float(loss_dict["siglip_loss"].detach().cpu()),
            "dino_loss": float(loss_dict["dino_loss"].detach().cpu()),
        }
        
        # 记录损失到wandb和prog_bar
        # total_loss显示在进度条上，所有loss都记录到wandb
        self.log("train/total_loss", loss_dict["total_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/siglip_loss", loss_dict["siglip_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/dino_loss", loss_dict["dino_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return loss_dict["total_loss"]
    
    def validation_step(self, batch, batch_idx):
        """Log clean loss and deterministic augmented loss for model selection."""
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
        else:
            raise ValueError("batch 需为 dict，包含 image 或 images")
        if images is None:
            raise ValueError("batch 中未找到 image 或 images")
        
        clean = self._distillation_loss(images)
        self._log_validation_losses("clean", clean)

        selected = clean
        if self.image_augmentation.enabled and self.image_augmentation.validation_enabled:
            augmented_images = self.image_augmentation(
                images, seed=self.image_augmentation.validation_seed + batch_idx
            )
            augmented = self._distillation_loss(augmented_images)
            self._log_validation_losses("aug", augmented)
            selected = augmented

        # Backward-compatible aliases.  Checkpointing and early stopping use
        # deterministic augmented validation when augmentation is enabled.
        self.log("val/total_loss", selected["total_loss"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/siglip_loss", selected["siglip_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/dino_loss", selected["dino_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", selected["total_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return selected["total_loss"]

    def _distillation_loss(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = self.forward(images, extract_hpcm_features=True, extract_vision_features=True)
        return self.loss_fn(
            aligned_siglip=results["aligned_siglip_features"],
            aligned_dino=results["aligned_dino_features"],
            vision_siglip=results["vision_siglip_features"],
            vision_dino=results["vision_dino_features"],
        )

    def _log_validation_losses(
        self, prefix: str, losses: Dict[str, torch.Tensor]
    ) -> None:
        for name in ("total_loss", "siglip_loss", "dino_loss"):
            self.log(
                f"val/{prefix}_{name}",
                losses[name],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
    
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        
        Returns:
            优化器和调度器配置
        """
        # 获取可训练参数
        trainable_params = self.get_trainable_parameters()
        
        if not trainable_params:
            raise ValueError("没有可训练的参数！请检查train_adapter配置。")
        
        # 优化器配置
        optimizer_type = self.optimizer_config.get("type", "adamw").lower()
        lr = self.learning_rate
        weight_decay = self.optimizer_config.get("weight_decay", 0.01)
        betas = tuple(self.optimizer_config.get("betas", [0.9, 0.999]))
        eps = self.optimizer_config.get("eps", 1.0e-8)
        
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
        elif optimizer_type == "sgd":
            momentum = self.optimizer_config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # 学习率调度器配置
        scheduler_config = self.lr_scheduler_config
        scheduler_type = scheduler_config.get("type", "cosine").lower()
        
        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            min_lr = scheduler_config.get("min_lr", 1.0e-6)
            # scheduler 按 optimizer step 更新，T_max 也必须是 step 数，不能使用 epoch 数。
            total_steps = int(self.trainer.estimated_stepping_batches)
            warmup_steps = int(scheduler_config.get("warmup_steps", 0) or 0)
            cosine_steps = max(1, total_steps - warmup_steps)
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=min_lr,
            )
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            start_factor = scheduler_config.get("start_factor", 1.0)
            end_factor = scheduler_config.get("end_factor", 0.0)
            total_iters = scheduler_config.get("total_iters", None)
            scheduler = LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters,
            )
        elif scheduler_type == "polynomial":
            from torch.optim.lr_scheduler import PolynomialLR
            total_iters = scheduler_config.get("total_iters", None)
            power = scheduler_config.get("power", 1.0)
            scheduler = PolynomialLR(
                optimizer,
                total_iters=total_iters,
                power=power,
            )
        elif scheduler_type == "constant":
            from torch.optim.lr_scheduler import ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            # 如果没有调度器，返回None
            return optimizer
        
        # Warmup配置
        warmup_steps = scheduler_config.get("warmup_steps", None)
        warmup_ratio = scheduler_config.get("warmup_ratio", None)
        
        if warmup_steps is not None or warmup_ratio is not None:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            
            if warmup_steps is None and warmup_ratio is not None:
                # 根据warmup_ratio计算warmup_steps
                if hasattr(self.trainer, "estimated_stepping_batches"):
                    total_steps = self.trainer.estimated_stepping_batches
                elif hasattr(self.trainer, "max_steps") and self.trainer.max_steps:
                    total_steps = self.trainer.max_steps
                else:
                    total_steps = 10000  # 默认值
                warmup_steps = int(total_steps * warmup_ratio)
            
            warmup_steps = max(1, int(warmup_steps))
            warmup = LinearLR(
                optimizer,
                start_factor=1.0 / warmup_steps,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[warmup_steps],
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 或 "epoch"
                "frequency": 1,
            },
        }


def load_model_from_config(config_path: Path) -> AlignModel:
    """
    从YAML配置文件加载模型
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        AlignModel实例
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("model", {})
    hpcm_config = model_config.get("hpcm", model_config.get("hpcm_encoder", {}))
    vision_config = model_config.get("vision", {})
    adapter_config = model_config.get("adapter", {})
    loss_config = config.get("loss", {})
    training_config = config.get("training", {})
    optimizer_config = training_config.get("optimizer", {})
    lr_scheduler_config = training_config.get("lr_scheduler", {})

    vision_backbone_id = vision_config.get("backbone_id", "dinosiglip-vit-so-224px")
    vision_backbone_checkpoint = vision_config.get("checkpoint")
    if vision_backbone_checkpoint:
        vision_backbone_checkpoint = project_root / vision_backbone_checkpoint

    model = AlignModel(
        hpcm_root=hpcm_config.get("hpcm_root", ""),
        hpcm_checkpoint=hpcm_config.get("checkpoint"),
        hpcm_model_name=hpcm_config.get("model_name", "HPCM_Base"),
        hpcm_scale_table_levels=hpcm_config.get("scale_table_levels", 60),
        freeze_hpcm_encoder=hpcm_config.get("freeze", True),
        vision_backbone_id=vision_backbone_id,
        vision_backbone_checkpoint=vision_backbone_checkpoint,
        vision_backbone_max_layer=vision_config.get("max_layer", 2),
        image_resize_strategy=vision_config.get("image_resize_strategy", "letterbox"),
        image_sequence_len=vision_config.get("image_sequence_len", 1),
        adapter_config=adapter_config if adapter_config else None,
        freeze_vision_backbone=vision_config.get("freeze_backbone", True),
        train_adapter=adapter_config.get("train_adapter", True) if adapter_config else True,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        learning_rate=optimizer_config.get("lr"),
    )
    return model


def create_align_model(
    hpcm_root: str = "",
    hpcm_checkpoint: Optional[str] = None,
    freeze_hpcm_encoder: bool = True,
    vision_backbone_id: str = "dinosiglip-vit-so-224px",
    vision_backbone_checkpoint: Optional[Path] = None,
    vision_backbone_max_layer: int = 2,
    adapter_config: Optional[Dict[str, Any]] = None,
    freeze_vision_backbone: bool = True,
    train_adapter: bool = True,
    loss_config: Optional[Dict[str, Any]] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    lr_scheduler_config: Optional[Dict[str, Any]] = None,
    learning_rate: Optional[float] = None,
) -> AlignModel:
    """创建 Align 模型（HPCM 编码器版：图像 -> HPCM -> Adapter -> 对齐特征，无 CGIC）。"""
    if vision_backbone_checkpoint is None:
        vision_backbone_checkpoint = project_root / "Align_adapter/vision_ckpt/vision_backbone_partial_layer2.safetensors"

    model = AlignModel(
        hpcm_root=hpcm_root,
        hpcm_checkpoint=hpcm_checkpoint,
        freeze_hpcm_encoder=freeze_hpcm_encoder,
        vision_backbone_id=vision_backbone_id,
        vision_backbone_checkpoint=vision_backbone_checkpoint,
        vision_backbone_max_layer=vision_backbone_max_layer,
        adapter_config=adapter_config,
        freeze_vision_backbone=freeze_vision_backbone,
        train_adapter=train_adapter,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        learning_rate=learning_rate,
    )
    return model


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("Align Model 测试")
    print("=" * 80)
    
    model = create_align_model(
        freeze_vision_backbone=True,
        train_adapter=True,
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    frozen_params = sum(p.numel() for p in model.get_frozen_parameters())
    
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {frozen_params:,}")
    
    print("\n" + "=" * 80)
    print("✓ 模型创建成功")
    print("=" * 80)
