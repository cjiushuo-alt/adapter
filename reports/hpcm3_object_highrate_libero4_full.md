# Object 高码率 HPCM3：四套 LIBERO 全量轨迹图像联合训练

本实验只改变高码率 Object HPCM3 的训练语料规模：固定 λ=0.0483 HPCM、Object-Pro teacher、2 ResBlocks、block-2 边界蒸馏及 `SigLIP_MSE + 5×DINO_MSE`；没有 Prefix Head、full-visual loss 或旧 Adapter 初始化。Object task 0 五个配对状态的结果为原始 Object-Pro **5/5**、高码率重建 RGB→Object-Pro **5/5**、本次 Adapter **5/5**（旧高码率 Object Adapter 为 1/5）。高于旧高码率 Object Adapter 的 1/5，但五个状态样本不足以证明 suite 级改善。不把这五个曾用于历史实验的状态视为全新、无偏的论文正式测试集。

## 历史协议与本轮固定变量

历史 Spatial W&B 本地 run `loexl72x` 的 summary 记录 `global_step=48,829`、epoch 3、`val_loss=0.016296`；仓库历史配置为四套数据、batch 32、2 ResBlocks、同一边界损失。但历史 train/val 是逐帧随机划分，同 trajectory 可跨两侧，且剥离后的 Spatial `last-v1.pt` 不含可复核的训练 manifest，因此不能声称其四套全部图像已由该 checkpoint 自证。本轮用真实存在的四套 RLDS 导出 PNG，按 trajectory 分离 train/val，**训练侧不设 `train_max_samples`**。相对[旧高码率 Object 训练](hpcm3_object_highrate_boundary.md)，保留 batch 16、AdamW `1e-4`、cosine、FP16、Object-Pro 原生 teacher preprocessing 与 raw 256 HPCM 输入。为匹配历史 Spatial 约 156 万次图像曝光，预算上限 100,000 optimizer steps × batch 16；每 8,192 step 做一次验证，`min_delta=1e-4, patience=5` 早停。验证从 held-out 轨迹固定抽取 512 张，simulator 的 40 张只用于诊断，**不参与选模**。

| suite | train 轨迹 | train 图像 | val 轨迹 | val 图像 |
| --- | ---: | ---: | ---: | ---: |
| libero_10_no_noops | 342 | 182890 | 37 | 20048 |
| libero_goal_no_noops | 385 | 94748 | 43 | 9336 |
| libero_object_no_noops | 416 | 123150 | 38 | 10818 |
| libero_spatial_no_noops | 380 | 93052 | 52 | 12888 |
| **总计** | **1523** | **493840** | **170** | **53090** |

split manifest SHA256 `3e1c1fa86d5cd73a7aff6979f08d47017e1cd1f8ba6f839e88ae391b9aa8fd15`；train/val trajectory 交集 0，解码 RGB exact-frame 交集 **0**。历史 Object rollout audit 中 2480 个 RGB SHA 与训练图像精确重合 **0**；独立 simulator 20 个 state 与正式默认 state 碰撞 **0**。但 RLDS PNG 导出缺少完整 demo simulator initial state，**无法排除训练 demo 与正式初始状态等值**；这仍是论文正式使用前必须补齐的审计。正式 rollout 图像只用于 SHA 审计和最终闭环测试，未进入训练、验证或选模。四套数据中各抽一张 held-out 图像做 HPCM 与 Object-Pro block-2 数值预检，逐图值见[预检结果](../results/hpcm3_object_highrate_libero4_full/preflight_parity.json)。

## 训练与独立 simulator 曲线

模型由随机初始化开始，clean trajectory-held-out `val_loss` 决定最佳权重；simulator fresh-reset 图像来自前次固定 manifest，非正式评测 initial states。

| Optimizer step | Train | Clean val | Sim agent | Sim wrist | Sim all |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.099281 | 0.087121 | 0.186397 | 0.214411 | 0.200404 |
| 16384 | 0.059286 | 0.062788 | 0.155517 | 0.202318 | 0.178918 |
| 24576 | 0.042462 | 0.050679 | 0.140308 | 0.211032 | 0.175670 |
| 32768 | 0.046018 | 0.044763 | 0.140989 | 0.202609 | 0.171799 |
| 40960 | 0.035581 | 0.040914 | 0.118910 | 0.168816 | 0.143863 |
| 49152 | 0.036530 | 0.038080 | 0.132543 | 0.168843 | 0.150693 |
| 57344 | 0.035973 | 0.035858 | 0.126612 | 0.190881 | 0.158747 |
| 65536 | 0.035832 | 0.034443 | 0.112689 | 0.162051 | 0.137370 |
| 73728 | 0.030817 | 0.033224 | 0.106366 | 0.158931 | 0.132648 |
| 81920 | 0.028906 | 0.032308 | 0.107045 | 0.154298 | 0.130671 |
| 90112 | 0.033419 | 0.031816 | 0.109084 | 0.148020 | 0.128552 |
| 98304 | 0.032197 | 0.031610 | 0.105476 | 0.143660 | 0.124568 |

实际停止 epoch **4**、optimizer steps **100000**，最佳 clean validation loss **0.03160997**。曲线完整记录：[clean](../results/hpcm3_object_highrate_libero4_full/validation_curve.jsonl)、[simulator 分视角及 DINO/SigLIP](../results/hpcm3_object_highrate_libero4_full/sim_validation_curve.jsonl)。训练日志与完整 checkpoint 留在机械盘 `/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_libero4_full`；只有小体积配置/指标进 Git。

## Object task 0：五个配对 episode

沿用先前 task 0 默认 initial states 0–4、seed 7、10 dummy steps、8-action queue、双相机和 `libero_object_no_noops`。A/B 复用历史同状态结果，C 本轮独立执行；初始 state SHA 逐条核对。C 按每次策略查询的真实 HPCM bitstream 计码率，重建图不输入 C。后续轨迹因策略不同不逐帧配对。

| state index | A 原始 RGB | B 重建 RGB | C 本次 HPCM3 | C exact train-frame matches | C agent/wrist bpp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 1 | 0 | 0.5689 / 0.4161 |
| 1 | 1 | 1 | 1 | 0 | 0.5545 / 0.4883 |
| 2 | 1 | 1 | 1 | 0 | 0.5497 / 0.4842 |
| 3 | 1 | 1 | 1 | 0 | 0.5522 / 0.4772 |
| 4 | 1 | 1 | 1 | 0 | 0.5493 / 0.5023 |
| **合计** | **5/5** | **5/5** | **5/5** | **0** | **0.5574 / 0.4638** |

[逐 episode](../results/hpcm3_object_highrate_libero4_full/episode_results.jsonl)、[frame audit](../results/hpcm3_object_highrate_libero4_full/audit.jsonl)、[真实码率](../results/hpcm3_object_highrate_libero4_full/rate.jsonl)与[完整 summary](../results/hpcm3_object_highrate_libero4_full/eval_summary.json)可复核。若 C 表现较好，仍需预先冻结方案后在未用于历次诊断的新初始状态和全 task 范围重复，不能据这五个样本直接宣称论文级收益。

## 权重及可复现性

| 权重 | SHA256 |
| --- | --- |
| HPCM λ=0.0483 | `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe` |
| Object-Pro model.safetensors | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| 本轮最佳 Adapter | `2ba5ee6a9500a812ef58b8c3080744aa0b5700b641106219b72fd2d949ad01d5` |
| 本轮最佳完整训练状态 | `0fcf49c65ed1a9cf3efe5ee1954ae64292edd2b1cc98f48cf222e0c0113f7432` |

训练启动时 Git commit `4a359d8aca7d2680b67c4415ca7e6a4b2961b0b1`。[训练配置](../results/hpcm3_object_highrate_libero4_full/train_config.yaml)、[评测配置](../results/hpcm3_object_highrate_libero4_full/eval_config.yaml)、[数据审计](../results/hpcm3_object_highrate_libero4_full/data_audit.json)和上述结果随报告提交；大型权重、原图、W&B 离线二进制与训练日志不上传。
