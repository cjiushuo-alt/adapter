# HPCM3 Object full-visual 离线诊断：Epoch 10 与 Epoch 29

日期：2026-10-11；分支：`exp/hpcm3-object-dist`。本诊断仅读取权重和图像，在当前 LIBERO simulator 中渲染静态初始状态；**没有修改权重、训练、调用策略动作或执行 `env.step()` / 正式 rollout**。

## 结论

在同一批图像上，Epoch 29 的 clean 特征对齐明显好于 Epoch 10；但当前 simulator 的 Object 静态初始帧上，实际推理路径的加权 full-visual loss 仅从 **6.784 降到 6.714（约 1.0%）**。主相机反而从 **9.724 升至 10.074**，腕相机从 **3.845 降至 3.353**；主相机 DINO MSE 从 **1.383 升至 1.491**。Object simulator / Object clean 比值由 **3.63× 升至 4.50×**。因此，这批 probe **不支持“继续训练到 Epoch 29 已显著改善 simulator-domain feature alignment”**；也不能由静态初始帧推断完整动态 rollout 分布。

代码与数值检查没有发现 posterior 层边界或 DINO/SigLIP、agent/wrist token 顺序错位。确实存在输入协议差异：训练 HPCM 用解码后的 256px RGB，而当前 rollout 配置 `hpcm_raw_input_bypass=false`，HPCM 用的是经过 VLA processor 后再放大到 256px 的图像。不过即使两路都采用训练侧原始 RGB，simulator/clean gap 仍约 **3.43×→3.80×**；故输入协议差异**不是唯一解释**，不能据此直接断言它造成 0/5 闭环失败。

## 样本、权重和口径

- Clean：精确复现当前训练配置的 trajectory-held-out `val_max_samples=512` 子集，512 张来自 **157 条 validation trajectories**；四 suite 分别为 LIBERO-10 180、Goal 99、Object 99、Spatial 134 张。Object 的 99 张中 agent 51、wrist 48。与 `LiberoImageDataset` 实际选出的 512 个路径集合逐一比对，完全相同。训练/验证 trajectory 不交叉。
- Simulator：当前 LIBERO-Object 的 10 个 task，各取默认 initial-state 索引 **7、23、41**，每个状态只调用 `set_init_state` 渲染 agent/wrist 两路，共 **30 个状态、60 张图**；没有等待 10 个 no-op steps，更没有策略控制。这些 state 未用于此前 0–4 号 episode 的小规模 rollout，但**无法排除**与 expert 训练 demo 的 initial state 重合。
- 两个 Adapter 在完全相同的 clean 与 simulator 图像上比较。所有 loss 均为每张图 256 个 token 的 `SigLIP_MSE + 5 × DINO_MSE`，再对图像取均值；表中的比值是同一协议、同一视角下 `Object simulator mean / Object clean mean`。DINO/SigLIP 列均为**未加权** MSE。
- `train_canonical`：解码 256px RGB 同时进入 HPCM 和 teacher；teacher 用训练代码的 bilinear 224px 与归一化，student 穿过冻结视觉塔的 blocks 3 至倒数第二层。
- `rollout_effective`：同一 RGB 先走当前 rollout 的 JPEG/Lanczos3 224px、0.9 中心裁剪与 Object-Pro processor；teacher 为完整 HF VLA 视觉塔，student 的 HPCM 使用 SigLIP processor 通道再 bicubic 回 256px，之后调用实际 `CombinedVisionModel._forward_with_hpcm`。两套协议都以 **batch=1** 计算，匹配推理批量；因此这里的 clean 值**不是**训练时 batch=32、固定增强的 checkpoint `val_loss`。

| 组件 | SHA256 |
|---|---|
| Epoch 10 提取的 Adapter `.pt` | `55fdf7c93f1a70862279a072a27133e50864df7d4a3b07b5ff132d2cfef205f9` |
| Epoch 29 提取的 Adapter `.pt` | `8c38aae72470af636fecb7fb3c5928428ef164c4eda25675d13c6fb6159e6a63` |
| Epoch 29 完整 Lightning checkpoint | `3f04599200cc13be57545f346265e36c0cb7a918b5981cbc38b518013eab0b5f` |
| Object-Pro `model.safetensors` | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| HPCM `0.0018.pth.tar` | `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964` |
| trajectory split manifest | `3e1c1fa86d5cd73a7aff6979f08d47017e1cd1f8ba6f839e88ae391b9aa8fd15` |

Epoch 10 的完整 Lightning checkpoint 已被 `save_top_k=1` 后续覆盖，当前只有提取的 Adapter；**无法补算 Epoch 10 完整 checkpoint SHA**。Epoch 29 的 42 个提取 Adapter 张量与完整 checkpoint 中对应张量逐元素一致。

## 1. 训练与推理路径一致性

训练图像由 [`dataset.py`](../Align_adapter-HPCM3/code/dataset/dataset.py) 变为 `[-1,1]`；[`model.py`](../Align_adapter-HPCM3/code/model/model.py) 让 HPCM 在原 256px 图像上提取量化 latent，Adapter 输出 `(256,1024)` DINO 和 `(256,1152)` SigLIP，再执行完整塔 `blocks[3:-1]`。推理 [`combined_vision_model.py`](../Align_adapter-HPCM3/code/inference/model/combined_vision_model.py) 截去 blocks 0–2 后，同样不执行最后一个 block，最终拼成每视角 `(256,2176)`，DINO 在前、SigLIP 在后。

验证证据：

- Object-Pro 的 685 个 HF 视觉权重张量均能对应加载；相同的 224px 输入下，训练 tower 与 HF tower 的 teacher 输出在一个 4 图 batch 上 `max_abs=0`，形状 `(4,256,2176)`。
- batch=1、同一原始 RGB：HPCM 两种调用方式的输出 `max_abs=0`。在全部 512 clean + 60 simulator 图上，训练 student 与实际推理后段 student 的 DINO 平均 MAE 约 **0.0034–0.0040**、SigLIP **0.0065–0.0067**，平均相对 L2 约 **0.19%–0.25%**；所有图的单图 MAE 均低于 **0.01**。这是 FP16/BF16 权重与输出转换下的小数值差异，不是大规模后段拼接失配。
- 逐 token 数值与双视角 wrapper 测试确认，单视角前 1024 维为 DINO、后 1152 维为 SigLIP；双视角输出 `(1,512,2176)` 的前 256 个 token 对应 agent，后 256 个对应 wrist。
- 但当前 Object 评测配置的 raw bypass 为 `false`。在 60 张 simulator 图上，推理 HPCM 输入与原始 RGB（均在 `[-1,1]`）的像素 MAE：agent **0.100**、wrist **0.117**；DINO teacher 输入的归一化张量 MAE：**0.211 / 0.248**。因此“相同 RGB”不等于两条路径实际吃到相同像素。
- 额外检查 64 张 held-out 图：HPCM 以 batch=4 与逐张 batch=1 运行时，latent 的平均 MAE 为 **0.00543**、p90 **0.00880**，约 **0.73%** 元素的绝对差大于 0.1，最大接近 1。量化附近的数值路径对 batch size 敏感（机制为推断）；本报告的核心指标统一采用 batch=1，不能与训练日志的 batch=32 loss 逐位比较。

## 2. Full-visual loss：Object held-out clean vs simulator

每行 `DINO`、`SigLIP` 格式均为 `clean → simulator`；`all` 的 clean/simulator 样本数为 99/60，agent 为 51/30，wrist 为 48/30。

### 训练侧 canonical RGB 协议

| Epoch | 视角 | Clean total | Simulator total | Sim/Clean | DINO MSE | SigLIP MSE |
|---:|---|---:|---:|---:|---:|---:|
| 10 | all | 2.106 | 7.225 | 3.43× | 0.301 → 1.046 | 0.601 → 1.997 |
| 29 | all | 1.882 | 7.160 | 3.80× | 0.270 → 1.073 | 0.533 → 1.793 |
| 10 | agent | 2.291 | 10.141 | 4.43× | 0.331 → 1.479 | 0.636 → 2.744 |
| 29 | agent | 2.164 | 10.591 | 4.89× | 0.310 → 1.615 | 0.613 → 2.517 |
| 10 | wrist | 1.909 | 4.309 | 2.26× | 0.269 → 0.612 | 0.564 → 1.249 |
| 29 | wrist | 1.582 | 3.728 | 2.36× | 0.227 → 0.532 | 0.448 → 1.069 |

四 suite 的 512 张 clean 验证图整体均值由 **2.862 降至 2.452**。这与 checkpoint 选择损失 `2.6160 → 2.0581` 的下降方向一致，但两者一个是本次 batch=1 clean probe、一个是训练时的固定增强验证损失，数值不可直接比较。

### 当前 rollout-effective 预处理协议

| Epoch | 视角 | Clean total | Simulator total | Sim/Clean | DINO MSE | SigLIP MSE |
|---:|---|---:|---:|---:|---:|---:|
| 10 | all | 1.871 | 6.784 | 3.63× | 0.287 → 0.967 | 0.434 → 1.947 |
| 29 | all | 1.493 | 6.714 | 4.50× | 0.230 → 0.986 | 0.344 → 1.782 |
| 10 | agent | 1.837 | 9.724 | 5.29× | 0.304 → 1.383 | 0.316 → 2.807 |
| 29 | agent | 1.393 | 10.074 | 7.23× | 0.228 → 1.491 | 0.252 → 2.618 |
| 10 | wrist | 1.907 | 3.845 | 2.02× | 0.270 → 0.552 | 0.558 → 1.087 |
| 29 | wrist | 1.599 | 3.353 | 2.10× | 0.231 → 0.481 | 0.442 → 0.947 |

对 10 个 task 的配对差值按 task 重采样 20,000 次，rollout-effective simulator total 的 Epoch 29−10 均值差为 **−0.071**，task-cluster bootstrap 95% 区间 **[−0.218, +0.090]**；不能据此确认整体改善。agent 差值 **+0.350**（区间 **[+0.061, +0.664]**；7/10 个 task 变差），wrist 差值 **−0.492**（区间 **[−0.568, −0.420]**；10/10 个 task 改善）。该区间只反映这 10 个 task、固定 3 个 state/task 的重采样，不是 LIBERO 全轨迹或新场景的置信区间。

## 3. 判断与下一步

1. **先不要因 clean/aug val loss 下降就继续堆 epoch。** 在这批当前 simulator 静态帧上，Epoch 29 的总体对齐几乎未改善，尤其 agent-view DINO 变差；此前两个 checkpoint 的第一个 Object task 都是 0/5 rollout，虽与本离线结果方向一致，但不能证明因果。
2. **固定目标输入协议后做离线 A/B。** 当前训练与实际 HPCM 推理输入不一致；建议在相同、排除正式 eval initial states 的 probe 上，分别比较 raw bypass 和 processor-fed HPCM，并让训练的 student/teacher 使用与选定推理协议一致的预处理。不要只看总体 loss，要分别监控 agent/wrist、DINO/SigLIP。
3. **建立独立的 simulator-domain validation gate。** 采集与训练 demo、正式评测 initial states 隔离的静态及中途帧；用 trajectory/initial-state 划分，按 task 聚合。只有 agent-view DINO 和总体 simulator loss 都稳定下降、且 clean loss 不明显退化后，再启动正式 rollout。

局限：simulator probe 仅是静态 initial-state 图像，未执行 rollout 前 10 个 no-op steps，也没有覆盖策略生成的中后期帧；clean 与 simulator 图像虽同属 Object suite，却并非逐场景配对。初始状态与训练 expert demo 的重合尚不可核验。特征 MSE 也不是动作或成功率指标。故本报告判断只针对上述 60 张非正式 probe，不外推为完整闭环成功率。

复现脚本：[`fullvisual_offline_diagnostic.py`](../Align_adapter-HPCM3/code/analysis/fullvisual_offline_diagnostic.py)。本次使用 `--exact-validation-subset --validation-max-samples 512 --sim-state-indices 7 23 41 --sim-task-limit 10 --batch-size 1`，输出 JSON 的 SHA256 为 `e0f47e263f1d3a48762e44ef5f673ef415a785279f534b6c8133b16114030bac`（本地 `/tmp/hpcm3_fullvisual_exact_val512_sim60_batch1.json`；不含 RGB 或权重）。
