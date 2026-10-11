# HPCM3 Object Epoch 29：推理预处理 raw bypass 离线 A/B

日期：2026-10-11；分支：`exp/hpcm3-object-dist`。**结论：不建议为 Epoch 29 启用 `hpcm_raw_input_bypass=true` 来改善 Object 的视觉特征对齐，也不建议仅凭这个改动启动小规模 rollout。** 在相同的 60 张 simulator 静态图上，B 的加权 full-visual loss 为 **12.103**，显著高于当前 A 的 **6.632**（+82.5%）；10/10 个 task 的配对均值均变差。agent、wrist、DINO、SigLIP 无一改善。这是静态初始状态特征诊断，不能直接换算为闭环成功率。

## 实验设置与可比性

- 权重不变：Epoch 29 Adapter、Object-Pro VLA 视觉塔和 HPCM checkpoint。A/B 共用同一 Adapter、冻结 posterior 和**相同的 Object-Pro 原始完整视觉塔 teacher 输出**。teacher 的输入始终采用正式推理侧 `prepare_images_for_vla` 的 JPEG/Lanczos3 224px、0.9 中心裁剪，再经 HF Object-Pro processor；没有把 B 的 teacher 改成 raw-RGB teacher。
- 唯一干预是 HPCM 输入。A（当前 `hpcm_raw_input_bypass=false`）把 processor 的 SigLIP 三通道 bicubic 重采样到 256px；B（`true`）直接使用 simulator 原始 256×256 RGB，线性缩放至 `[-1,1]`。两者均调用实际 `CombinedVisionModel._forward_with_hpcm`，使用 FP16 Adapter、同一冻结 HF posterior `blocks[3:-1]`，batch=1。代码与 [`CombinedVisionModelWrapper.set_raw_images`](../Align_adapter-HPCM3/code/inference/model/model.py) 的 raw 转换一致；但本次没有调用完整策略或动作 head。
- 样本完全配对：复用[上次诊断](hpcm3_object_fullvisual_offline_diagnosis.md)的 LIBERO-Object 10 个 task、每 task 默认 initial state 索引 7/23/41、agent/wrist 双视角，共 30 个状态、60 张图；只执行 `set_init_state`，`env.step()`、动作预测、训练和正式 rollout 均为 **0**。每张新渲染图在同一次进程中同时用于 A/B 和同一个 teacher。
- 与上次的状态数组 SHA256 和 task/state/view 键 **60/60 一致**，但重新渲染的 RGB SHA256 仅 **41/60 相同**：agent 30/30，wrist 11/30。上次没有保存 RGB，故不能声称 60 张都与历史帧逐像素相同；下表只用本次同帧 A/B 配对比较。作为口径 sanity check，上次 A 的总体 loss 为 6.714，本次为 6.632，差别与 wrist 帧重渲染变化相容，不能当成独立验证。
- Loss：逐帧对 256 个 token 求未加权 DINO、SigLIP MSE；`full-visual = SigLIP_MSE + 5 × DINO_MSE`，再对帧平均。95% 区间按 **10 个 task** 对 B−A 配对 task 均值重采样 20,000 次，仅衡量这 10 个 task、固定三个状态的变异；不是全 LIBERO rollout 的置信区间，也不是随机分配实验的因果检验。

| 权重 / 数据 | SHA256 |
|---|---|
| Epoch 29 Adapter `adapter-epoch29.pt` | `8c38aae72470af636fecb7fb3c5928428ef164c4eda25675d13c6fb6159e6a63` |
| Epoch 29 完整 Lightning checkpoint（仅核对，未加载评估） | `3f04599200cc13be57545f346265e36c0cb7a918b5981cbc38b518013eab0b5f` |
| Object-Pro `model.safetensors` | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| HPCM `0.0018.pth.tar` | `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964` |
| 上次诊断样本元数据 JSON | `e0f47e263f1d3a48762e44ef5f673ef415a785279f534b6c8133b16114030bac` |

## 结果

数值越低越好。`Δ` 为 B−A；区间仅用于 full-visual 列。

| 视角（帧数） | 指标 | A：processor-fed | B：raw bypass | Δ / 相对变化 |
|---|---|---:|---:|---:|
| 总体（60） | Full-visual | 6.632 | 12.103 | **+5.471 / +82.5%** |
|  | DINO MSE | 0.977 | 1.741 | +0.763 / +78.1% |
|  | SigLIP MSE | 1.745 | 3.400 | +1.655 / +94.8% |
| Agent（30） | Full-visual | 9.915 | 15.344 | **+5.429 / +54.7%** |
|  | DINO MSE | 1.475 | 2.268 | +0.793 / +53.8% |
|  | SigLIP MSE | 2.540 | 4.005 | +1.464 / +57.6% |
| Wrist（30） | Full-visual | 3.349 | 8.863 | **+5.514 / +164.6%** |
|  | DINO MSE | 0.480 | 1.214 | +0.734 / +152.9% |
|  | SigLIP MSE | 0.950 | 2.795 | +1.845 / +194.2% |

按 task 配对重采样，full-visual 的 B−A 95% 区间：总体 **[+5.211, +5.706]**，agent **[+4.984, +5.801]**，wrist **[+5.217, +5.772]**；每个视角均为 **10/10 task 变差**。这批 probe 上 raw bypass 的负面差异不仅大，而且跨 task 和双视觉分支一致。其含义是：虽然 raw 输入更接近 Adapter 训练时 HPCM 接收的 RGB，**但在实际 Object-Pro 推理 teacher 始终吃 processor 图像的口径下，单独改 HPCM 输入反而使 teacher/student 不匹配更严重**。不应把“恢复训练侧 HPCM 像素”直接等同于“改善完整 VLA 特征对齐”。

## 决策与边界

维持当前 A 配置；**不建议把 B 作为下一次小规模 Object rollout 的优先候选**。先调查训练时同图增强、HPCM 输入和 teacher 输入是否成对一致，并建立独立 simulator-domain 验证集；如果以后修正训练/teacher 协议，再重新做同帧 A/B，并在总体及 agent-view DINO 均改善后考虑受控少量 rollout。本次没有改变任何权重、训练状态或评测 YAML。

局限：仅是固定三个静态初始状态/task，没有 rollout 的 no-op 等待、运动中帧或策略状态；initial state 与 expert demo 是否重叠仍无法排除。特征 MSE 与动作质量、成功率并非同一指标。因此“显著变差”仅指上述固定 probe 的 feature alignment，不外推到全分布；极小概率的动作侧反例仍需后续闭环实验判断。

复现脚本：[`object_raw_bypass_ab.py`](../Align_adapter-HPCM3/code/analysis/object_raw_bypass_ab.py)。本次指标 JSON 位于本机 `/tmp/hpcm3_object_epoch29_raw_bypass_ab.json`，SHA256 `f893c0e31b7a4c34f74345a363271da4c5ab9769cdc2a2635a9c16951ca8b88a`；JSON 仅含指标、state/RGB hash 和样本元数据，不含图像或权重。
