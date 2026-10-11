# HPCM λ=0.0018 是否破坏 Object 任务信息：task 0 配对控制实验

日期：2026-10-11；分支：`exp/hpcm3-object-dist`。本阶段**未训练、未改动 HPCM / VLA / HPCM3 权重或架构**。在 LIBERO-Object task 0 的默认 initial state 0–4 上，A（原始 Object-Pro）**5/5**，B（真实 HPCM bitstream 重建 RGB → 原始 Object-Pro）**0/5**，C（HPCM latent → Epoch 29 Adapter → Object-Pro）**0/5**。因此当前不能把 Object 失败仅归因于 Adapter 特征对齐；在此 λ 和这五个状态上，**即使绕开 Adapter，压缩重建也没有保留足以让原策略成功的视觉输入**。这不是对四个 suite、全部 Object task 或完整任务信息的证明。

## 配对协议与复用结果

- A 复用 [`results/object_pro_intact_smoke/episode_results.jsonl`](../results/object_pro_intact_smoke/episode_results.jsonl)；C 复用 [Epoch 29 结果](../results/hpcm3_object_fullvisual_aug_epoch29/object/episode_results.jsonl)。二者均为 Object-Pro、seed 7、task 0、默认 initial state 0–4、10 个 dummy steps、256px simulator、两个相机、8-action 队列，C 为当前 `hpcm_raw_input_bypass=false`。五个 initial-state SHA 和每个 episode 第一次策略查询的 agent/wrist 原始 RGB SHA 在 A/C 之间完全一致。
- B 使用相同 Object-Pro 权重、原生 `get_vla_action` 与 `prepare_images_for_vla` 路径；每次**策略查询**时，先对 simulator 两路原始 256×256 RGB 逐张调用 HPCM `compress()`，再把真实 y/z 熵编码字节流传给 `decompress()`，将 `x_hat` 量化为 uint8 RGB。重建图随后进入原生 JPEG/Lanczos3 224px 缩放、0.9 中心裁剪和 Object-Pro processor。没有把 HPCM latent 喂给 B 策略，也没有把未经量化的 `g_s(g_a(x))` 当成压缩重建。非查询的队列执行步不需要策略图像，故不压缩；PSNR/bpp 只统计实际送入策略的图像。
- B 重新跑的五个 initial-state SHA 和首次查询双视角 RGB SHA 均与历史 A/C 对应。由于三组之后动作不同，**后续帧不是逐帧相同**；“配对”只指起始状态与首查询观测。每个 B episode 共 35 次策略查询、70 张压缩图；无评测异常。首查询的 HPCM `forward(training=False)` 重建与 `compress→decompress` 的浮点 `x_hat` 最大绝对差为 **0**；另一次同状态静态双视角校验也为 0。
- B 的 codec 输入是原始 256px RGB；历史 C 因 `hpcm_raw_input_bypass=false`，HPCM 输入来自 Object-Pro processor 的 SigLIP 通道再放大到 256px。两者测试的是各自已定义的路径，**不是同一 latent 张量的“解码策略 vs Adapter 策略”比较**；不能把 B/C 的共同失败严格归因为同一处 latent 信息损失。
- 实际 bpp 按每帧 `(len(y_string)+len(z_string))*8/(256*256)`，不是 likelihood 估计；PSNR 按 uint8 解码重建 RGB 与原始 uint8 RGB 的 MSE 计算。HPCM checkpoint 以 60 档 scale table 更新，256px 无 padding。B 不保存 bitstream 或图像，只保存 SHA、字节数和逐帧指标。

| 权重 | SHA256 |
|---|---|
| HPCM λ=0.0018 `0.0018.pth.tar` | `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964` |
| Object-Pro `model.safetensors` | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| Object-Pro action head | `dd5642994a89f81dfe33cdca460479972548de8eaa26aacdbb3b0a7516b2d723` |
| Object-Pro proprio projector | `3201fd9f18cb178958d656ac2ddd7163f8a7853535856243907f5b633af7c14c` |
| C：Epoch 29 Adapter | `8c38aae72470af636fecb7fb3c5928428ef164c4eda25675d13c6fb6159e6a63` |

## 五个 episode 结果

以下 PSNR/bpp 为每个 B episode 实际策略查询帧的视角均值；bpp 是 y+z 字节流的实际码率。

| Initial state | A | B | C | B queries | Agent PSNR / bpp | Wrist PSNR / bpp |
|---:|:---:|:---:|:---:|---:|---:|---:|
| 0 | 成功 | 失败 | 失败 | 35 | 30.47 dB / 0.0969 | 33.29 dB / 0.0485 |
| 1 | 成功 | 失败 | 失败 | 35 | 30.58 dB / 0.0970 | 32.68 dB / 0.0542 |
| 2 | 成功 | 失败 | 失败 | 35 | 30.43 dB / 0.0968 | 32.51 dB / 0.0516 |
| 3 | 成功 | 失败 | 失败 | 35 | 30.51 dB / 0.0982 | 32.58 dB / 0.0533 |
| 4 | 成功 | 失败 | 失败 | 35 | 30.56 dB / 0.0958 | 32.79 dB / 0.0546 |
| **总计/均值** | **5/5** | **0/5** | **0/5** | **175** | **30.51 dB / 0.0970** | **32.77 dB / 0.0524** |

逐帧范围：agent 30.26–30.92 dB、0.0864–0.1016 bpp；wrist 30.45–34.76 dB、0.0391–0.0942 bpp。总共 350 张压缩图，均由真实 bitstream 解码。原始数据见 [`task0_ep5`](../results/hpcm3_object_compression_control/task0_ep5/) 中的 `episode_results.jsonl`、`frame_metrics.jsonl`、`summary.json` 和 `provenance.json`；逐帧 JSON 只含数值与哈希。

## 重建图检查与解释

检查 task 0、state 0、10 个 dummy steps 后的首查询双视角原图/重建图（本机临时图，不上传仓库）：重建保留了主要物体的位置和大色块，但主视角多个小包装的标签/字形和边缘明显模糊；腕视角小包装上的细字与纹理也被抹平。**这是单帧定性观察，不足以证明具体物体识别错误或单独造成失败。** 全图约 30–33 dB 的 PSNR 仍可能掩盖 Object 任务所依赖的小区域细节；闭环 0/5 同时也可能来自压缩造成的视觉分布偏移，二者尚不能拆分。注意 B 的后续帧沿 B 策略轨迹采集，不能与 A/C 后续帧直接做配对像素分析。

按预设“若结果有希望再扩展”的门槛，B 目前 0/5，**不扩展到 20**。当前最值得做的是固定同一 Object-Pro 策略和初始状态，比较更高码率 HPCM checkpoint 的 B 控制，辅以对象区域的重建质量/视觉特征诊断；然后在 Spatial task 0 做相同 B 控制，才能更直接解释“Spatial 成功、Object 失败”的差异。若高码率 B 恢复成功而 C 仍失败，再考虑 Adapter 特征对齐与 simulator-domain 泛化。

**暂不建议只用独立 simulator 图像微调 Epoch 29 Adapter。** B 已绕过 Adapter 却失败，单纯降低 Adapter 特征 loss 不能保证恢复被压缩/重建破坏的可辨细节，也不能保证解决视觉分布偏移。只有在确认某个 HPCM 码率的 B 能保持策略成功、并建立与训练 demo/正式 eval initial states 隔离的 simulator 验证集后，才值得评估小规模 Adapter 微调；其训练/验证仍需按 trajectory/initial state 隔离。

局限：仅一个 Object task、五个初始状态，无成功率置信结论；初始状态是否与专家 demo 重合未完全排除。A/C 为历史运行，但首查询图像哈希和初始状态已核对；B 的完整动作轨迹与历史 A/C 不同。此实验说明当前 λ 的重建路径在这些状态下不足以维持原策略成功，**尚不能定量分摊“信息损失”与“策略域偏移”的贡献，也不能推断 Spatial 在同一压缩控制下会成功**。

复现：[`object_hpcm_reconstruction_control.py`](../Align_adapter-HPCM3/code/analysis/object_hpcm_reconstruction_control.py)、[配置](../Align_adapter-HPCM3/code/configs/object_hpcm_reconstruction_control_ep5.yaml)。先加 `--probe-only` 检查 bitstream/forward 校验；正式 B 使用 `--output-dir results/hpcm3_object_compression_control/task0_ep5`。`summary.json` SHA256：`0be7ca2efac0066084d6f3a574d6aada77fb749a7e52ea8d8b8ecdce82bb951d`。
