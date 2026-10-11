# HPCM3 第二个 LIBERO suite：高码率重建筛选

日期：2026-10-11；分支：`exp/hpcm3-object-dist`。**Object 是本轮最值得继续验证的候选，但目前得到的是“高码率 HPCM 重建 RGB → 原始 Object-Pro”的正向控制结果，不是高码率 HPCM3 Adapter 的成功结果。** LIBERO-Object task 0 在 λ=0.0483 下先导 **5/5**，达到预设 ≥4/5 门槛后扩展为 **20/20**。按预设顺序，Object 通过后没有启动 Goal 或 LIBERO-10 的筛选，也没有训练 Adapter 或运行完整 suite 评测。

## 协议与可复现性

- 复用[之前 Object λ=0.0018 的评测代码](../Align_adapter-HPCM3/code/analysis/object_hpcm_reconstruction_control.py)及 seed **7**、默认 initial states **0–4**；扩展运行使用 states **0–19**。task 0 为 “pick up the alphabet soup and place it in the basket”。每次策略查询的 simulator 原始 256×256 agent/wrist RGB 分别通过 HPCM `compress()` 生成真正的 y/z 熵编码 bitstream，再 `decompress()` 得到 uint8 重建 RGB，随后走原始 Object-Pro 的 JPEG/Lanczos3 224px、中心裁剪及 processor。策略为未经修改的 Object-Pro；使用 `libero_object_no_noops`、10 个 dummy steps 和 8-action 队列。
- 本轮唯一的控制变量是 HPCM checkpoint，从 `0.0018.pth.tar` 改为已有的 `HPCM/ckpt/0.0483.pth.tar`；没有替代权重。配置：[5-episode](../Align_adapter-HPCM3/code/configs/object_hpcm_highrate_reconstruction_ep5.yaml)、[20-episode](../Align_adapter-HPCM3/code/configs/object_hpcm_highrate_reconstruction_ep20.yaml)。命令示例：`python Align_adapter-HPCM3/code/analysis/object_hpcm_reconstruction_control.py --config Align_adapter-HPCM3/code/configs/object_hpcm_highrate_reconstruction_ep20.yaml --output-dir results/hpcm3_suite_highrate_screen/object_task0_ep20`，需在具备本仓库 LIBERO、HPCM 和 Object-Pro 依赖的环境中执行。
- bpp 由 **8 × (y bitstream 字节数 + z bitstream 字节数) / (256 × 256)** 计算；PSNR 用原始/解码后 uint8 RGB 计算，均为策略实际查询帧的逐帧平均。首查询的 HPCM `forward(training=False)` 与 bitstream 解码 `x_hat` 最大绝对差 **0**，验证重建流程，但策略始终使用真正 bitstream 的解码输出。没有保存或上传图像、bitstream 或权重。
- 五状态高码率先导与 20-episode 扩展的前 **5 条 episode 记录、174 条逐帧记录字节级相同**；这五个状态与低码率 Object 运行的首查询双视角原图 SHA256 也 **10/10 相同**。历史 A/C 控制与本轮使用相同的初始状态数组 SHA，但 A/C 的首帧不保证逐像素配对；闭环分叉后的帧更不可逐帧配对。A/C 只覆盖前五状态，不能把其 `5/5` 或 `0/5` 误写成 20 个状态的结果。

| 关键文件 | SHA256 |
|---|---|
| HPCM 高码率 `HPCM/ckpt/0.0483.pth.tar` | `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe` |
| HPCM 低码率 `HPCM/ckpt/0.0018.pth.tar` | `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964` |
| Object-Pro `model.safetensors` | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| Object-Pro action head | `dd5642994a89f81dfe33cdca460479972548de8eaa26aacdbb3b0a7516b2d723` |
| Object-Pro proprio projector | `3201fd9f18cb178958d656ac2ddd7163f8a7853535856243907f5b633af7c14c` |
| 历史 Object HPCM3 Epoch 29（仅 C 参考，未用于本轮 B 推理） | `8c38aae72470af636fecb7fb3c5928428ef164c4eda25675d13c6fb6159e6a63` |

评测时的 HEAD commit：`828b825acae40c9958e044630f898883d51526ae`；20-episode 支持代码当时尚未提交，随本报告一起提交。高码率 [5-episode 结果](../results/hpcm3_suite_highrate_screen/object_task0_ep5/summary.json) SHA256：`de4846613183566264cbbdd9a9028a01062d30b2d271ca95c06548071f3a298c`；[20-episode 结果](../results/hpcm3_suite_highrate_screen/object_task0_ep20/summary.json) SHA256：`df5bcd3038ef62556bd8267ca9b088d97f90ed2ea7f4c1d7cfcca4985ae5c4bf`。各 episode 的成功标记、initial-state SHA、查询数、分视角 PSNR/bpp 在对应目录的 `episode_results.jsonl`；每张查询图的源/重建 SHA、bitstream 字节数与指标在 `frame_metrics.jsonl`；完整配置和权重 SHA 在 `config.yaml`、`provenance.json`。

## 结果

| Suite / 码率 / task | A 原图 | B 重建 RGB | C HPCM3 | B agent PSNR / 实际 bpp | B wrist PSNR / 实际 bpp | B 查询 / 双视角帧 |
|---|---:|---:|---:|---:|---:|---:|
| Spatial / λ=0.0018 / task 0 | 5/5 | **5/5** | 5/5 | 32.99 dB / 0.0713 | 31.96 dB / 0.0829 | 51 / 102 |
| Object / λ=0.0018 / task 0 | 5/5 | **0/5** | 0/5 | 30.51 dB / 0.0970 | 32.77 dB / 0.0524 | 175 / 350 |
| Object / λ=0.0483 / task 0，先导 | 历史 5/5 | **5/5** | 高码率未测 | 39.04 dB / 0.5541 | 39.93 dB / 0.4890 | 87 / 174 |
| Object / λ=0.0483 / task 0，扩展 | 仅前五状态有 A | **20/20** | 高码率未测 | 39.03 dB / 0.5560 | 40.03 dB / 0.4781 | 363 / 726 |
| Goal、LIBERO-10 / λ=0.0483 | 未测 | **未测** | 未测 | — | — | — |

Spatial/Object 低码率数据引自[先前 A/B/C 报告](hpcm3_spatial_vs_object_reconstruction_control.md)，未在本轮重跑。Object 高码率 B 的 20 个 episode **0–19 全部成功**；前五个为 5/5，后十五个为 15/15，逐项记录见[20-episode JSONL](../results/hpcm3_suite_highrate_screen/object_task0_ep20/episode_results.jsonl)。这些是同一 task 的默认 initial states，不能外推为整个 Object suite 的成功率。高、低码率 B 的整段 PSNR/bpp 采自不同闭环轨迹：低码率失败时五次均运行到 35 次查询，高码率成功时每次为 15–32 次；不能将整段均值当作严格的同帧画质比较。前五个首查询 RGB 相同使**首帧**比较可控，但并不使后续轨迹逐帧配对。

## 判断与下一步

1. **Object 的低码率失败与可恢复的码率瓶颈相容。** 在相同 Object-Pro 策略、task 0 和五个状态上，真实重建输入从 λ=0.0018 的 0/5 升至 λ=0.0483 的 5/5，扩展又得 20/20；说明更高码率的重建 RGB 对该任务上的原始策略可用。不能从这个控制实验断定低码率 HPCM3 失败*仅*由压缩信息损失造成，也不能把 PSNR 当成任务信息的充分指标。
2. **推荐 Object 高码率作为论文第二个正向结果的优先候选**，先在独立 task/initial states 做原始 VLA 与重建 VLA 配对，再验证与高码率 HPCM latent 匹配的 Adapter。现有 Object Epoch 29 Adapter 是针对 λ=0.0018 的训练产物，**本轮未将其与 λ=0.0483 latent 拼接评测，当前没有“Object 高码率 HPCM3 成功”的证据**。若未来需要高码率 HPCM3 主结果，应另行设计/训练匹配码率的 Adapter 并按无数据重合协议评测；本轮未获授权进行训练。
3. Object 达到预设 ≥4/5 扩展门槛，故条件式 Goal、LIBERO-10 筛选没有触发；不应称它们为失败或被排除。Spatial 的 λ=0.0018 已有 task 0 正向 A/B/C；Object 的 λ=0.0483 正向 B 是控制基线，二者尚不能被写成两个同条件的 HPCM3 suite 主结果。

本轮只运行 Object task 0 的 **5 + 20** 个高码率 B episode（扩展重新运行前五状态），没有训练、修改网络结构/loss、覆盖历史结果、进行 500-episode 正式评测或修改任何 checkpoint。
