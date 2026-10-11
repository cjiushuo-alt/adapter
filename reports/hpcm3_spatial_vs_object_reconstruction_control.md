# Spatial / Object：HPCM λ=0.0018 重建 RGB 控制实验

日期：2026-10-11；分支：`exp/hpcm3-object-dist`。**结论：Spatial task 0 的五个初始状态上，原始 Spatial-Pro（A）、真实 HPCM bitstream 重建 RGB → Spatial-Pro（B）、历史 Spatial HPCM3 → Spatial-Pro（C）均为 5/5；对应 Object task 0 为 5/5、0/5、0/5。** 这表明 Spatial 成功至少**不需要**诉诸“Adapter 从 latent 恢复了原始策略无法从重建 RGB 使用的信息”这一解释：完整 Spatial-Pro 直接吃 HPCM 重建图也能完成这五个 episode。Object 的 B 失败则提示当前 λ 下重建图对 Object-Pro 的闭环控制不足，但不能单凭 5 个状态把原因唯一归为不可逆的信息损失。

## 复用协议与配对程度

- 复用 [Object 重建控制](hpcm3_object_compression_information_control.md)的 [`ReconstructionControl`](../Align_adapter-HPCM3/code/analysis/object_hpcm_reconstruction_control.py)：在每次策略查询时，对 simulator 原始 256×256 agent/wrist RGB 分别执行真正的 HPCM `compress()`/`decompress()`，将 y/z 熵编码字节流解码为 uint8 RGB，然后进入 Spatial-Pro 的原生 JPEG/Lanczos3 224px、0.9 中心裁剪和 HF processor。A/B 都调用原始 Spatial-Pro 策略、动作头和 proprio projector，未修改任何权重；C 复用历史 `last-v1.pt` HPCM3 的同五个状态结果。
- 三组均为 LIBERO-Spatial task 0（“pick up the black bowl between the plate and the ramekin and place it on the plate”）、默认 initial state **0–4**、seed **7**、10 个 dummy steps、双相机、8-action 队列、`libero_spatial_no_noops`。五个 initial-state SHA 在本次 A/B 与历史 C 之间一致。A/B 各组开始前在模型/codec 加载后重新设为 seed 7；同一 episode 的 A/B **首查询双视角 RGB 哈希 5/5 完全相同**。
- 历史 C 首查询 RGB 哈希与本次重渲染的 A/B **0/5 相同**，尽管状态数组 SHA 一致；当前 simulator 的重渲染有额外变异。因此 C 是“同状态、同基础 seed 的历史成功参考”，**不是**与本次 A/B 逐像素严格配对的运行。A/B 闭环分叉后的帧也不能逐帧配对。C 评测使用历史 processor-fed HPCM 路径（非 raw bypass），不是 B 的同一 latent 张量。
- 为避免上传历史 31 MB 的完整 rollout audit，仅提交其前五个首查询状态/双视角哈希的[核对快照](../results/hpcm3_spatial_compression_control/task0_ep5/historical_c_initial_frames.json)；完整源 audit 的 SHA256 `c8fe1209772df1682e22f60127e3c0c8103074367808334824d7d67d90f1ed87` 已记录于 provenance。
- B 的实际 bpp 按 `(len(y_string)+len(z_string))*8/(256*256)`；PSNR 是原始/重建 uint8 RGB 的逐帧 MSE。仅统计真正送入策略的图像，不统计动作队列执行期间未查询策略的帧。Spatial B 共 **51 次查询、102 张压缩图**；首查询双视角的 HPCM `forward(training=False)` 与 bitstream 解码 `x_hat` 最大绝对差均为 **0**。未保存/提交 RGB、bitstream、checkpoint。

| 权重 | SHA256 |
|---|---|
| 共同 HPCM λ=0.0018 `0.0018.pth.tar` | `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964` |
| Spatial-Pro `model.safetensors` | `5d70e76161d92fa962c53e78e10b4af3426b6ec7e5c0ea872e7a8162c961a8d3` |
| Spatial-Pro action head | `76aa486231ebce057aa398dc5fcecf71698b293bdcc16d7354e1ac8cb807bdb9` |
| Spatial-Pro proprio projector | `1c61a08975ffade832a2dbfa0587ef6b109ec27f66eb8124f02727aa9827008e` |
| 历史 Spatial HPCM3 `last-v1.pt`（C） | `2860078fd497f3453846f21f9d5ac809307e9deb2ceaf134b4153f161c83a501` |

## Spatial 五个 episode

| State | A | B | 历史 C | B 查询数 | B agent PSNR / 实际 bpp | B wrist PSNR / 实际 bpp |
|---:|:---:|:---:|:---:|---:|---:|---:|
| 0 | 成功 | 成功 | 成功 | 11 | 33.02 dB / 0.0710 | 32.10 dB / 0.0802 |
| 1 | 成功 | 成功 | 成功 | 10 | 33.03 dB / 0.0710 | 31.98 dB / 0.0826 |
| 2 | 成功 | 成功 | 成功 | 11 | 33.02 dB / 0.0704 | 31.99 dB / 0.0830 |
| 3 | 成功 | 成功 | 成功 | 10 | 32.87 dB / 0.0720 | 31.91 dB / 0.0821 |
| 4 | 成功 | 成功 | 成功 | 9 | 33.02 dB / 0.0723 | 31.81 dB / 0.0872 |
| **合计/逐帧均值** | **5/5** | **5/5** | **5/5** | **51** | **32.99 dB / 0.0713** | **31.96 dB / 0.0829** |

结果见 [`task0_ep5`](../results/hpcm3_spatial_compression_control/task0_ep5/) 的 `episode_results.jsonl`、`frame_metrics.jsonl`、`summary.json`、`provenance.json`；历史 C 的五条成功记录见 [`spatial_hpcm3_repro_smoke`](../results/spatial_hpcm3_repro_smoke/episode_results.jsonl)。`summary.json` SHA256 为 `a9ad1fdfc4d73bd64b491c536e982c62df4016317ffc3a961ca5230f28caba3d`。

## 与 Object 同协议 B 控制并排

| Suite / task 0 | A 原图 | B 重建图 | C HPCM3 | B agent PSNR / bpp | B wrist PSNR / bpp | B 查询 / 压缩帧 |
|---|---:|---:|---:|---:|---:|---:|
| Spatial | 5/5 | **5/5** | 5/5 | 32.99 dB / 0.0713 | 31.96 dB / 0.0829 | 51 / 102 |
| Object | 5/5 | **0/5** | 0/5 | 30.51 dB / 0.0970 | 32.77 dB / 0.0524 | 175 / 350 |

Object 与 Spatial 的 B 沿不同闭环轨迹采样：Object 五次都走满 35 次查询，而 Spatial 在 9–11 次查询后成功。因此整段轨迹的 PSNR/bpp 平均值**不能直接当作同一图像分布的压缩难度排名**。只看首查询五对图：Spatial agent/wrist 平均 **33.49/32.57 dB、0.0613/0.0554 bpp**；Object 为 **30.80/30.58 dB、0.0881/0.0902 bpp**。这一差别与 Object 首帧更难在相同 λ 下重建的假设一致，但场景、目标大小、任务语义和 suite-specific VLA 权重均不同，PSNR 也不能衡量关键物体身份是否保留。

## 解释与下一步

1. **Spatial 在这五个状态上对重建 RGB 足够鲁棒。** B 与 A/C 同为 5/5，使“Spatial 成功完全依赖 Adapter 对 latent 的独特利用”缺乏必要性证据；C 成功仍证明历史 Adapter 可用于该 Spatial-Pro，但不能从天花板结果量化其相对 B 的额外贡献。
2. **Object 的压缩输入是优先瓶颈，但不是唯一已证原因。** Object B 绕开 Adapter 仍为 0/5；结合[Object 单帧重建检查](hpcm3_object_compression_information_control.md)中小包装标签/边缘模糊，值得优先验证关键物体细节与压缩域偏移。Object C 与 B 的 HPCM 输入协议不同，且 Object-Pro/Spatial-Pro 视觉塔和下游权重不同，不能将 C 的失败简单等同于同一 latent 的信息丢失。
3. **建议下一步先做无训练码率扫掠。** 在 Object task 0 的同五个状态上测试更高码率 HPCM checkpoint 的 B；若先导 5 个 episode 恢复，再扩到 20，并增加对象区域重建/视觉特征与动作分歧分析。之后在更多 Spatial/Object tasks 和独立 initial states 上重复 A/B/C，区分任务和 checkpoint 特异性。只有 Object B 能成功而 C 仍失败时，才把独立 simulator 图像上的 Epoch 29 Adapter 微调作为优先实验；否则先解决压缩图对原策略不可用的问题。

本结论只覆盖各 suite 的 task 0、五个状态，不推断整体 suite 成功率，也不证明 Spatial 与 Object 差异具有单一因果来源。本次没有新训练或权重修改。

复现：[`spatial_hpcm_reconstruction_control.py`](../Align_adapter-HPCM3/code/analysis/spatial_hpcm_reconstruction_control.py) 与[配置](../Align_adapter-HPCM3/code/configs/spatial_hpcm_reconstruction_control_ep5.yaml)。
