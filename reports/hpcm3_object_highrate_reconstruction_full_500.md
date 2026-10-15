# Object λ=0.0483 RGB 重建：完整 500-episode 配对控制

固定 HPCM `0.0483.pth.tar` 和完整 Object-Pro，使用同一 raw 256 RGB 数值路径进行真实 y/z 熵编码、解码与 RGB 重建，再由 Object-Pro 原生 JPEG/Lanczos3、center crop、processor 处理。LIBERO-Object 官方 DEFAULT 初始状态、10 tasks × 50 episodes、seed 7、10 dummy steps、8-action queue 与[已有高码率 HPCM3 正式结果](hpcm3_object_highrate_libero4_full_500.md)一致；模型权重未修改。逐 episode 的 state SHA 500/500 一致，首查询双视图的 RGB SHA **1000/1000 一致**。首查询 y/z bitstream 长度逐视图完全一致 **336/1000**；y_bytes 差值范围 [-24, 28]、z_bytes 差值范围 [-4, 4]。这表明两种策略加载上下文中真实 codec 编码并非保证逐字节相同；不能把相同 RGB/权重误写为完全同一 bitstream。后续轨迹因策略分支不同不逐帧配对。

| 路径 | Success | Agent / wrist 真实 bpp | Agent / wrist PSNR |
| --- | ---: | ---: | ---: |
| 本轮 RGB Reconstruction→完整 Object-Pro | **487/500 (97.4%)** | 0.5540 / 0.5098 | 39.01 / 39.81 dB |
| 已有 HPCM3→Object-Pro 后半视觉塔 | **472/500** | 0.5553 / 0.5020 | 未在历史 500-episode 运行中记录；不得填入本轮 PSNR |

同一初始状态上的 Reconstruction 独赢 **24** 个、HPCM3 独赢 **9** 个。成功率不能单靠码率或 PSNR 解释；若差距很小，不能声称一条路径优于另一条。bpp 为逐查询 y/z bitstream 字节数计算的全帧均值，不是 entropy model 估计。PSNR 是本轮重建闭环轨迹上的逐帧均值，不是两条策略的同帧 PSNR。

| Task | Reconstruction | HPCM3 |
| ---: | ---: | ---: |
| 0 | 48/50 | 40/50 |
| 1 | 49/50 | 46/50 |
| 2 | 50/50 | 50/50 |
| 3 | 42/50 | 45/50 |
| 4 | 50/50 | 47/50 |
| 5 | 49/50 | 46/50 |
| 6 | 50/50 | 49/50 |
| 7 | 49/50 | 49/50 |
| 8 | 50/50 | 50/50 |
| 9 | 50/50 | 50/50 |

完整[配置](../results/hpcm3_object_highrate_reconstruction_full_eval/config.yaml)、[逐 episode](../results/hpcm3_object_highrate_reconstruction_full_eval/episode_results.jsonl)、[逐查询码率与 PSNR](../results/hpcm3_object_highrate_reconstruction_full_eval/rate.jsonl)、[汇总](../results/hpcm3_object_highrate_reconstruction_full_eval/summary.json)和[provenance](../results/hpcm3_object_highrate_reconstruction_full_eval/provenance.json)已保存。HPCM SHA `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe`，Object-Pro model SHA `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604`，HPCM3 Adapter SHA `2ba5ee6a9500a812ef58b8c3080744aa0b5700b641106219b72fd2d949ad01d5`。本轮抽检 20 个真实熵解码 latent 与 HPCM3 forward latent 逐元素相等（max abs=0），不将抽样核对表述为全帧证明。训练图像与 rollout raw RGB exact SHA 重合 **0**；但 RLDS PNG 没有 demo 完整 simulator initial state，训练 demo 与正式初始状态是否等值仍无法确认。完整逐帧 SHA audit 留在机械盘 `/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_reconstruction_full/full_eval_attempt2/audit.jsonl`，SHA `334b5a29a0665c8fb974363add98b9832d0ecd0777bec474abf59c153c09a67b`，不上传 Git。
