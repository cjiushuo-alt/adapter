# Spatial λ=0.0018：raw256/trajectory-split HPCM3 重训及完整评测

本轮只替换高码率 Spatial 实验的 HPCM checkpoint 为冻结的 `0.0018.pth.tar`，从随机初始化训练独立 Adapter；四套全量 LIBERO 训练轨迹、trajectory-level split、raw256 HPCM 输入、Spatial-Pro teacher、2 ResBlocks、block-2 `SigLIP_MSE + 5×DINO_MSE`、batch 16、seed 42、验证频率/early stopping 等与高码率配置一致。历史低码率 `last-v1.pt` 权重**未复用**，也未作为严格对照。正式 rollout 使用同一 Spatial-Pro 和已有[低码率重建 RGB 完整结果](hpcm3_spatial_bitrate.md)的 DEFAULT 500 初始状态、seed 7、10 dummy steps、8-action queue。

| λ=0.0018 路径 | Success | Agent / wrist 真实 bpp | Agent / wrist PSNR |
| --- | ---: | ---: | ---: |
| 已有 Reconstruction→完整 Spatial-Pro | **485/500 (97.0%)** | 0.0704 / 0.0810 | 32.96 / 32.13 dB |
| 新 HPCM3→Spatial-Pro 后半视觉塔 | **484/500 (96.8%)** | 0.0702 / 0.0797 | 32.97 / 32.20 dB¹ |

¹ HPCM3 行的 PSNR 是对其实际观察帧做**诊断性重建**，重建 RGB 未输入 HPCM3 策略。两条闭环轨迹不同，整段均值不是逐帧配对画质。500 个 initial-state SHA 与首查询 1000 个双视图原始 RGB SHA 均一致；首查询 y/z bitstream 长度逐视图一致 1000/1000。配对状态中 HPCM3 独赢 **6** 个，重建独赢 **7** 个；不能将很小的差异解读为稳健优势。

| Task | Reconstruction | HPCM3 |
| ---: | ---: | ---: |
| 0 | 50/50 | 50/50 |
| 1 | 50/50 | 50/50 |
| 2 | 50/50 | 50/50 |
| 3 | 50/50 | 50/50 |
| 4 | 49/50 | 50/50 |
| 5 | 41/50 | 38/50 |
| 6 | 50/50 | 50/50 |
| 7 | 48/50 | 48/50 |
| 8 | 47/50 | 48/50 |
| 9 | 50/50 | 50/50 |

训练实际 **100000** optimizer steps，最佳 clean trajectory-held-out val loss **0.019068**；[训练曲线](../results/hpcm3_spatial_lowrate_retrain/validation_curve.jsonl)和[独立 simulator 曲线](../results/hpcm3_spatial_lowrate_retrain/sim_validation_curve.jsonl)分开保存，后者未用于选模。训练前四套 held-out 图像上的 teacher block-2 与 HPCM 输入数值核对误差为 0。新 Adapter SHA `60cf9fdd35c5f2a312ace7bc50eff015eb28bd00a593ef354c4b7982bd6b10cc`，HPCM SHA `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964`；其他权重 SHA 与运行 commit 见[provenance](../results/hpcm3_spatial_lowrate_retrain/provenance.json)。逐 episode 结果、实际 y/z bitstream bpp/PSNR、完整[汇总 JSON](../results/hpcm3_spatial_lowrate_retrain/summary.json)已保存；抽查 20 张视图的真实熵解码 latent 与前向 latent max abs=0，并非所有帧的证明。训练 raw RGB 与本轮 rollout 精确 SHA 重合 0；但 RLDS PNG 无 demo 完整 simulator initial state，**不能排除 demo 与正式初始状态等值**。正式评测状态未用于本轮训练或选模。逐帧 audit 保留在机械盘 `/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_lowrate_libero4_full/eval/lambda_0.0018_hpcm3/audit.jsonl`，SHA `f9a63529a6bb00a66912b6128f192188d509955ffa52b981452566d9edefeb94`，不进 Git。
