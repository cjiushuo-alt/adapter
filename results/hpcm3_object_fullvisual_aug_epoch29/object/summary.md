# HPCM3 Object-Pro：Epoch 29 rollout 检查

- 训练已按要求暂停在 Epoch 30 的约 7% 处；未完成的 Epoch 30 步数未保存，保留的完整训练 checkpoint 是 `epoch-epoch=29-step-step=462990-val_loss=2.0581.ckpt`。
- 本次从该 checkpoint 提取推理 Adapter，SHA256 为 `8c38aae72470af636fecb7fb3c5928428ef164c4eda25675d13c6fb6159e6a63`。权重保存在机械盘，未纳入 Git。
- 沿用 Epoch 10 相同的 Object-Pro VLA、HPCM、Adapter 结构与预处理、seed 7、LIBERO-Object 默认 initial states、每 task 50 episodes 及失败保护阈值（至少 5 episodes，成功率低于 20% 则停止）。
- 实际仅运行 task 0（“pick up the alphabet soup and place it in the basket”）的前 **5 个 episodes：0/5 成功**。触发保护后停止；**不是整个 Object suite 的成功率**。日志没有 episode 执行异常。
- Frame audit 扫描训练 split 的 493,840 个独立 RGB hash。本次记录 1,400 个 rollout step（每步主、腕两路图像），与训练帧的 exact RGB match 为 **0**。
- 与 Epoch 10 的 0/5 对比：本次 5 个 initial-state SHA256 和各 episode 首个主相机 frame SHA256 均与 Epoch 10 完全一致；验证损失从 2.6160 降至 2.0581，但这 5 个相同初始条件下的成功数没有改善。

原始记录见同目录的 `config.yaml`、`run.log`、`episode_results.jsonl`、`audit.jsonl` 和 `review_required.json`。由于保护条件提前终止，评测脚本没有生成完整 suite 的 `summary.json`。
