# HPCM3 Object-Pro：Epoch 10 rollout 检查

- 训练 checkpoint：`/media/cjs/shared/linux/Align_adapter/hpcm3_object_fullvisual_aug_libero4/checkpoints/epoch-epoch=10-step-step=169763-val_loss=2.6160.ckpt`，该轮验证损失为 **2.6160**。
- 推理 Adapter：从上述 checkpoint 提取；SHA256 为 `55fdf7c93f1a70862279a072a27133e50864df7d4a3b07b5ff132d2cfef205f9`。权重文件保留在机械盘，未纳入 Git。
- 评测配置：LIBERO-Object / Object-Pro VLA，seed 7，默认 initial states，每个 task 计划 50 episodes；前 5 个 episodes 若成功率低于 20% 则停止。
- 实际结果：仅完成 task 0（“pick up the alphabet soup and place it in the basket”）的前 **5 个 episodes，0/5 成功**。保护条件触发后停止，**这不是整个 Object suite 的成功率**。
- Frame audit：扫描当前 trajectory 划分的训练集，得到 493,840 个独立 RGB frame hash；本次 rollout 记录 1,400 帧，两路相机均无 exact train-frame match（0）。

原始记录见同目录的 `config.yaml`、`run.log`、`episode_results.jsonl`、`audit.jsonl` 和 `review_required.json`。本次未修改 checkpoint，也未执行更多 rollout。
