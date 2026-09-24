# HPCM3：Spatial 成功而 Object 失败的对照调查

日期：2026-09-24  
分支：`exp/hpcm3-object-dist`  
结论基于本地 checkpoint、训练日志、W&B 指标、逐 tensor 权重对比、拼接诊断和受保护 rollout；不是根据 README 推测。

## 结论摘要

Spatial 能成功、Object 仍为 0/5，并不是因为 Object 测试脚本加载了错误 suite、动作归一化键或 VLA checkpoint。三个控制实验已经确认：完整 Object-Pro 为 5/5，当前代码下历史 Spatial HPCM3 为 5/5，而同一套代码换成 Object HPCM3 后为 0/5。

主要原因有两层：

1. **Spatial-Pro 与 Object-Pro 只有结构相同，视觉塔权重并不相同。** 两个 `config.json` 完全一致，但 DINO 塔只有 4.66% 的 tensor 完全相同，SigLIP 塔只有 7.89% 的 tensor 完全相同。Object-Pro 的 LLM、projector、action head 和 proprio projector 也都不同。因此，历史 Spatial HPCM3 实际上是一个 **Spatial-Pro 专用的视觉替代器**，不能直接被视为适配所有 suite 的公共视觉编码器。
2. **新 Object Adapter 在离线验证集上收敛，但没有泛化到当前 rollout 渲染域。** 最佳验证损失为 `0.02494`，而 Object rollout 初始帧上的同一加权边界损失达到 `0.22895`（raw 路径）或 `0.26484`（历史 Spatial 预处理路径），是验证损失的约 9.2–10.6 倍。Spatial 的对应 rollout 边界损失只有 `0.02427`，与其历史验证损失 `0.01630` 同量级。

因此，继续增加 Object 训练 epoch 不是合适修复方向。当前失败的核心是 **teacher/task-specific visual tower + Object render-domain generalization gap**，其次是 Object 对细粒度物体身份更敏感。

## 1. 已核验的实验结果

| 实验 | VLA | Adapter | 预处理 | 结果 |
|---|---|---|---|---:|
| 完整 Object-Pro 控制 | Object-Pro | 不替换视觉塔 | 原生 | 5/5 |
| Spatial HPCM3 复现 | Spatial-Pro | 历史 `last-v1.pt` | 历史 Spatial 路径 | 5/5 |
| 旧 Object Adapter | Object-Pro | Object 专用 | 历史路径 | 0/5 |
| 旧 Object Adapter raw 对照 | Object-Pro | Object 专用 | raw 256 | 0/5 |
| 新四-suite Object Adapter | Object-Pro | Epoch 28，val=0.0248 | 历史路径 | 0/5 |
| 新四-suite Object Adapter raw 对照 | Object-Pro | Epoch 28，val=0.0248 | raw 256 | 0/5 |

新 Object 正式测试在前 5 个 episode 全部失败后按 guard 停止，没有浪费完整 500-episode 预算。历史路径检查了 1,400 条 main/wrist rollout frame 记录，exact train-frame match 为 0。

相关证据：

- `results/object_pro_intact_smoke/episode_results.jsonl`
- `results/spatial_hpcm3_repro_smoke/episode_results.jsonl`
- `results/hpcm3_object_dist_libero4/object/`
- `results/hpcm3_object_dist_libero4/object_raw_control/`
- `results/hpcm3_splice_parity/`

这些控制排除了以下解释：

- Object-Pro 本身失效；
- `libero_object_no_noops` 动作反归一化键错误；
- suite/task/initial-state 加载错误；
- HPCM3 拼接代码普遍不可用；
- 只需要关闭 center crop 或切换 raw 输入；
- 训练集与这 5 个 rollout 出现 exact RGB frame 重合。

## 2. 两个 Pro checkpoint 实际并不共享视觉编码器

### 2.1 结构配置相同

两个 checkpoint 的 `config.json` SHA256 都是：

```text
05978420d096033d0fe7b4905f0178edfcc964f58c67f8776293b81e93d9a7b6
```

二者均为：

- Qwen2.5-0.5B；
- `dinosiglip-vit-so-224px` 双塔；
- DINO 1024 维 + SigLIP 1152 维；
- 两幅图输入；
- 相同 projector 和 action-policy 结构。

“结构一致”不能推出“权重一致”。

### 2.2 视觉与非视觉权重均不同

逐 tensor 比较 `outputs/LIBERO-Spatial-Pro/model.safetensors` 和 `outputs/LIBERO-Object-Pro/model.safetensors`：

| 模块 | tensor 数 | 完全相同 tensor | 相同元素比例 | Object–Spatial 相对 L2 |
|---|---:|---:|---:|---:|
| DINO vision tower | 343 | 16（4.66%） | 4.82% | 0.1165 |
| SigLIP vision tower | 342 | 27（7.89%） | 7.67% | 0.1242 |
| Projector | 6 | 0 | 0.34% | 0.5755 |
| Language model | 290 | 0 | 0.42% | 0.2184 |
| Action queries | 1 | 0 | 4.61% | 0.2594 |

外置策略模块也不相同：

| 文件 | Spatial SHA256 | Object SHA256 |
|---|---|---|
| `action_head--checkpoint.pt` | `76aa4862…807bdb9` | `dd564299…6b2d723` |
| `proprio_projector--checkpoint.pt` | `1c61a089…9827008e` | `3201fd9f…3af7c14c` |
| `dataset_statistics.json` | `3f8b7854…d5d9282` | `2bff9cb2…371aa7` |

代码也允许这种情况：`prismatic/models/vlms/prismatic.py` 的 `vla-full-train` 和 `vla-sandwich-train` 分支会令 vision backbone 可训练。无论发布模型当时采用哪个具体 recipe，最终 checkpoint 的逐 tensor 证据已经证明两个视觉塔发生了 suite-specific 更新。

### 2.3 HPCM3 的两个 teacher 都提取正确

这次没有把 Spatial teacher 错用于 Object：

- Spatial partial teacher：103 个 tensor，与 Spatial-Pro 的 blocks 0–2 逐 tensor 完全一致；
- Object partial teacher：103 个 tensor，与 Object-Pro 的 blocks 0–2 逐 tensor 完全一致；
- 两个 partial teacher 的 SHA256 分别为 `46c4b671…a9baa` 和 `592479da…7e4e9`，确认它们彼此不同。

因此，Object 失败不是 teacher 提取或加载错 checkpoint，而是固定 HPCM3 容量对 Object-specific teacher 的逼近和域外泛化仍不足。

## 3. 训练损失与 rollout 边界损失发生严重脱节

损失定义均为：

```text
total = SigLIP_MSE + 5 × DINO_MSE
```

在同一个 task 0 / episode 0 initial state 上做 intact-vs-spliced 诊断：

| 模型/输入 | DINO 边界 MSE | SigLIP 边界 MSE | 加权边界损失 | 相对对应 val loss |
|---|---:|---:|---:|---:|
| Spatial HPCM3，历史路径 | 0.002735 | 0.010589 | **0.024267** | 约 1.49× |
| 新 Object，历史路径 | 0.046248 | 0.033607 | **0.264849** | 约 10.62× |
| 新 Object，raw 256 | 0.041188 | 0.023009 | **0.228947** | 约 9.18× |

训练侧：

- 历史 Spatial 只运行约 48,829 step（3 个完整 epoch + 第 4 个 epoch 的约 17%），最终 `val_loss=0.01630`；
- 新 Object 运行到 Epoch 29，被人工停止；最佳完整验证是 Epoch 28、step 447,557、`val_loss=0.02494`；
- Object 已使用约 9 倍更新步数，仍无法达到 Spatial 的验证误差或 rollout 边界误差。

这说明问题不是 Object “还没训够”。继续优化同一个离线 MSE 目标，主要会继续改善当前离线分布，而不会自动消除 rollout 分布差异。

新 Object 相比旧 Object 确实有进步：历史路径 DINO/SigLIP 边界 MSE 从 `0.06771/0.08713` 降到 `0.04625/0.03361`。但这个幅度仍不足以形成可用闭环策略。

另一个值得注意的现象是：单个 initial state 上的首个 8-action chunk MAE 并不可靠。新 Object raw 的 action MAE 约 `0.0207`，比 Spatial 的 `0.0966` 更小，却仍然 0/5。闭环控制会反复重新观测，物体身份或位置表征的小偏差可以持续累积；不能用单步 action 接近程度代替 rollout。

## 4. Object 离线图像和当前 rollout 存在明显渲染域偏移

训练图像来自 `libero_rlds` 中已经编码的成功 demo PNG；当前 rollout 则由本机现有 LIBERO/MuJoCo 环境重新渲染。对相同 suite/task 的代表性初始帧进行像素统计：

| 对照 | Train 平均亮度 | Rollout 平均亮度 | 均值偏移 | 像素 MSE | 像素 MAE |
|---|---:|---:|---:|---:|---:|
| Object | 0.5383 | 0.3993 | **0.1390** | **0.03307** | **0.16204** |
| Spatial | 0.4635 | 0.4691 | **0.0056** | **0.00997** | **0.03773** |

Object 训练帧表现为浅灰地面、更亮的整体照明；当前 rollout 是明显更暗的木纹地面和不同的物体材质/布局。Spatial 的台面、背景、相机和主体外观则高度接近。

这与 feature 诊断完全一致：Spatial 的 rollout 特征仍落在 Adapter 熟悉的区域，而 Object rollout 经过 HPCM 后落到了训练验证集没有覆盖好的区域。trajectory-held-out validation 只能避免同轨迹重合，不能检测 simulator/render-version shift。

四-suite 训练也没有解决这一点：训练集包含 546,930 张图，其中 Object 为 133,968 张；trajectory train split 中 Object 有 416 条 demo。增加其他 suite 图像提升了总体覆盖，却没有引入“当前 Object rollout 的暗地面/当前资产版本”这一关键域。

## 5. 为什么 Object 对残余视觉误差更敏感

这一点是基于 task 定义和观测的合理推断，不是单独实验已经证明的因果结论：

- Spatial task 的核心目标始终是黑碗，主要根据“碗相对 plate/ramekin/cookie box 的空间位置”选择实例；
- Object task 在同一桌面同时放置多个外形、大小和纹理相近的食品容器，需要根据语言精确识别 alphabet soup、ketchup、milk 等具体实例；
- 因此 Object 更依赖细粒度纹理、颜色和文字标签，恰好也是压缩与渲染亮度变化更容易破坏的信息。

这可以解释为什么 Object 的边界 cosine 已达到 DINO `0.958`、SigLIP `0.974`，但闭环仍完全失败：全局平均相似度很高，不代表少量关键 object token 被正确保留。

## 6. 根因排序

### 已证实

1. **Pro checkpoint 的视觉塔是 suite-specific，不是共享冻结权重。**
2. **Object offline validation 与当前 rollout 之间存在约 9–11 倍的 feature-loss gap。**
3. **Object 训练/rollout 图像域偏移显著大于 Spatial。**
4. **更多 epoch、四-suite 数据、历史预处理和 raw 预处理都没有恢复 Object。**
5. **完整 Object-Pro 可成功，因此 suite、动作头和 simulator 主链路可用。**

### 高概率但仍需定向实验确认

1. Object-specific fine-tuned blocks 0–2 所定义的目标映射比 Spatial teacher 更难由当前 HPCM latent + HPCM3 Adapter 表示；
2. Object 的关键小物体 token 误差被全局 MSE 平均掉；
3. Object 后半视觉塔/projector/action policy 对早期 token 扰动的任务敏感性与 Spatial 不同。

## 7. 对“同一个视觉编码器适配不同下游任务”的影响

如果论文目标是“同一压缩视觉编码器/同一 Adapter 无需重训即可服务多个下游任务”，当前发布的四个 Pro checkpoint 不是干净的实验条件，因为它们连被模拟的 teacher vision tower 都不同。

有两条可行路线：

### 路线 A：保留发布的 suite-specific Pro checkpoint

- 每个 suite 单独训练 HPCM Adapter；
- 用与当前评测完全同版本的 simulator/assets 重新生成训练 demo；
- train/val 按 trajectory 划分，eval initial states 必须排除；
- 加入 token-aware、后半视觉输出或 action-level 蒸馏，不能只依赖全局 layer-2 MSE。

这条路线能回答“压缩前端能否替换每个 task-specific VLA 的视觉塔”，但不能声称一个 Adapter 跨所有下游 VLA 通用。

### 路线 B：重新建立共享视觉塔实验条件（更符合原研究目标）

- 选定一套固定的原始 DINOv2 + SigLIP 权重并在所有 suite 中冻结；
- 在同一个共享视觉塔之后，分别训练或加载 suite-specific projector/LLM/action head；
- 只训练一次 HPCM Adapter 去模拟这个固定视觉塔；
- 再分别连接 Spatial/Object/Goal/LIBERO-10 非视觉部分评测。

这是验证“一次视觉压缩训练，多下游任务复用”的正确设计。代价是需要重新训练下游非视觉部分，不能直接把当前发布 Pro checkpoint 当作共享 encoder 基线。

## 8. 建议的下一步

1. **停止继续训练当前 Object Adapter。** Epoch 28 已足以证明相同离线目标继续下降不能解决 rollout。
2. **先做 shared-frozen-vision 小规模可行性实验。** 固定 Spatial-Pro 的视觉塔或原始 pretrained 双塔，只训练 Object 非视觉部分，然后接历史 Spatial HPCM3；先用 task 0 的 20 episodes 验证。
3. 若必须保留 Object-Pro，使用当前 simulator 重新采集 Object 训练 demo，但与 500 个默认 eval initial states 做 hash 隔离；先训练 task 0 小集并以 rollout-domain held-out feature loss 作为 gate。
4. 新增 per-token 诊断：报告目标物体区域 token 的 MSE/cosine、top-k token error，而不是只看全局均值。
5. 训练期间加入固定的、非 eval-state 的 simulator-rendered probe set。只有 probe feature loss 接近 Spatial 的约 `0.024` 水平后，才启动完整 rollout。

## 最终判断

Spatial 的成功不是 HPCM3 对所有 VLA-Adapter-Pro 天然通用的证据，而是“Spatial-specific teacher + 与训练相近的 Spatial rollout 渲染域”下的成功。Object 使用了另一套经过 suite-specific 更新的视觉塔，同时现有离线 Object 图像与当前 rollout 有明显渲染差异；当前 Adapter 在离线验证集上的低 loss 因此不能转化为闭环成功率。

如果研究目标保持为“一次训练视觉编码器，适配多个任务”，下一步应切换到 **共享且冻结的视觉塔 + suite-specific 非视觉后端**，而不是继续为当前 Object-Pro teacher 增加蒸馏 epoch。
