# HPCM3 与 VLA-Adapter 图像增强协议一致性验证

日期：2026-09-28  
分支：`exp/hpcm3-object-dist`

## 结论

**验证支持原假设：HPCM3 没有按 VLA-Adapter 的图像增强协议训练，且当前 Object HPCM3 对该协议中的外观扰动明显缺乏鲁棒性。**

在各自对应的 suite held-out 图像上，完整 VLA 增强使：

- Spatial HPCM3 加权边界蒸馏损失从 `0.015770` 上升到 `0.026507`，为 clean 的 **1.681×**；
- Object HPCM3（Epoch 28）从 `0.027903` 上升到 `0.122526`，为 clean 的 **4.391×**；
- Object 的 brightness 单项已经达到 clean 的 **2.928×**，与此前发现的 Object rollout 明显变暗高度一致；
- Object HPCM3 在 Spatial 图像上也达到 **3.427×**，说明脆弱性不只是 Object 图像内容造成的，还与 Object teacher/Adapter 映射本身有关。

因此，下一步不应继续增加当前 Object Adapter 的 epoch，也没有证据要求先加深 Adapter。优先方案应是：**保持结构、teacher、HPCM 和损失不变，加入与 VLA-Adapter 一致的训练增强后重训 Object Adapter。**

本验证证明了“增强协议不一致”和“Object 对这些扰动更敏感”，但尚未证明增强重训一定能恢复闭环成功率；最终因果闭环仍需要 augmentation-matched Object Adapter 的 rollout A/B。

## 1. 要验证的问题

VLA-Adapter 的 LIBERO 训练入口默认开启 `image_aug=True`，训练增强为：

```text
random_resized_crop: scale=[0.9, 0.9], ratio=[1.0, 1.0]
random_brightness:   [0.2]
random_contrast:     [0.8, 1.2]
random_saturation:   [0.8, 1.2]
random_hue:          [0.05]
```

当前 HPCM3 Dataset 只执行确定性中心方形裁剪、resize 到 256 和归一化，没有上述随机增强。因此本实验检查：

1. 当前 HPCM3 在 clean held-out 图像上的 loss 是否与训练记录一致；
2. 加入 VLA 的 crop、亮度、对比度和颜色增强后，边界蒸馏 loss 上升多少；
3. Object 是否比 Spatial 更敏感；
4. 差异来自图像 suite，还是来自 Adapter/teacher 组合本身。

## 2. 实验设计

诊断脚本：`Align_adapter-HPCM3/code/analysis/augmentation_sensitivity.py`

### 2.1 数据

- split manifest：`hpcm3_object_dist_libero4/data_split.json`
- manifest SHA256：`3e1c1fa86d5cd73a7aff6979f08d47017e1cd1f8ba6f839e88ae391b9aa8fd15`
- 只从 trajectory-held-out validation episodes 抽样；
- LIBERO-Spatial：64 张，其中 agent view 32 张、wrist view 32 张；
- LIBERO-Object：64 张，其中 agent view 32 张、wrist view 32 张；
- 固定 seed：42；
- 每张图、每种增强使用由路径和模式确定的 stateless seed。

### 2.2 模型

| 系统 | Adapter SHA256 | teacher partial SHA256 |
|---|---|---|
| Spatial HPCM3 | `2860078fd497f3453846f21f9d5ac809307e9deb2ceaf134b4153f161c83a501` | `46c4b6716c8eaa43405c17048a9f0bc25c825e74c0698bac559172c7867a9baa` |
| Object HPCM3 Epoch 28 | `01bb664adec72e5af20a3077dc314fc23f50a31632638374f697b959e51f99e1` | `592479daa4b1fa1aa8d6756a1117d281d8fa492939aec3a948aded580db7e4e9` |

两套系统使用同一个 HPCM checkpoint：

```text
dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964
```

### 2.3 增强和指标

增强直接调用当前训练环境中的 `dlimp.transforms.augment_image`，参数和操作顺序与 VLA-Adapter 代码一致：

- `clean`：不增强；
- `crop`：只做 random resized crop；
- `brightness`：只做 brightness；
- `contrast`：只做 contrast；
- `color`：brightness + contrast + saturation + hue；
- `full`：crop + brightness + contrast + saturation + hue。

每次增强只生成一份 RGB，**同一份增强结果同时送入 HPCM student 和 vision teacher**，因此 loss 不包含两条分支输入不一致造成的伪误差。

指标与训练一致：

```text
weighted_loss = SigLIP_MSE + 5 × DINO_MSE
```

## 3. 主要结果

### 3.1 对应 suite 结果

| 系统 / probe suite | 模式 | Weighted loss | 相对 clean | DINO MSE | SigLIP MSE |
|---|---|---:|---:|---:|---:|
| Spatial HPCM3 / Spatial | clean | 0.015770 | 1.000× | 0.001876 | 0.006388 |
| Spatial HPCM3 / Spatial | crop | 0.023105 | 1.465× | 0.002590 | 0.010155 |
| Spatial HPCM3 / Spatial | brightness | 0.018709 | 1.186× | 0.002171 | 0.007852 |
| Spatial HPCM3 / Spatial | contrast | 0.016864 | 1.069× | 0.002017 | 0.006780 |
| Spatial HPCM3 / Spatial | color | 0.020112 | 1.275× | 0.002318 | 0.008521 |
| Spatial HPCM3 / Spatial | **full** | **0.026507** | **1.681×** | **0.002881** | **0.012102** |
| Object HPCM3 / Object | clean | 0.027903 | 1.000× | 0.004414 | 0.005832 |
| Object HPCM3 / Object | crop | 0.079567 | 2.852× | 0.013220 | 0.013469 |
| Object HPCM3 / Object | brightness | 0.081699 | 2.928× | 0.013988 | 0.011759 |
| Object HPCM3 / Object | contrast | 0.030709 | 1.101× | 0.004907 | 0.006176 |
| Object HPCM3 / Object | color | 0.084415 | 3.025× | 0.014601 | 0.011409 |
| Object HPCM3 / Object | **full** | **0.122526** | **4.391×** | **0.020990** | **0.017577** |

sanity check：

- Spatial clean `0.015770` 与历史 Spatial val loss `0.01630` 接近；
- Object clean `0.027903` 与 Epoch 28 最佳 val loss `0.02494` 同量级。

这说明探针与实际训练 loss 标度一致，并非诊断实现产生了新的数值口径。

### 3.2 交叉 suite 对照

| Adapter / teacher | Probe 图像 | Clean | Full aug | 相对 clean |
|---|---|---:|---:|---:|
| Spatial | Spatial | 0.015770 | 0.026507 | 1.681× |
| Spatial | Object | 0.017693 | 0.037086 | 2.096× |
| Object | Spatial | 0.022987 | 0.078768 | 3.427× |
| Object | Object | 0.027903 | 0.122526 | 4.391× |

交叉结果表明：

1. Object 图像确实比 Spatial 图像更容易放大误差，尤其是 crop；
2. 但更主要的差异来自 Object Adapter/teacher 组合：它在 Spatial 图像上仍然比 Spatial HPCM3 脆弱得多；
3. Object 的增强误差主要由 DINO 分支放大。Object/Object 的 DINO MSE 从 `0.004414` 上升到 `0.020990`，约 **4.76×**。

## 4. 与 rollout 失败的关系

此前审计发现：

- Object 训练帧与当前 rollout 的平均亮度偏移为 `0.1390`，像素 MAE 为 `0.16204`；
- Spatial 对应值只有 `0.0056` 和 `0.03773`；
- Object rollout 边界 loss 是其 val loss 的约 9.2–10.6 倍；
- Spatial rollout 边界 loss 只约为历史 val loss 的 1.49 倍；
- Object raw/no-crop 对照仍失败，因此不能把问题只归因于 rollout 中的固定 center crop。

本次新实验补上了关键机制证据：

- 当前 Object Adapter 对亮度扰动单项就有接近 3× 的 loss 放大；
- 完整 VLA 增强达到 4.391×；
- Spatial 对相同协议明显更稳定。

因此，“HPCM3 未接受 VLA 增强训练，导致 Object 对 rollout 外观变化不鲁棒”已经从单纯推测提升为**有直接离线扰动实验支持的高概率根因**。

但 augmentation probe 仍不是 rollout A/B。Object-Pro 与 Spatial-Pro 的视觉塔权重不同、Object 又更依赖细粒度物体身份，这些因素仍可能共同作用。

## 5. 下一步建议

### 5.1 应做

保持以下内容不变：

- HPCM checkpoint；
- HPCM3 网络深度和参数量；
- Object-Pro teacher blocks 0–2；
- `SigLIP_MSE + 5 × DINO_MSE`；
- trajectory split；
- optimizer 和 early stopping 主框架。

只修改训练数据路径：训练样本进入 HPCM 和 teacher 前，先生成一份 VLA-compatible augmentation，然后两条分支共享这份图像。

验证至少同时记录：

1. clean trajectory-held-out loss；
2. fixed-seed augmented trajectory-held-out loss；
3. 非 eval initial state 的当前 simulator-rendered probe loss。

建议 rollout gate：增强重训后的 Object full-augmentation loss 应显著低于当前 `0.122526`，且 clean loss 不应明显劣于 `0.027903`；只有 simulator probe 接近 Spatial rollout 的约 `0.024`–`0.03` 量级后，再运行完整 Object rollout。

### 5.2 不应做

- 不继续给当前无增强 Object Adapter 堆 epoch；
- 不先通过加深 Adapter 掩盖输入分布问题；
- 不让 student 和 teacher 各自独立采样增强参数；
- 不用 augmented val 单一指标替代 clean val 和 simulator probe。

## 6. 局限性

1. 每个 suite 使用 64 张图，每张图每种模式采样一次固定随机增强；这是诊断规模，不是论文最终统计规模。
2. 增强在 HPCM 使用的 256×256 canonical RGB 上执行；VLA RLDS 管线通常在 policy resize 后执行。所用算子、参数和顺序一致，但插值层面不是逐像素 bit-exact。
3. 本实验没有训练新权重，也没有运行新的闭环 rollout，因此只能确定当前模型的增强敏感性，不能提前保证增强重训后的成功率。
4. 历史 Spatial checkpoint 原训练采用 frame split；本次 probe 为公平比较，对两套 checkpoint 都统一使用新的 trajectory-held-out manifest。

## 7. 复现命令

Spatial：

```bash
/home/cjs/anaconda3/envs/align-adapter/bin/python \
  Align_adapter-HPCM3/code/analysis/augmentation_sensitivity.py \
  --label spatial-hpcm3 \
  --adapter-checkpoint Align_adapter-HPCM3/ckpt/last-v1.pt \
  --teacher-checkpoint Align_adapter/vision_ckpt/vision_backbone_partial_layer2.safetensors \
  --split-manifest /media/cjs/shared/linux/Align_adapter/hpcm3_object_dist_libero4/data_split.json \
  --suites libero_spatial_no_noops libero_object_no_noops \
  --samples-per-view 32 --batch-size 8 --seed 42 \
  --output /tmp/hpcm3-spatial-augmentation-sensitivity-n64.json
```

Object：

```bash
/home/cjs/anaconda3/envs/align-adapter/bin/python \
  Align_adapter-HPCM3/code/analysis/augmentation_sensitivity.py \
  --label object-hpcm3-epoch28 \
  --adapter-checkpoint /media/cjs/shared/linux/Align_adapter/hpcm3_object_dist_libero4/adapter-best.pt \
  --teacher-checkpoint /media/cjs/shared/linux/Align_adapter/hpcm3_object_dist/teacher/vision_backbone_partial_layer2.safetensors \
  --split-manifest /media/cjs/shared/linux/Align_adapter/hpcm3_object_dist_libero4/data_split.json \
  --suites libero_spatial_no_noops libero_object_no_noops \
  --samples-per-view 32 --batch-size 8 --seed 42 \
  --output /tmp/hpcm3-object-augmentation-sensitivity-n64.json
```

诊断没有保存或提交任何 RGB、feature tensor、checkpoint 或其他大文件。
