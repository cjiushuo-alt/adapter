# Object 高码率 HPCM3 离线诊断：prefix token、数据覆盖与 simulator 域差

日期：2026-10-13；分支：`exp/hpcm3-object-dist`。**首要发现是实现层面的非等价：当前 HPCM3 拼接只给 DINO 后半塔 256 个 patch token，丢弃了完整 Object-Pro 在 block-2 仍保留的 1 个 CLS + 4 个 register token。** 即使把真实 block-2 patch 特征直接注入现有截断塔，最终 DINO 特征也不能复现原塔。补齐真实 prefix token 后，oracle 的最终特征和 projector 输出接近数值精度一致。独立 simulator 图像另有显著的 block-2 特征域差；固定 16,384 张采样在 wrist 上存在次要覆盖差距，但不能替代前两个解释。本阶段没有训练、修改权重、环境 `step()` 或正式 LIBERO rollout。

此前同五个 Object task 0 默认初始状态的闭环结果是原始 Object-Pro **5/5**、真实高码率 HPCM 重建 RGB→Object-Pro **5/5**、新高码率 HPCM3→Object-Pro **1/5**，详见[训练/评测报告](hpcm3_object_highrate_boundary.md)。本报告的误差只来自**新的离线图像样本**，不能把静态特征误差直接解释为上述五个 episode 的因果成功率。

## 协议和可复现性

- 固定 HPCM `0.0483.pth.tar`、最佳 Object Adapter、Object-Pro VLA/action head/proprio projector；不写 checkpoint。HPCM 输入严格复现数据集与正式推理的 **CPU 上 uint8→FP32→`[-1,1]` 后再拷贝到 GPU**，teacher 则走 Object-Pro 原生 JPEG/Lanczos3、中心裁剪及 processor，双塔、projector 与动作头使用原权重。首次单图核对：stage raw 与直接 raw 最大差 0、wrapper 与直接 CombinedVisionModel 的最终特征最大差 **0**、重复前向最大差 **0**，DINO/SigLIP 拼接顺序均正确。曾发现仅把归一化移到 GPU 就有约 `1.2e-7` 输入差并触发量化边界变化；此错误协议的中间结果不用于下文判断。
- [诊断脚本](../Align_adapter-HPCM3/code/analysis/diagnose_object_highrate_offline.py)按训练 `random.Random(42)` 的实际固定 cap 算法重建 16,384 张已抽中图像集合，再从**相同训练轨迹**选未抽中图像；另选 trajectory-held-out 图像。每类/每视角随机固定 **16 张、来自 16 条不同轨迹**；已抽中与未抽中按轨迹配对，但不是同时间帧。独立 simulator 使用先前保存的 [fresh-reset manifest](../reports/hpcm3_object_highrate_boundary.md) 中 10 个 Object task × 2 次 reset × 2 视角，共 40 张（每视角 20 张）；该 manifest 与正式评测 initial-state 哈希碰撞数为 0，诊断没有读取或渲染正式 initial states。固定采样 seed `20261013`，每张解码 RGB 的 SHA256、样本路径、逐样本和 256-token 均值数组均在[完整指标 JSON](../results/hpcm3_object_highrate_offline/v4/diagnostic.json)。
- Oracle 1（*patch-only*）：在内存中把 Adapter 的输出替换为**同一 RGB** 的真实 Object-Pro block-2 patch tokens，仍走实际截断后半塔和 projector。Oracle 2（*with-prefix*，仅离线诊断）：同时恢复真实 DINO CLS/register tokens 后再过相同的后半塔；不是可部署的 HPCM3 路径，也未修改任何磁盘权重。每类/每视角用固定的前 **2 张**作特征 oracle；动作核对用独立 simulator 的 **3 对**双视角静态图像、相同 task 文本和固定零 proprio，仅调用一次动作预测，不执行动作。动作 parity 是数值诊断，不是成功率。
- 配置：[诊断 YAML](../Align_adapter-HPCM3/code/configs/diagnose_object_highrate_offline.yaml)。执行命令：`HF_HUB_OFFLINE=1 python Align_adapter-HPCM3/code/analysis/diagnose_object_highrate_offline.py --config Align_adapter-HPCM3/code/configs/diagnose_object_highrate_offline.yaml`。该脚本若目标目录已存在会拒绝覆盖。

| 固定输入 | SHA256 |
| --- | --- |
| HPCM `0.0483.pth.tar` | `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe` |
| 最佳 Adapter `adapter-best.pt` | `422c8216e4fbb91a9e3775148a2e43b9486cbb76dd530b601a6063ca309d0fb2` |
| Object-Pro `model.safetensors` | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| action head / proprio projector | `dd5642994a89f81dfe33cdca460479972548de8eaa26aacdbb3b0a7516b2d723` / `3201fd9f18cb178958d656ac2ddd7163f8a7853535856243907f5b633af7c14c` |
| trajectory split / simulator 样本 manifest | `378be2671d58e3e508e6761e2ea83f42fce8f48a6a63af22ecec19ef7d413855` / `8784c72a8337878ab582e26b95e5fa4dfa098cd3c6d58409976e60c3af2b99c7` |
| 最终离线指标 JSON | `8b441e839dd0b1ee0263471c7629d18f5a50b7e4f38a984f8895b7b00282f37e` |

诊断进程记录的 Git commit 为 `ec0fffb`；脚本、YAML、最终 JSON 与本报告随后的提交一起推送。原图和 checkpoint 仍在机械盘，没有上传。

## Oracle：先定位拼接非等价

Object-Pro 的 DINO 型号为 `vit_large_patch14_reg4_dinov2`，block-2 内部 token 实际包含 **5 个 prefix（CLS + 4 registers）+ 256 个 patch**；`get_intermediate_layers()` 默认只返回 256 个 patch。SigLIP `vit_so400m_patch14_siglip_224` 无 prefix。现有 Adapter 输出只有 256 个 token，`skip_vision_backbone_layers()` 把它们直接交给后续 self-attention，导致 DINO 的 attention 上下文与原塔不同。下表是独立 simulator 样本的 oracle 平均 MSE（每视角 n=2）；这些样本上的 SigLIP patch-only oracle 已非常接近，而 DINO 仍远离原塔。

| 注入方式 / 视角 | 最终 DINO MSE | 最终 SigLIP MSE | Projector MSE | 最终 DINO token cosine |
| --- | ---: | ---: | ---: | ---: |
| patch-only / agent | 2.0275 | 0.00195 | 0.06424 | 0.9227 |
| **with-prefix / agent** | **0.00168** | 0.00195 | **0.000229** | **0.99995** |
| patch-only / wrist | 1.5412 | 0.00281 | 0.04493 | 0.9398 |
| **with-prefix / wrist** | **0.00128** | 0.00281 | **0.000233** | **0.99996** |

三对独立 simulator 静态图像的原始 Object-Pro 对比：patch-only oracle 动作 chunk 平均绝对差分别为 **0.00539 / 0.00536 / 0.00805**，with-prefix 降为 **0.00124 / 0.00126 / 0.00151**，真实 Adapter 则为 **0.02076 / 0.01462 / 0.02900**。相应最大绝对差和 8×7 动作形状见 JSON。with-prefix 仍非逐元素零差，主要存在 BF16 原塔与拼接 FP16/FP32 后半塔的数值路径差；因此结论是**结构性差异已被定位并大幅消除**，不是声称位级完全一致，更不能由三次静态动作预测推断闭环效果。

## 逐层误差与数据覆盖

以下均为同一 RGB 上 Student 对完整 Object-Pro 的 token MSE；block-2 为 Adapter 输出，最终层是截断塔输出，Projector 为融合后的投影。每个 RLDS 组/视角 n=16；simulator 组/视角 n=20。完整 JSON 同时给出平均 token cosine、relative L2、p90、逐 token MSE/cosine 数组和最高误差 token 排名。

| 来源 / 视角 | block-2 DINO | block-2 SigLIP | 最终 DINO | 最终 SigLIP | Projector |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练已抽中 / agent | 0.00226 | 0.00514 | 3.192 | 0.623 | 0.194 |
| 训练未抽中 / agent | 0.00214 | 0.00497 | 2.883 | 0.527 | 0.169 |
| held-out 轨迹 / agent | 0.00207 | 0.00472 | 2.869 | 0.371 | 0.150 |
| **独立 simulator / agent** | **0.06253** | **0.06402** | **5.439** | **4.112** | **0.546** |
| 训练已抽中 / wrist | 0.00659 | 0.01296 | 2.072 | 1.019 | 0.133 |
| 训练未抽中 / wrist | 0.01170 | 0.01583 | 2.619 | 1.153 | 0.161 |
| held-out 轨迹 / wrist | 0.01265 | 0.01755 | 2.028 | 1.370 | 0.150 |
| **独立 simulator / wrist** | **0.05240** | **0.06087** | **3.284** | **4.413** | **0.406** |

对应的 token cosine 也显示层级放大：simulator agent DINO 为 block-2 **0.867**、最终 **0.813**，SigLIP 为 **0.931→0.853**，融合 projector 为 **0.910**；simulator wrist 分别为 DINO **0.902→0.899**、SigLIP **0.938→0.838**、projector **0.926**。simulator agent 最终 DINO 高误差 token 为 **82、97、81**（各自进入单图 top-5 的次数为 **16/20、11/20、13/20**）；agent SigLIP 的 token **233** 为 **19/20**。wrist DINO 的 token **156** 为 **15/20**；这些是 16×16 patch 网格的索引，尚未与物体位置或任务因果区域对应，不应直接解释为“特定物体损坏”。

训练固定 cap 实际选中 agent **8,210**、wrist **8,174** 张；训练轨迹其余 agent **52,066**、wrist **52,102** 张，held-out 各 **6,708** 张。对这次按轨迹配对、但非同帧的 16 组小样本，agent 已抽中/未抽中 block-2 加权损失 `SigLIP+5×DINO` 为 **0.0164/0.0157**，没有可见的欠覆盖；wrist 为 **0.0459/0.0743**，存在差距。wrist 未抽中与 held-out（0.0808）接近，但远低于 simulator wrist（0.3229）。这些组的画面时刻并未匹配，16 张/组也不足以断言 cap *导致* wrist 差距；可判断的是**固定采样不足不是当前 simulator 误差的唯一或首要解释**。

独立 simulator 的 block-2 加权损失 agent **0.3767**、wrist **0.3229**，约为已抽中 RLDS 的 **22.9× / 7.0×**；融合 projector MSE 分别为 **0.546/0.406**，高于 held-out RLDS 的 **0.150/0.150**。域差在 Adapter 边界已出现，不能全归因于后半塔；但后半塔还额外受到 DINO prefix 丢失的结构性误差。simulator 图像来自 fresh reset 静态观测，不代表完整 rollout 外观分布，且各组每视角仅 16–20 张，以上比值只是本固定样本集的描述统计。

## 判断与最低成本下一步

**优先级：实现问题（DINO prefix 缺失）＞ simulator-domain 泛化 ＞ 固定采样覆盖。** 这是对当前故障排查顺序的判断，不是对 1/5 成功率的单因果证明。只用真实 patch token 的 oracle 都不能复现 Object-Pro，而补齐真实 prefix 后最终 DINO MSE 从约 1.5–2.0 降至约 0.001–0.002，说明在现有模型结构下直接加训练图像/epoch 无法消除这一确定的上下文缺口；simulator 的边界域差也确实独立存在，需在结构问题厘清后处理。

成本最低的一项后续实验：**在本报告同一 40 张独立 simulator 图像上，保持已训练 Adapter patch tokens 不变，仅离线注入完整 Object-Pro 的真实 DINO prefix tokens**，重新计算最终层、projector 及上述三对静态动作误差。它不训练、不跑环境、也不使用正式评测 states；可以直接量化“只补 prefix”能挽回多少 student 误差。若改善很小，先解决 simulator-domain 边界对齐；若改善显著，再设计可部署的 prefix-aware Adapter 及严格分离数据的重训协议。该真实 prefix oracle 本身依赖原图视觉塔，**不能**作为压缩模型的正式性能结果。
