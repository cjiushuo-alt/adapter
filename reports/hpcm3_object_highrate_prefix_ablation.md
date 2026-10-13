# Object 高码率 HPCM3：仅补 DINO prefix 的离线消融

日期：2026-10-13；分支：`exp/hpcm3-object-dist`。结论：**真实 CLS + 4 个 register tokens 确实改善了当前 Adapter 的 DINO 后半塔输出，但对 projector 的收益只有约 3–4%；三对静态动作的平均误差下降约 16%，仍明显偏离 Object-Pro。** 因而 prefix 缺失是已证实的结构问题，却不足以单独解释现有高码率 HPCM3 闭环仅 1/5 的现象。此结果支持把 prefix 纳入后续设计，但**不支持立即只为 prefix 重训一版并预期闭环恢复**；应同时处理 simulator 域的 patch/SigLIP 对齐误差。真实 prefix 取自原始 Object-Pro，是不可部署的诊断性 oracle。

## 固定协议与实现核对

沿用[前次离线诊断](hpcm3_object_highrate_offline_diagnostic.md)的独立 fresh-reset simulator manifest：10 个 Object task × 2 次 reset × agent/wrist 双视角，共 40 张，其中 agent/wrist 各 20 张。manifest 记录与正式评测 initial-state 的 SHA256 碰撞数为 **0**；本次只读取已保存图像，**没有**访问正式评测状态、调用环境 `step()`、训练或修改任何权重。全部 40 张 RGB SHA256 与前次逐张一致；无 prefix 的 Student MSE 与前次结果最大绝对差为 **0**。seed 为 `20261013`。动作诊断使用其中 `(task 0, reset 0)`、`(task 0, reset 1)`、`(task 1, reset 0)` 三对静态双视角图像，相同任务文本与全零合成 proprio；动作头输出为 `8×7` chunk，**不是** rollout 成功率。

对照组保留现有 HPCM→Adapter 预测的 **256 个 DINO patch tokens** 和 SigLIP 分支。干预组只在 DINO block-2 与后半塔之间，将同一 RGB 的原始 Object-Pro `get_intermediate_layers(..., return_prefix_tokens=True)` 所得 5 个 token 按 **`[CLS, register×4, raster-order patch×256]`** 拼成 261 个 token；运行原有截断塔保留的 block 3 及后续 blocks，并同原实现一样省略最后一个 block，再仅取 256 个 patch 输出，继续原融合/projector/动作头。DINO patch 边界、SigLIP 边界与最终 SigLIP 特征在两组之间经逐元素相等断言；仅 DINO attention 上下文改变。teacher 边界是 BF16，Adapter/后半塔权重为 FP16；prefix 与预测 patch 在后半塔前均转 FP32，后半塔以 FP32 执行，最终输出转 BF16。故核对的是**注入位置、token 顺序和执行逻辑的结构一致性**，不是声称与完整 BF16 原塔位级一致。原始 HPCM 输入的 CPU uint8→FP32→`[-1,1]` 路径、Object-Pro 原生预处理及权重不变；wrapper 与直接模型在同 RGB 上最大差 0，重复前向最大差 0，融合顺序为 DINO 在前、SigLIP 在后。

脚本：[离线诊断代码](../Align_adapter-HPCM3/code/analysis/diagnose_object_highrate_offline.py)；[配置](../Align_adapter-HPCM3/code/configs/diagnose_object_highrate_prefix_ablation.yaml)；[逐图、逐 token 指标及动作 JSON](../results/hpcm3_object_highrate_prefix_ablation/diagnostic.json)。执行：`HF_HUB_OFFLINE=1 python Align_adapter-HPCM3/code/analysis/diagnose_object_highrate_offline.py --config Align_adapter-HPCM3/code/configs/diagnose_object_highrate_prefix_ablation.yaml`。输出目录已存在时脚本拒绝覆盖。

| 固定输入/记录 | SHA256 |
| --- | --- |
| HPCM `0.0483.pth.tar` | `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe` |
| 最佳 Object Adapter `adapter-best.pt` | `422c8216e4fbb91a9e3775148a2e43b9486cbb76dd530b601a6063ca309d0fb2` |
| Object-Pro `model.safetensors` | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| action head / proprio projector | `dd5642994a89f81dfe33cdca460479972548de8eaa26aacdbb3b0a7516b2d723` / `3201fd9f18cb178958d656ac2ddd7163f8a7853535856243907f5b633af7c14c` |
| 独立 simulator manifest / 前次诊断 JSON | `8784c72a8337878ab582e26b95e5fa4dfa098cd3c6d58409976e60c3af2b99c7` / `8b441e839dd0b1ee0263471c7629d18f5a50b7e4f38a984f8895b7b00282f37e` |
| 本次配置 / 结果 JSON | `177f88c948d45d9433b57f3422b4be802f66ffe0c15598509b2e7f30c151069f` / `4549393222ef7f3337b456997fcb12d488d15d974e5977c662a44aa62904e236` |

## 40 张图像：相同样本的配对特征误差

表中为相对于完整 Object-Pro 同 RGB 输出的 token MSE；`↓` 为只补真实 DINO prefix 后的相对下降。每行每视角 `n=20`。最终融合特征为 DINO/SigLIP 拼接，projector 是其下游输出。

| 视角 | block-2 DINO | 最终 DINO | 最终 SigLIP | 最终融合 | Projector |
| --- | ---: | ---: | ---: | ---: | ---: |
| agent，原 Student | 0.06253 | 5.43905 | 4.11239 | 4.73670 | 0.54648 |
| agent，补 prefix | **0.06253** | **4.46994 (↓17.8%)** | **4.11239** | **4.28065 (↓9.6%)** | **0.53002 (↓3.0%)** |
| wrist，原 Student | 0.05240 | 3.28428 | 4.41347 | 3.88209 | 0.40574 |
| wrist，补 prefix | **0.05240** | **2.38324 (↓27.4%)** | **4.41347** | **3.45807 (↓10.9%)** | **0.39009 (↓3.9%)** |

最终 DINO token cosine：agent **0.8131→0.8404**、wrist **0.8989→0.9239**；projector token cosine：agent **0.9095→0.9164**、wrist **0.9259→0.9304**。最终 DINO、融合与 projector 的 MSE 均在各视角 **20/20** 张上下降，说明这不是单帧偶然现象。按图像配对 bootstrap（20,000 次，固定 seed `20261013`）的平均 MSE 降幅 95% 区间分别为：agent DINO `0.969 [0.946, 0.993]`、projector `0.0165 [0.0149, 0.0180]`；wrist DINO `0.901 [0.870, 0.928]`、projector `0.0156 [0.0133, 0.0180]`。这些只是当前 20 张/视角静态样本内的不确定度，不能外推为全 rollout 分布或任务成功率。

## 三对静态图像：动作预测误差

下表为与原始 Object-Pro 动作 chunk 的 MAE；相同 RGB、任务文本、零 proprio、action head 和归一化路径，仅切换 prefix。完整 JSON 还保存每对的最大绝对差及此前的 patch oracle 对照。

| task / reset | 原 Student | 补 prefix | MAE 降幅 |
| --- | ---: | ---: | ---: |
| 0 / 0 | 0.02076 | 0.01607 | 22.6% |
| 0 / 1 | 0.01462 | 0.01192 | 18.5% |
| 1 / 0 | 0.02900 | 0.02607 | 10.1% |
| **三对平均** | **0.02146** | **0.01802** | **16.0%** |

三对 MAE 都改善，但第三对最大绝对误差从 `0.10627` 略升至 `0.10666`；不能宣称所有动作分量都改善。三对静态动作更不足以预测闭环成功率。

## 判断

前次报告中“真实 patch + 真实 prefix”的 oracle 几乎复现完整 Object-Pro，定位了**缺失 prefix 的实现非等价**。本次更严格地**保持真实 Adapter 预测 patch 不变**：虽然 DINO 后半塔误差稳定下降，projector 只下降约 3–4%，静态动作仍有较大残差。真实 prefix 无法在压缩推理中直接取得；若训练 prefix-aware HPCM3，还必须预测它，效果不能等同本次 oracle。与此同时，block-2 DINO MSE 保持 `0.052–0.063`、最终 SigLIP MSE 保持 `4.11–4.41`，说明 simulator-domain 边界对齐和另一视觉塔的误差仍在。

因此，**不建议仅因本次结果就单独投入 prefix-aware HPCM3 重训并把它视为恢复闭环的充分方案**。值得在后续方案中让 DINO prefix 可预测、维持正确 token 顺序，但应先用独立 simulator 数据对 patch/SigLIP 域差做低成本离线验证，并用保护的评测状态做最终闭环确认。此诊断没有用正式状态调参，也没有产生新的成功率结果。
