# Object 高码率 HPCM3：边界蒸馏训练与受保护闭环先导

日期：2026-10-13；分支：`exp/hpcm3-object-dist`。**结果：Object task 0 的五个配对初始状态上，原始 Object-Pro 为 5/5，高码率 HPCM 重建 RGB→原始 Object-Pro 为 5/5，新训练的高码率 HPCM3→Object-Pro 为 1/5。** 这是 task 0 先导而非完整 LIBERO-Object suite 评测；当前不能作为论文第二个正向主结果，亦不宜直接扩展为正式 500-episode 评测。

## 固定协议与数据

- HPCM 使用冻结的 `HPCM/ckpt/0.0483.pth.tar`。Adapter 从随机初始化训练，**没有**加载低码率 Epoch 29 Adapter。结构与最初 Spatial 成功方案一致：2 个 ResBlock、双塔 ViT block-2 截断边界，损失 `SigLIP_MSE + 5 × DINO_MSE`；没有 full-visual loss、额外架构或多组 loss 搜索。Object-Pro 的视觉后半塔、projector、VLA 和动作头均保持原权重。[训练配置](../Align_adapter-HPCM3/code/configs/train_object_highrate_boundary.yaml)和[评测配置](../Align_adapter-HPCM3/code/configs/eval_object_highrate_boundary_task0_ep5.yaml)给出全部参数。
- 训练数据为已提取的 `Align_adapter/dataset/libero_object_no_noops` 双视角 RGB。按 **trajectory** 划分 408 train / 46 val，两个轨迹集合无交集；随机种子 42。训练轨迹共有 120,552 张双视角图像，预先固定的确定性 cap 选取 16,384 张用于优化，覆盖全部 408 条训练轨迹；验证取 held-out 轨迹的 512 张。batch 16、AdamW 1e-4、cosine 调度、FP16。该 cap 是首个完整 epoch 之前为成本控制固定的，不是按验证或闭环结果调节。训练 split manifest SHA256 `378be2671d58e3e508e6761e2ea83f42fce8f48a6a63af22ecec19ef7d413855`，原件保存在机械盘。
- HPCM 输入在训练与推理均为 simulator / 数据集解码后的原始 256×256 RGB，归一化至 `[-1,1]`，推理启用 `hpcm_raw_input_bypass=true`。Teacher 在训练时复现 Object-Pro 的策略预处理：JPEG/Lanczos3 到 224px、0.9 中心裁剪及原生 processor。两张 held-out 图像的训练 HPCM latent 与独立推理实例逐元素最大差 **0**，Object-Pro 整塔与截断 teacher 的 block-2 特征逐元素最大差 **0**；[预检脚本](../Align_adapter-HPCM3/code/analysis/verify_object_highrate_protocol.py)产物 SHA256 `a4af32cd28faa463493489957d57c9959124c64b7e461575661e3250a1ff5ca1`。这是少量图像的数值一致性检验，不等于全分布证明。
- 独立 simulator-domain 验证使用所有 10 个 Object task 各 2 次 fresh `env.reset()`、双视角共 40 张图像；seed 20261012 起，各 task 独立递增。与正式评测的默认 initial-state 数组做 state-hash 拒绝检查，**0 次碰撞**；与全部训练轨迹图像的 RGB SHA256 比较，**0 次精确帧重合**。这些图像没有用于梯度更新、调参或选模；只作诊断。manifest SHA256 `8784c72a8337878ab582e26b95e5fa4dfa098cd3c6d58409976e60c3af2b99c7`，原图/manifest 留在机械盘，未上传 Git。
- 曾有两次在训练步开始前分别因 HF 镜像 TLS、W&B 在线初始化超时而退出；均未产生可用权重。最终实验使用本地模型缓存及 **W&B offline**，run ID `4pcqaalw`，记录在机械盘，尚未声称云端已同步。为保证验证集早停，从第 39 轮完整 optimizer/scheduler 状态续跑一次，将安全上限 40 提高到 50；在第 40 轮结束时早停，实际 **41 个 epoch（0–40）**，非触及新上限。最佳权重由 held-out clean validation 的 `val_loss` 选出，为 epoch 39、step 40,960，精确 loss **0.0455577448**。early stopping 参数为 `min_delta=1e-4, patience=5`。模型和完整训练 checkpoint 仅存机械盘。

## 离线蒸馏与 simulator-domain 验证

下表均为 block-2 边界加权损失；`sim` 为独立 fresh-reset 图像，**不是**正式评测观测。完整 41 轮、agent/wrist 及 DINO/SigLIP 分项见[训练曲线](../results/hpcm3_object_highrate_boundary/training_curve.jsonl)。

| Epoch | Train | Clean val | Sim agent | Sim wrist | Sim all | Sim SigLIP MSE | Sim DINO MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.0298 | 0.3856 | 1.0414 | 1.1302 | 1.0858 | 0.5453 | 0.1081 |
| 10 | 0.0540 | 0.0611 | 0.5760 | 0.4565 | 0.5163 | 0.1045 | 0.0824 |
| 29 | 0.0318 | 0.0463 | 0.3757 | 0.3026 | 0.3392 | 0.0634 | 0.0551 |
| **39（选中）** | **0.0299** | **0.0456** | **0.3765** | **0.3228** | **0.3497** | **0.0624** | **0.0575** |
| 40（早停） | 0.0299 | 0.0456 | 0.3769 | 0.3220 | 0.3495 | 0.0620 | 0.0575 |

Simulator 对齐从 epoch 0 到 39 明显改善（1.0858→0.3497），但 epoch 29→39 **没有继续改善**（0.3392→0.3497），且最佳 checkpoint 的 simulator/clean 加权 loss 约 **7.7×**。跨域样本和 clean val 的采样协议不同，不能把此比值解释成因果机制；它与闭环失败相容。我们严格按预设的 clean val 选模，**没有**用 simulator 指标改选 epoch 29。

## 受保护的 Object task 0 五状态评测

三组使用相同 task 描述“pick up the alphabet soup and place it in the basket”、seed 7、默认 initial states 0–4、10 个 dummy steps、8-action queue、双视角和 `libero_object_no_noops` 动作反归一化。A 与 B 复用[历史原始 Object-Pro 记录](../results/object_pro_intact_smoke/episode_results.jsonl)及[真实高码率 bitstream 重建记录](../results/hpcm3_suite_highrate_screen/object_task0_ep20/episode_results.jsonl)的前五条；其 initial-state SHA256 与本轮 C 逐条相同。C 是唯一新运行的闭环 arm：原始 RGB→HPCM latent→本轮 Adapter→Object-Pro 后半视觉塔和策略，**不**把重建 RGB 输入 C 策略。[评测脚本](../Align_adapter-HPCM3/code/analysis/eval_object_highrate_boundary_task0.py)同时以 `compress/decompress` 的真实 y/z bitstream 在 C 的每次策略查询图像上计码率，但重建图仅供计量。A/B/C 因动作不同，后续 RGB 轨迹并非逐帧配对；下表只是相同初始状态的 episode 配对。

| Episode / default initial state | A 原始 RGB | B 高码率重建 RGB | C 新 HPCM3 | C agent / wrist 实际 bpp |
| ---: | :---: | :---: | :---: | ---: |
| 0 | ✓ | ✓ | ✗ | 0.5731 / 0.3602 |
| 1 | ✓ | ✓ | ✗ | 0.5745 / 0.3759 |
| 2 | ✓ | ✓ | ✓ | 0.5478 / 0.4861 |
| 3 | ✓ | ✓ | ✗ | 0.5730 / 0.3718 |
| 4 | ✓ | ✓ | ✗ | 0.5682 / 0.3636 |
| **合计** | **5/5** | **5/5** | **1/5** | **查询帧均值 0.5699 / 0.3793** |

C 的五条 episode 共 155 次策略查询、310 张双视角图像；针对训练轨迹全部 **120,552** 个 RGB SHA256 的 exact-frame 审计结果为 **0**。详见[summary](../results/hpcm3_object_highrate_boundary/task0_ep5/summary.json)、[逐 episode](../results/hpcm3_object_highrate_boundary/task0_ep5/episode_results.jsonl)、[帧 SHA 审计](../results/hpcm3_object_highrate_boundary/task0_ep5/audit.jsonl)及[逐帧真实码率](../results/hpcm3_object_highrate_boundary/task0_ep5/rate.jsonl)。A/B 的 5/5 是历史结果，不是本轮重跑；B 的平均 bpp 与 C 不应直接比较，因为成功/失败引起的闭环轨迹长度不同。

| 文件/权重 | SHA256 |
| --- | --- |
| HPCM `0.0483.pth.tar` | `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe` |
| 最佳 Adapter `adapter-best.pt` | `422c8216e4fbb91a9e3775148a2e43b9486cbb76dd530b601a6063ca309d0fb2` |
| 最佳完整训练状态 `epoch=39, step=40960` | `ec399315a41addd33eef3e4e6979662acc7d8116a748a4fce28684fdd0e92ee0` |
| Object-Pro `model.safetensors` | `c83c07b9c1d4b527ba5d0f821b28c340c745f4aec282a4953511d494c536d604` |
| Object-Pro action head / proprio projector | `dd5642994a89f81dfe33cdca460479972548de8eaa26aacdbb3b0a7516b2d723` / `3201fd9f18cb178958d656ac2ddd7163f8a7853535856243907f5b633af7c14c` |
| 完整训练轨迹帧 SHA 缓存 | `c69f8763b199a7ffedc632ab0cc93e6dacc6b11088e5a7c3f6cb2add1fb93dc2` |

评测进程启动时的 Git commit 为 `2ad24a60db37ee1c6e5647bfc26752d80bcc1c33`；后续仅补充曲线导出和本报告，未改变评测权重。run 的小文件提交 Git；126 MB Adapter、约 1 GB 训练状态、W&B 本地二进制、数据及 simulator 原图均留在机械盘，不提交。

## 判断与下一步

高码率 HPCM **重建 RGB** 能维持本 task 的 Object-Pro 5/5，说明在该控制下压缩信息足以支持策略；**不能**据此推定 Adapter 特征已足够。新的边界蒸馏虽使 clean 与独立 simulator 特征误差明显下降，却只得到 C 1/5；当前主要问题是 latent→策略特征的闭环可用性/跨域泛化，而非需要立即加深 Adapter 或继续堆 epoch。五状态样本太少，不能断言整个 Object suite 成败率。

目前 RLDS 提取图像缺少可与正式 default initial states 直接等值比较的完整 simulator state，故**不能证明训练轨迹和评测 initial state 完全不重合**；已确认的是 trajectory-level train/val 分离、独立 simulator 验证与正式 state-hash 不碰撞、评测帧与全部训练轨迹 RGB 精确哈希无重合。正式论文使用前，应取得原始 demo 的完整 simulator states，补做 same-initial-state 审计，并在冻结协议后跑更多 task/initial states。

**不建议将本 checkpoint 作为第二个正向论文结果。** 本轮预设“先 5 个受保护 episode；若正向再扩展”的门槛未达到，因此没有启动 20-episode 或 suite-wide 测试，也没有根据这五条结果重新选模或训练。下一步若获授权，优先在**与正式评测 state 严格分离**的独立 simulator 图像上分析/缩小特征域差（尤其 agent 视角和视觉后半塔的误差传播），明确新的训练/选模协议后再做一次预注册式验证；不能把这五个正式探针状态反馈用于微调。
