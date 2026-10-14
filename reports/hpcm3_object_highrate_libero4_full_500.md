# Object 高码率 HPCM3：LIBERO-Object 全套 500-episode 闭环评测

冻结四套 LIBERO 全量图像训练的 Object HPCM3 Adapter（SHA256 `2ba5ee6a9500a812ef58b8c3080744aa0b5700b641106219b72fd2d949ad01d5`），按官方 `DEFAULT` 初始状态对 LIBERO-Object 全部 10 个任务各运行 50 个 episode。总体 **472/500 = 94.4%**；本阶段未训练、改动架构/loss/权重或按结果选模。完整[配置](../results/hpcm3_object_highrate_libero4_full_eval/config.yaml)、[逐 episode 结果](../results/hpcm3_object_highrate_libero4_full_eval/episode_results.jsonl)、[逐查询真实码率](../results/hpcm3_object_highrate_libero4_full_eval/rate.jsonl)与[汇总 JSON](../results/hpcm3_object_highrate_libero4_full_eval/summary.json)已保存。

## 协议和结果

Object-Pro 权重和动作头不变，raw 256×256 双相机 RGB 经 HPCM→Adapter→Object-Pro 后半视觉塔；`hpcm_raw_input_bypass=true`，Object-Pro 原生 processor，seed 7、每次起始状态前 10 个 dummy steps、8-action queue、`libero_object_no_noops`，不保存 rollout 视频。失败被定义为达到任务 horizon 未成功；运行中若发生模型/环境异常，脚本会报错而非算作策略失败。本轮记录 **28** 个未成功 episode，**0** 个 exact train-frame matches。逐帧 SHA audit 包含失败轨迹。单个任务的 50 状态结果如下：

| Task | 描述 | Success | Rate | Failures | Agent / wrist bpp¹ | Exact train-frame matches |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | pick up the alphabet soup and place it in the basket | 40/50 | 80.0% | 10 | 0.5579 / 0.4714 | 0 |
| 1 | pick up the cream cheese and place it in the basket | 46/50 | 92.0% | 4 | 0.5546 / 0.5026 | 0 |
| 2 | pick up the salad dressing and place it in the basket | 50/50 | 100.0% | 0 | 0.5569 / 0.5262 | 0 |
| 3 | pick up the bbq sauce and place it in the basket | 45/50 | 90.0% | 5 | 0.5422 / 0.4722 | 0 |
| 4 | pick up the ketchup and place it in the basket | 47/50 | 94.0% | 3 | 0.5550 / 0.5006 | 0 |
| 5 | pick up the tomato sauce and place it in the basket | 46/50 | 92.0% | 4 | 0.5573 / 0.5189 | 0 |
| 6 | pick up the butter and place it in the basket | 49/50 | 98.0% | 1 | 0.5569 / 0.5191 | 0 |
| 7 | pick up the milk and place it in the basket | 49/50 | 98.0% | 1 | 0.5541 / 0.5388 | 0 |
| 8 | pick up the chocolate pudding and place it in the basket | 50/50 | 100.0% | 0 | 0.5596 / 0.5168 | 0 |
| 9 | pick up the orange juice and place it in the basket | 50/50 | 100.0% | 0 | 0.5522 / 0.5201 | 0 |
| **总体** | 10 tasks | **472/500** | **94.4%** | **28** | **0.5553 / 0.5020** | **0** |

¹ 任务行是 episode 内查询均值再按 50 episode 平均；总体行是所有真实 policy-query bitstream 的加权均值，两者权重不同。码率用真实 y/z 熵编码字节数计算，`bpp = 8×(y_bytes+z_bytes)/(256×256)`，不是 entropy model 的估计值；只统计策略实际查询的 agent/wrist 图像。共 **18980** 张压缩视图。完整[逐 task JSONL](../results/hpcm3_object_highrate_libero4_full_eval/task_results.jsonl)与[逐 episode JSONL](../results/hpcm3_object_highrate_libero4_full_eval/episode_results.jsonl)可复核每个失败和 initial-state SHA256。未保存视频，故 `horizon_without_success` 不能再细分为抓取、识别或放置错误。

## 与现有基线的正确比较范围

| 路径 | 可核对的范围 | 结果 | 与本轮配对检查 |
| --- | --- | ---: | --- |
| 原始 RGB→Object-Pro | task 0，DEFAULT states 0–4 | 5/5 | state SHA 5/5 相同；本轮 5/5 |
| 高码率真实重建 RGB→Object-Pro | task 0，DEFAULT states 0–19 | 20/20 | state SHA 20/20 相同；本轮 16/20 |
| 本轮 HPCM3→Object-Pro | 全部 task 0–9，各 DEFAULT states 0–49 | 472/500 | 此行为完整 suite 结果 |

前两组来自既有记录，[原始 Object-Pro](../results/object_pro_intact_smoke/episode_results.jsonl)和[高码率重建控制](../results/hpcm3_suite_highrate_screen/object_task0_ep20/episode_results.jsonl)；**没有**可复用的、同 Object-Pro 权重/高码率/同协议的 500-episode A 或 B 基线，不能把 5/5、20/20 当作 500-episode 对照成功率。仓库中另一个 500-episode Object 结果用了不同 VLA 策略和 λ=0.0018 HPCM，也不纳入本表。因策略不同，三组除 initial states 外的后续观测轨迹不逐帧配对。

## bitstream latent 与泄漏边界

每次查询的 bpp 对 raw RGB 进行真正 `compress()`；在 8 张**非正式评测** fresh-reset 图像上，HPCM3 forward 的 320×16×16 latent 与 bitstream `decompress()` 送进 decoder 的 latent **8/8 逐元素相等，max abs=0**，见[独立预检](../results/hpcm3_object_highrate_libero4_full_eval/entropy_latent_parity.json)。本轮还对每个 task 的首个 episode、首个查询双视图做相同检查，共 **20** 张，最大差 **0**；**没有**逐张验证所有 18980 个 latent，因此不能把采样核对表述为全帧定理。码率统计输入使用与 HPCM3 raw bypass 相同的 CPU FP32 归一化路径。

训练前的[数据审计](hpcm3_object_highrate_libero4_full.md)确认四套数据按 trajectory 分离，训练 **493,840** 张 RGB 与 held-out 验证图像无 exact-frame 重合；本轮逐帧 audit 对正式 rollout 与全部训练图像的 SHA256 比较结果见上表。但 RLDS PNG 导出没有 demo 的完整 simulator initial state，**无法证明训练 demonstration 与这 500 个评测 initial states 不重合**。`same task/environment` 本身不算泄漏，`same initial state/demo` 若存在则是论文正式结论风险。正式 state 没有用于本轮训练、调参或选模；这套状态曾用于先导评测，亦不是全新盲测。

完整逐帧 audit JSONL 留在机械盘 `/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_libero4_full/full_eval/audit.jsonl`，SHA256 `2e37d359cec804bf970aa628e3cc5f5a6734d97411f81438c9caa3d94eda004f`，大小 38462298 bytes；因体积较大未上传 Git，索引见[审计文件信息](../results/hpcm3_object_highrate_libero4_full_eval/large_audit_provenance.json)。模型 SHA、action head、proprio projector、评测启动 commit 和训练哈希缓存 SHA 均见[provenance](../results/hpcm3_object_highrate_libero4_full_eval/provenance.json)。运行耗时 1.52 小时。
