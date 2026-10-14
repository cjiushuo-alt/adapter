"""Publish small 10x50 evaluation outputs and a scope-correct Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
DEST = REPO / "results/hpcm3_object_highrate_libero4_full_eval"
REPORT = REPO / "reports/hpcm3_object_highrate_libero4_full_500.md"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.eval_dir
    summary = json.loads((source / "summary.json").read_text())
    episodes = records(source / "episode_results.jsonl")
    rates = records(source / "rate.jsonl")
    tasks = records(source / "task_results.jsonl")
    if summary["total_episodes"] != 500 or len(episodes) != 500 or len(tasks) != 10:
        raise ValueError("Incomplete formal 10x50 result")
    if len(rates) != summary["compressed_views"] or summary["entropy_latent_checked_views"] != 20:
        raise ValueError("Incomplete bitstream/latent audit")
    if sum(row["success"] for row in episodes) != summary["success_episodes"]:
        raise ValueError("Episode/summary success mismatch")
    if sum(row["exact_train_frame_matches"] for row in episodes) != summary["exact_train_frame_matches"]:
        raise ValueError("Exact-frame audit mismatch")
    for task_id in range(10):
        subset = [row for row in episodes if row["task_id"] == task_id]
        if len(subset) != 50 or [row["episode_index"] for row in subset] != list(range(50)):
            raise ValueError(f"Task {task_id} is not 50 ordered episodes")

    a_path = REPO / "results/object_pro_intact_smoke/episode_results.jsonl"
    b_path = REPO / "results/hpcm3_suite_highrate_screen/object_task0_ep20/episode_results.jsonl"
    a, b = records(a_path), records(b_path)
    task0 = [row for row in episodes if row["task_id"] == 0]
    if len(a) != 5 or len(b) != 20:
        raise ValueError("Historical baseline scope changed")
    for index, row in enumerate(a):
        if row["initial_state_sha256"] != task0[index]["initial_state_sha256"]:
            raise ValueError("Object-Pro baseline state mismatch")
    for index, row in enumerate(b):
        if row["initial_state_sha256"] != task0[index]["initial_state_sha256"]:
            raise ValueError("Reconstruction baseline state mismatch")
    if not all(row["A_success"] is True for row in b[:5]):
        raise ValueError("Historical B baseline A-control changed")
    if summary["checkpoints"]["adapter"]["sha256"] != \
       "2ba5ee6a9500a812ef58b8c3080744aa0b5700b641106219b72fd2d949ad01d5":
        raise ValueError("Wrong frozen Object Adapter")

    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "provenance.json", "summary.json", "episode_results.jsonl",
                 "task_results.jsonl", "rate.jsonl"):
        shutil.copy2(source / name, DEST / name)
    audit_path = source / "audit.jsonl"
    audit_sha = sha(audit_path)
    (DEST / "large_audit_provenance.json").write_text(json.dumps({
        "path": str(audit_path), "sha256": audit_sha, "bytes": audit_path.stat().st_size,
        "note": "Complete frame-by-frame JSONL retained on HDD; not uploaded because it is large",
    }, indent=2) + "\n")

    table = []
    for task in tasks:
        task_id = task["task_id"]
        subset = [row for row in episodes if row["task_id"] == task_id]
        agent = sum(row["bpp_actual_mean"]["agent"] for row in subset) / len(subset)
        wrist = sum(row["bpp_actual_mean"]["wrist"] for row in subset) / len(subset)
        table.append(f"| {task_id} | {task['task_description']} | {task['success_episodes']}/50 | "
                     f"{task['success_rate']:.1%} | {50-task['success_episodes']} | "
                     f"{agent:.4f} / {wrist:.4f} | {task['exact_train_frame_matches']} |")
    a_success = sum(row["success"] for row in a)
    b_success = sum(row["B_success"] for row in b)
    c_first5 = sum(row["success"] for row in task0[:5])
    c_first20 = sum(row["success"] for row in task0[:20])
    failures = sum(row["failure_type"] == "horizon_without_success" for row in episodes)
    contents = f"""# Object 高码率 HPCM3：LIBERO-Object 全套 500-episode 闭环评测

冻结四套 LIBERO 全量图像训练的 Object HPCM3 Adapter（SHA256 `{summary['checkpoints']['adapter']['sha256']}`），按官方 `DEFAULT` 初始状态对 LIBERO-Object 全部 10 个任务各运行 50 个 episode。总体 **{summary['success_episodes']}/500 = {summary['success_rate']:.1%}**；本阶段未训练、改动架构/loss/权重或按结果选模。完整[配置](../results/hpcm3_object_highrate_libero4_full_eval/config.yaml)、[逐 episode 结果](../results/hpcm3_object_highrate_libero4_full_eval/episode_results.jsonl)、[逐查询真实码率](../results/hpcm3_object_highrate_libero4_full_eval/rate.jsonl)与[汇总 JSON](../results/hpcm3_object_highrate_libero4_full_eval/summary.json)已保存。

## 协议和结果

Object-Pro 权重和动作头不变，raw 256×256 双相机 RGB 经 HPCM→Adapter→Object-Pro 后半视觉塔；`hpcm_raw_input_bypass=true`，Object-Pro 原生 processor，seed 7、每次起始状态前 10 个 dummy steps、8-action queue、`libero_object_no_noops`，不保存 rollout 视频。失败被定义为达到任务 horizon 未成功；运行中若发生模型/环境异常，脚本会报错而非算作策略失败。本轮记录 **{failures}** 个未成功 episode，**{summary['exact_train_frame_matches']}** 个 exact train-frame matches。逐帧 SHA audit 包含失败轨迹。单个任务的 50 状态结果如下：

| Task | 描述 | Success | Rate | Failures | Agent / wrist bpp¹ | Exact train-frame matches |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}
| **总体** | 10 tasks | **{summary['success_episodes']}/500** | **{summary['success_rate']:.1%}** | **{500-summary['success_episodes']}** | **{summary['mean_actual_bpp']['agent']:.4f} / {summary['mean_actual_bpp']['wrist']:.4f}** | **{summary['exact_train_frame_matches']}** |

¹ 任务行是 episode 内查询均值再按 50 episode 平均；总体行是所有真实 policy-query bitstream 的加权均值，两者权重不同。码率用真实 y/z 熵编码字节数计算，`bpp = 8×(y_bytes+z_bytes)/(256×256)`，不是 entropy model 的估计值；只统计策略实际查询的 agent/wrist 图像。共 **{summary['compressed_views']}** 张压缩视图。完整[逐 task JSONL](../results/hpcm3_object_highrate_libero4_full_eval/task_results.jsonl)与[逐 episode JSONL](../results/hpcm3_object_highrate_libero4_full_eval/episode_results.jsonl)可复核每个失败和 initial-state SHA256。未保存视频，故 `horizon_without_success` 不能再细分为抓取、识别或放置错误。

## 与现有基线的正确比较范围

| 路径 | 可核对的范围 | 结果 | 与本轮配对检查 |
| --- | --- | ---: | --- |
| 原始 RGB→Object-Pro | task 0，DEFAULT states 0–4 | {a_success}/5 | state SHA 5/5 相同；本轮 {c_first5}/5 |
| 高码率真实重建 RGB→Object-Pro | task 0，DEFAULT states 0–19 | {b_success}/20 | state SHA 20/20 相同；本轮 {c_first20}/20 |
| 本轮 HPCM3→Object-Pro | 全部 task 0–9，各 DEFAULT states 0–49 | {summary['success_episodes']}/500 | 此行为完整 suite 结果 |

前两组来自既有记录，[原始 Object-Pro](../results/object_pro_intact_smoke/episode_results.jsonl)和[高码率重建控制](../results/hpcm3_suite_highrate_screen/object_task0_ep20/episode_results.jsonl)；**没有**可复用的、同 Object-Pro 权重/高码率/同协议的 500-episode A 或 B 基线，不能把 5/5、20/20 当作 500-episode 对照成功率。仓库中另一个 500-episode Object 结果用了不同 VLA 策略和 λ=0.0018 HPCM，也不纳入本表。因策略不同，三组除 initial states 外的后续观测轨迹不逐帧配对。

## bitstream latent 与泄漏边界

每次查询的 bpp 对 raw RGB 进行真正 `compress()`；在 8 张**非正式评测** fresh-reset 图像上，HPCM3 forward 的 320×16×16 latent 与 bitstream `decompress()` 送进 decoder 的 latent **8/8 逐元素相等，max abs=0**，见[独立预检](../results/hpcm3_object_highrate_libero4_full_eval/entropy_latent_parity.json)。本轮还对每个 task 的首个 episode、首个查询双视图做相同检查，共 **{summary['entropy_latent_checked_views']}** 张，最大差 **{summary['entropy_latent_max_abs']:.6g}**；**没有**逐张验证所有 {summary['compressed_views']} 个 latent，因此不能把采样核对表述为全帧定理。码率统计输入使用与 HPCM3 raw bypass 相同的 CPU FP32 归一化路径。

训练前的[数据审计](hpcm3_object_highrate_libero4_full.md)确认四套数据按 trajectory 分离，训练 **493,840** 张 RGB 与 held-out 验证图像无 exact-frame 重合；本轮逐帧 audit 对正式 rollout 与全部训练图像的 SHA256 比较结果见上表。但 RLDS PNG 导出没有 demo 的完整 simulator initial state，**无法证明训练 demonstration 与这 500 个评测 initial states 不重合**。`same task/environment` 本身不算泄漏，`same initial state/demo` 若存在则是论文正式结论风险。正式 state 没有用于本轮训练、调参或选模；这套状态曾用于先导评测，亦不是全新盲测。

完整逐帧 audit JSONL 留在机械盘 `{audit_path}`，SHA256 `{audit_sha}`，大小 {audit_path.stat().st_size} bytes；因体积较大未上传 Git，索引见[审计文件信息](../results/hpcm3_object_highrate_libero4_full_eval/large_audit_provenance.json)。模型 SHA、action head、proprio projector、评测启动 commit 和训练哈希缓存 SHA 均见[provenance](../results/hpcm3_object_highrate_libero4_full_eval/provenance.json)。运行耗时 {summary['elapsed_seconds']/3600:.2f} 小时。
"""
    REPORT.write_text(contents)
    print(f"Wrote {REPORT} and small results in {DEST}", flush=True)


if __name__ == "__main__":
    main()
