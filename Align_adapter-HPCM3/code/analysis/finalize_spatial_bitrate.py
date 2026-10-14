"""Validate and publish small Spatial bitrate artifacts without merging protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results/hpcm3_spatial_bitrate"
REPORT = REPO / "reports/hpcm3_spatial_bitrate.md"
HDD = Path("/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full")
ARMS = (("0.0018", "reconstruction"), ("0.0483", "reconstruction"),
        ("0.0483", "hpcm3"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HDD)
    args = parser.parse_args()
    root = args.output_root
    if ROOT.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite published Spatial bitrate results")
    results = {}
    reference = records(REPO / "results/hpcm3_libero4/spatial/episode_results.jsonl")
    reference_summary = json.loads((REPO / "results/hpcm3_libero4/spatial/summary.json").read_text())
    if len(reference) != 500 or reference_summary["success_episodes"] != 491:
        raise ValueError("Historical Spatial result is incomplete or changed")
    for lam, arm in ARMS:
        source = root / "eval" / f"lambda_{lam}_{arm}"
        summary = json.loads((source / "summary.json").read_text())
        episodes = records(source / "episode_results.jsonl")
        rates = records(source / "rate.jsonl")
        tasks = records(source / "task_results.jsonl")
        if len(episodes) != 500 or len(tasks) != 10 or summary["total_episodes"] != 500 or \
                summary["entropy_latent_checked_views"] != 20 or summary["entropy_latent_max_abs"] != 0 or \
                len(rates) != summary["compressed_views"] or \
                sum(row["success"] for row in episodes) != summary["success_episodes"]:
            raise ValueError(f"Incomplete or inconsistent {lam}/{arm} result")
        if summary["arm"] != arm or summary["lambda"] != lam or summary["exact_train_frame_matches"] != 0:
            raise ValueError("Protocol or frame-overlap mismatch")
        for task in range(10):
            selected = [row for row in episodes if row["task_id"] == task]
            if len(selected) != 50 or [row["episode_index"] for row in selected] != list(range(50)):
                raise ValueError(f"Incomplete task {task}: {lam}/{arm}")
        for current, old in zip(episodes, reference):
            if (current["task_id"], current["episode_index"], current["initial_state_sha256"]) != \
                    (old["task_id"], old["episode_index"], old["initial_state_sha256"]):
                raise ValueError(f"Official initial-state ordering mismatch: {lam}/{arm}")
        results[(lam, arm)] = (source, summary, episodes, rates, tasks)
    # The first raw observation must also be identical across new paired arms.
    for task in range(10):
        for episode in range(50):
            first = []
            for key in ARMS:
                _, _, _, rates, _ = results[key]
                pair = {(r["view"]): r["source_rgb_sha256"] for r in rates
                        if r["task_id"] == task and r["episode_index"] == episode and r["policy_query_index"] == 0}
                if set(pair) != {"agent", "wrist"}:
                    raise ValueError("Missing first-query RGB hashes")
                first.append(pair)
            if first[0] != first[1] or first[0] != first[2]:
                raise ValueError(f"First-query RGB mismatch at {task}/{episode}")

    ROOT.mkdir(parents=True)
    for (lam, arm), (source, _summary, _episodes, _rates, _tasks) in results.items():
        target = ROOT / f"lambda_{lam}_{arm}"
        target.mkdir()
        for name in ("config.yaml", "provenance.json", "summary.json", "episode_results.jsonl",
                     "task_results.jsonl", "rate.jsonl"):
            shutil.copy2(source / name, target / name)
        audit = source / "audit.jsonl"
        (target / "large_audit_provenance.json").write_text(json.dumps({
            "path": str(audit), "sha256": sha(audit), "bytes": audit.stat().st_size,
            "note": "Full frame audit retained on HDD; not uploaded due to size"}, indent=2) + "\n")
    for source_name, target_name in (("bitrate_preflight.json", "independent_bitrate_preflight.json"),
                                     ("data_protocol_audit.v2.json", "data_protocol_audit.json"),
                                     ("preflight_teacher_parity.json", "teacher_parity.json"),
                                     ("validation_curve.jsonl", "validation_curve.jsonl"),
                                     ("sim_validation_metrics.jsonl", "sim_validation_curve.jsonl"),
                                     ("training_summary.json", "training_summary.json")):
        shutil.copy2(root / source_name, ROOT / target_name)
    training = json.loads((root / "training_summary.json").read_text())
    adapter = root / "adapter-best.pt"
    adapter_sha = sha(adapter)
    if adapter_sha != results[("0.0483", "hpcm3")][1]["checkpoints"]["adapter"]["sha256"]:
        raise ValueError("Best Adapter SHA differs from evaluation")

    with (ROOT / "plot_data.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("lambda", "arm", "bpp_agent", "bpp_wrist",
                                                    "bpp_two_view_mean", "psnr_agent_db", "psnr_wrist_db",
                                                    "success_episodes", "episodes", "success_rate", "protocol"))
        writer.writeheader()
        for (lam, arm), (_, summary, _, _, _) in results.items():
            a, w = summary["mean_actual_bpp"]["agent"], summary["mean_actual_bpp"]["wrist"]
            writer.writerow({"lambda": lam, "arm": arm, "bpp_agent": a, "bpp_wrist": w,
                             "bpp_two_view_mean": (a + w) / 2,
                             "psnr_agent_db": summary["mean_psnr_db"]["agent"],
                             "psnr_wrist_db": summary["mean_psnr_db"]["wrist"],
                             "success_episodes": summary["success_episodes"], "episodes": 500,
                             "success_rate": summary["success_rate"],
                             "protocol": "paired_raw256_bitstream_v1"})
    preflight = json.loads((root / "bitrate_preflight.json").read_text())
    low_probe = next(x for x in preflight["rates"] if x["lambda"] == "0.0018")
    low_probe_x = (low_probe["mean"]["agent"]["bpp_actual"] +
                   low_probe["mean"]["wrist"]["bpp_actual"]) / 2
    recon = [results[(lam, "reconstruction")][1] for lam in ("0.0018", "0.0483")]
    direct = results[("0.0483", "hpcm3")][1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot([(x["mean_actual_bpp"]["agent"] + x["mean_actual_bpp"]["wrist"]) / 2 for x in recon],
            [x["success_rate"] * 100 for x in recon], "o-", label="RGB reconstruction → Spatial-Pro")
    high_x = (direct["mean_actual_bpp"]["agent"] + direct["mean_actual_bpp"]["wrist"]) / 2
    ax.scatter([high_x], [direct["success_rate"] * 100], marker="s", s=90,
               label="HPCM3 high-rate (paired protocol)")
    ax.scatter([low_probe_x], [98.2], marker="s", s=90, facecolors="none", edgecolors="tab:orange",
               label="HPCM3 low-rate historical* (probe bpp)")
    ax.set(xlabel="Actual bitstream bpp, mean of agent and wrist views",
           ylabel="LIBERO-Spatial success rate (%)", ylim=(0, 105))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.11, 0.005, "*Different HPCM input protocol; x from independent static images, not its rollout.", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "bitrate_success.png", dpi=170)
    fig.savefig(ROOT / "bitrate_success.svg")
    plt.close(fig)

    table = []
    for (lam, arm), (_, summary, _, _, _) in results.items():
        bpp = summary["mean_actual_bpp"]
        psnr = summary["mean_psnr_db"]
        table.append(f"| {lam} | {arm} | {summary['success_episodes']}/500 ({summary['success_rate']:.1%}) | "
                     f"{bpp['agent']:.4f} / {bpp['wrist']:.4f} | "
                     f"{psnr['agent']:.2f} / {psnr['wrist']:.2f} | 0 |")
    task_table = []
    for task_id in range(10):
        values = [results[key][4][task_id]["success_episodes"] for key in ARMS]
        task_table.append(f"| {task_id} | {values[0]}/50 | {values[1]}/50 | {values[2]}/50 |")
    REPORT.write_text(f"""# LIBERO-Spatial Bitrate–Success Rate 实验

使用同一 Spatial-Pro 权重、官方 DEFAULT 初始状态（10 tasks × 50）、seed 7、10 dummy steps 与 8-action queue。新增三组评测使用相同 raw 256 RGB 熵编码输入，并对每个 task/episode 核对初始 state SHA 与首个双视图 RGB SHA。原始 Spatial-Pro 基线未重跑。只有本地已有 λ=0.0018 与 0.0483 两个 HPCM checkpoint，**没有中间码率点**。

![Bitrate–Success Rate](../results/hpcm3_spatial_bitrate/bitrate_success.png)

| λ | 路径 | Success | Agent / wrist 实际 bpp | Agent / wrist PSNR dB | Exact train-frame matches |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

高码率两条路径共享 Spatial-Pro、HPCM、codec 数值预处理及官方初始状态；后续闭环轨迹不同，故整段轨迹的 bpp/PSNR 是各自策略实际看到的图像分布，不是逐帧配对图像指标。表中 bpp 为真实 y/z bitstream 字节数除以 256²；PSNR 使用熵解码 RGB。每组 500 episode 的初始 state SHA、成功/失败、每 task 结果及逐查询 bpp/PSNR 在[结果目录](../results/hpcm3_spatial_bitrate/)；完整逐帧 SHA audit 留在机械盘，Git 仅保存路径和文件 SHA。新增三组每组抽检 20 张视图（每 task 首次查询双相机）的 bitstream 熵解码 latent 与 HPCM3 forward latent 完全一致，max abs=0；并非逐帧证明。训练图像与评测 raw RGB 的 exact SHA 匹配为 0。

## 各 task 成功次数

| Task | λ=0.0018 重建 | λ=0.0483 重建 | λ=0.0483 HPCM3 |
| ---: | ---: | ---: | ---: |
{chr(10).join(task_table)}

## 历史低码率 HPCM3 的使用边界

既有 Spatial HPCM3 [完整评测](../results/hpcm3_libero4/spatial/summary.json)为 **491/500（98.2%）**，Adapter SHA `{reference_summary['checkpoint_sha256']}`，不重跑。它与新增三组具有相同 Spatial-Pro、seed 和 500 个 DEFAULT 初始状态，但其 `hpcm_raw_input_bypass=false`：HPCM 接收 policy-processor 分支的图像，**不同于本轮 raw 256 输入**。历史结果也未记录每次策略查询的真实 bitstream。因此图中的空心历史点横坐标仅取独立 40 张 Spatial fresh-reset 图像上 λ=0.0018 的真实码率均值（agent/wrist {low_probe['mean']['agent']['bpp_actual']:.4f}/{low_probe['mean']['wrist']['bpp_actual']:.4f} bpp），**不是历史 rollout 的码率**，不能与实心点连接为同协议 HPCM3 曲线，也不能将高低码率 HPCM3 的差异归因于码率。低码率同协议 HPCM3 结果按要求没有重跑；严格的双路径、双码率论文曲线仍缺这一点。

## 训练、隔离和局限

唯一新增训练是 λ=0.0483 Spatial HPCM3，从随机初始化；四套 LIBERO 的全部训练轨迹图像，统一 Spatial-Pro block-2 teacher、2 ResBlocks、`SigLIP_MSE + 5×DINO_MSE`，无架构/loss 搜索。按 trajectory 划分 train 1523 / val 170 条，训练 RGB **493840** 张，val RGB **53090** 张，验证图像固定抽样 512；训练预算上限 100000 optimizer steps × batch 16，实际 **{training['global_step']}** steps，best clean val loss **{training['best_val_loss']:.6f}**。独立 Spatial simulator 40 张 fresh-reset 图像只作诊断、不参与选模。best Adapter SHA `{adapter_sha}`。HPCM λ=0.0018 SHA `{results[('0.0018', 'reconstruction')][1]['checkpoints']['hpcm']['sha256']}`；λ=0.0483 SHA `{results[('0.0483', 'reconstruction')][1]['checkpoints']['hpcm']['sha256']}`。Spatial-Pro 权重和 action head SHA 见各组 provenance。

[数据协议审计](../results/hpcm3_spatial_bitrate/data_protocol_audit.json)核对训练/验证轨迹交集、训练/验证及训练/历史正式评测 raw RGB 精确重合均为 0；独立 simulator 的 20 个 state 与 500 个正式 state 均不重合。**但 RLDS PNG 导出缺失 demo 完整 simulator initial state，无法排除训练 demo 与正式评测初始状态相同**。该限制也适用于已复用的历史低码率结果；正式初始状态未用于本轮训练、选模或预检。历史低码率 Spatial train/val 是逐帧随机划分，非本轮 trajectory-level 划分，不应把两者当作只变码率的严格消融。

[CSV](../results/hpcm3_spatial_bitrate/plot_data.csv)只纳入三组同协议新增评测；历史点因横坐标来自独立静态 probe，未混入 CSV 主表。完整训练 checkpoint、日志和审计 JSONL 在机械盘 `{root}`，不上传 Git。绘图数据为真实测量，无虚构中间码率。
""")
    print(f"Published {REPORT} and {ROOT}")


if __name__ == "__main__":
    main()
