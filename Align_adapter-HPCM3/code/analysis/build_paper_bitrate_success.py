"""Build the protocol-matched paper table/figure after low-rate Spatial finishes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
DEST = REPO / "results/hpcm3_paper_bitrate_success"
REPORT = REPO / "reports/hpcm3_paper_bitrate_success.md"
SOURCES = (
    ("Spatial", "0.0018", "Reconstruction", REPO / "results/hpcm3_spatial_bitrate/lambda_0.0018_reconstruction"),
    ("Spatial", "0.0018", "HPCM3", REPO / "results/hpcm3_spatial_lowrate_retrain"),
    ("Spatial", "0.0483", "Reconstruction", REPO / "results/hpcm3_spatial_bitrate/lambda_0.0483_reconstruction"),
    ("Spatial", "0.0483", "HPCM3", REPO / "results/hpcm3_spatial_bitrate/lambda_0.0483_hpcm3"),
    ("Object", "0.0483", "Reconstruction", REPO / "results/hpcm3_object_highrate_reconstruction_full_eval"),
    ("Object", "0.0483", "HPCM3", REPO / "results/hpcm3_object_highrate_libero4_full_eval"),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    if DEST.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite paper result")
    results = []
    for suite, lam, arm, path in SOURCES:
        summary_path = path / "summary.json"
        summary = json.loads(summary_path.read_text())
        episodes = read_jsonl(path / "episode_results.jsonl")
        assert summary["suite"] == f"libero_{suite.lower()}"
        assert summary["total_episodes"] == 500 and summary["task_count"] == 10
        assert summary["seed"] == 7 and summary["exact_train_frame_matches"] == 0
        assert len(episodes) == 500 and len(summary["per_task"]) == 10
        assert summary["success_episodes"] == sum(e["success"] for e in episodes)
        assert summary["checkpoints"]["hpcm"]["sha256"] == sha(REPO / "HPCM/ckpt" / f"{lam}.pth.tar")
        assert summary["mean_actual_bpp"]["agent"] > 0 and summary["mean_actual_bpp"]["wrist"] > 0
        by_task = [sum(e["success"] for e in episodes if e["task_id"] == task) for task in range(10)]
        assert [sum(e["task_id"] == task for e in episodes) for task in range(10)] == [50] * 10
        assert [e["episode_index"] for task in range(10) for e in episodes if e["task_id"] == task] == list(range(50)) * 10
        results.append({"suite": suite, "lambda": lam, "arm": arm, "path": path,
                        "summary": summary, "episodes": episodes, "per_task": by_task,
                        "summary_sha256": sha(summary_path),
                        "episode_results_sha256": sha(path / "episode_results.jsonl")})

    for suite, lam in (("Spatial", "0.0018"), ("Spatial", "0.0483"), ("Object", "0.0483")):
        a, b = [r for r in results if r["suite"] == suite and r["lambda"] == lam]
        assert a["arm"] == "Reconstruction" and b["arm"] == "HPCM3"
        for x, y in zip(a["episodes"], b["episodes"]):
            assert (x["task_id"], x["episode_index"], x["initial_state_sha256"]) == \
                   (y["task_id"], y["episode_index"], y["initial_state_sha256"])
            assert x["initial_state_source"] == y["initial_state_source"] == "DEFAULT"
        assert a["summary"]["checkpoints"]["hpcm"]["sha256"] == b["summary"]["checkpoints"]["hpcm"]["sha256"]
        vla_key = "spatial_pro" if suite == "Spatial" else "object_pro"
        assert a["summary"]["checkpoints"][vla_key] == b["summary"]["checkpoints"][vla_key]
        if suite == "Object":
            for key in ("action_head", "proprio_projector"):
                assert a["summary"]["checkpoints"][key] == b["summary"]["checkpoints"][key]
    spatial_vla = [r["summary"]["checkpoints"]["spatial_pro"] for r in results if r["suite"] == "Spatial"]
    assert spatial_vla.count(spatial_vla[0]) == len(spatial_vla)

    DEST.mkdir(parents=True)
    columns = ["suite", "lambda", "arm", "success_episodes", "total_episodes", "success_rate",
               "agent_bpp", "wrist_bpp", "two_view_mean_bpp", "agent_psnr_db", "wrist_psnr_db",
               "hpcm_sha256", "adapter_sha256", "vla_model_sha256", "action_head_sha256",
               "proprio_projector_sha256", "summary_sha256", "episode_results_sha256", "source"]
    rows = []
    for r in results:
        s = r["summary"]
        bpp = s["mean_actual_bpp"]
        psnr = s.get("mean_psnr_db") or {}
        checkpoints = s["checkpoints"]
        if r["suite"] == "Spatial":
            vla_sha = checkpoints["spatial_pro"]["model.safetensors"]
            head_sha = checkpoints["spatial_pro"]["action_head--checkpoint.pt"]
            proprio_sha = checkpoints["spatial_pro"]["proprio_projector--checkpoint.pt"]
        else:
            vla_sha = checkpoints["object_pro"]["sha256"]
            head_sha = checkpoints["action_head"]["sha256"]
            proprio_sha = checkpoints["proprio_projector"]["sha256"]
        rows.append({"suite": r["suite"], "lambda": r["lambda"], "arm": r["arm"],
                     "success_episodes": s["success_episodes"], "total_episodes": 500,
                     "success_rate": s["success_rate"], "agent_bpp": bpp["agent"],
                     "wrist_bpp": bpp["wrist"],
                     "two_view_mean_bpp": (bpp["agent"] + bpp["wrist"]) / 2,
                     "agent_psnr_db": psnr.get("agent", ""), "wrist_psnr_db": psnr.get("wrist", ""),
                     "hpcm_sha256": s["checkpoints"]["hpcm"]["sha256"],
                     "adapter_sha256": (s["checkpoints"].get("adapter") or {}).get("sha256", ""),
                     "vla_model_sha256": vla_sha, "action_head_sha256": head_sha,
                     "proprio_projector_sha256": proprio_sha,
                     "summary_sha256": r["summary_sha256"],
                     "episode_results_sha256": r["episode_results_sha256"],
                     "source": str(r["path"].relative_to(REPO))})
    with (DEST / "paper_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (DEST / "plot_data.json").write_text(json.dumps(rows, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {"Reconstruction": "#2878B5", "HPCM3": "#E07A2F"}
    for arm in ("Reconstruction", "HPCM3"):
        spatial = sorted((r for r in rows if r["suite"] == "Spatial" and r["arm"] == arm),
                         key=lambda r: r["two_view_mean_bpp"])
        ax.plot([r["two_view_mean_bpp"] for r in spatial],
                [100 * r["success_rate"] for r in spatial], "o-", color=colors[arm],
                linewidth=2, markersize=7, label=f"Spatial {arm}")
        object_point = next(r for r in rows if r["suite"] == "Object" and r["arm"] == arm)
        ax.scatter(object_point["two_view_mean_bpp"], 100 * object_point["success_rate"],
                   marker="s", s=70, color=colors[arm], label=f"Object {arm} (one rate)")
    ax.set(xlabel="Measured bitstream bpp (mean of agent and wrist)",
           ylabel="LIBERO success rate (%)",
           ylim=(max(0, min(100 * r["success_rate"] for r in rows) - 4), 100),
           title="Bitrate–Success: 10 tasks × 50 official initial states")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(DEST / "bitrate_success.png", dpi=200)
    plt.close(fig)

    def rate(row: dict) -> str:
        return f"{row['success_episodes']}/500 ({100 * row['success_rate']:.1f}%)"

    table = "\n".join(
        f"| {r['suite']} | {r['lambda']} | {r['arm']} | {rate(r)} | "
        f"{r['agent_bpp']:.4f} / {r['wrist_bpp']:.4f} | "
        f"{r['agent_psnr_db']:.2f} / {r['wrist_psnr_db']:.2f} |" if r["agent_psnr_db"] != "" else
        f"| {r['suite']} | {r['lambda']} | {r['arm']} | {rate(r)} | "
        f"{r['agent_bpp']:.4f} / {r['wrist_bpp']:.4f} | 未记录 |" for r in rows)
    task_table = "\n".join(
        "| " + " | ".join([f"{task}"] + [f"{r['per_task'][task]}/50" for r in results]) + " |"
        for task in range(10))
    REPORT.write_text(f"""# HPCM3 论文 Bitrate–Success 对照

![Bitrate–Success](../results/hpcm3_paper_bitrate_success/bitrate_success.png)

全部正式结果为官方 DEFAULT 初始状态、每 task 50 episodes、seed 7；同 suite、同 λ 两路径的 500 个 initial-state SHA 逐项相同，HPCM 和 VLA 权重 SHA 一致。Spatial 四点的 Spatial-Pro 权重 SHA 一致。全部 exact train-frame match 为 0；RLDS PNG 缺少 demo 完整 simulator state，仍**无法排除** demo 与正式初始状态等值。具体配置与逐 episode 证据见 CSV 的 `source` 路径和各实验报告。

| Suite | λ | 路径 | 成功率 | agent / wrist 实际 bpp | agent / wrist PSNR dB |
| --- | ---: | --- | ---: | ---: | ---: |
{table}

图的横轴是两视图各自真实 y/z bitstream bpp 的算术平均，不是 HPCM λ，也不是估计码率。HPCM3 的 PSNR 若有记录，仅是同一策略实际观察帧的**诊断性重建**，重建 RGB 并未输入该策略。各策略闭环轨迹不同，轨迹均值不能视为逐帧配对画质。Object HPCM3 历史 500-episode 运行未记录 PSNR，因此留空。

Spatial 在 λ=0.0018 和 0.0483 均有同协议 Reconstruction/HPCM3 正式对照，可连成各自的两点曲线。Object **只找到 λ=0.0483 checkpoint**，仅有一组双路径单点；没有第二码率，不连接成曲线，也不填充虚构点。更高码率检索见[checkpoint 核查](hpcm_object_higher_rate_checkpoint_inventory.md)。历史低码率 Spatial HPCM3 因输入协议不同未混入主表。不能仅凭这些点外推码率趋势或断言 HPCM3 优于重建。

## 每个任务成功次数（分母均为 50）

| Task | Spatial 0.0018 Rec | Spatial 0.0018 HPCM3 | Spatial 0.0483 Rec | Spatial 0.0483 HPCM3 | Object 0.0483 Rec | Object 0.0483 HPCM3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{task_table}

[CSV](../results/hpcm3_paper_bitrate_success/paper_table.csv)和[绘图数据](../results/hpcm3_paper_bitrate_success/plot_data.json)保留完整 checkpoint/结果 SHA。500-episode 原始 JSONL 和审计限定分别见[Spatial 高码率与重建](hpcm3_spatial_bitrate.md)、[Spatial 低码率重训](hpcm3_spatial_lowrate_retrain_500.md)、[Object HPCM3](hpcm3_object_highrate_libero4_full_500.md)、[Object RGB 重建](hpcm3_object_highrate_reconstruction_full_500.md)。大体积权重和逐帧 audit 不上传 Git。
""")
    print(f"Wrote {REPORT} and {DEST}")


if __name__ == "__main__":
    main()
