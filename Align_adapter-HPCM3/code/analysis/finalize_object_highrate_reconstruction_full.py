"""Publish the frozen Object λ=0.0483 reconstruction 10x50 control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
DEST = REPO / "results/hpcm3_object_highrate_reconstruction_full_eval"
REPORT = REPO / "reports/hpcm3_object_highrate_reconstruction_full_500.md"


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
    if DEST.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite published Object reconstruction result")
    summary = json.loads((source / "summary.json").read_text())
    episodes = records(source / "episode_results.jsonl")
    rates = records(source / "rate.jsonl")
    tasks = records(source / "task_results.jsonl")
    baseline_path = REPO / "results/hpcm3_object_highrate_libero4_full_eval/episode_results.jsonl"
    baseline = records(baseline_path)
    if summary["total_episodes"] != 500 or len(episodes) != len(baseline) != 500 or len(tasks) != 10:
        raise ValueError("Incomplete 10x50 Object result")
    if summary["first_query_rgb_matches_hpcm3"] != 1000 or \
            summary["entropy_latent_checked_views"] != 20 or summary["entropy_latent_max_abs"] != 0:
        raise ValueError("First-query or entropy latent parity not complete")
    if len(rates) != summary["compressed_views"] or \
            sum(row["success"] for row in episodes) != summary["success_episodes"] or \
            sum(row["exact_train_frame_matches"] for row in episodes) != summary["exact_train_frame_matches"]:
        raise ValueError("Summary/JSONL mismatch")
    for task_id in range(10):
        subset = [r for r in episodes if r["task_id"] == task_id]
        if len(subset) != 50 or [r["episode_index"] for r in subset] != list(range(50)):
            raise ValueError(f"Task {task_id} not fully evaluated")
    for row, old in zip(episodes, baseline):
        if (row["task_id"], row["episode_index"], row["initial_state_sha256"]) != \
                (old["task_id"], old["episode_index"], old["initial_state_sha256"]):
            raise ValueError("HPCM3/Reconstruction official initial states differ")
    if summary["exact_train_frame_matches"] != 0:
        raise ValueError("Exact train-frame overlap detected")

    DEST.mkdir(parents=True)
    for name in ("config.yaml", "provenance.json", "summary.json", "episode_results.jsonl",
                 "task_results.jsonl", "rate.jsonl"):
        shutil.copy2(source / name, DEST / name)
    audit = source / "audit.jsonl"
    (DEST / "large_audit_provenance.json").write_text(json.dumps({
        "path": str(audit), "sha256": sha(audit), "bytes": audit.stat().st_size,
        "note": "Full frame audit retained on HDD; omitted from Git due to size"}, indent=2) + "\n")
    b_only = sum(row["success"] and not old["success"] for row, old in zip(episodes, baseline))
    c_only = sum(old["success"] and not row["success"] for row, old in zip(episodes, baseline))
    table = []
    for task_id, task in enumerate(tasks):
        old = sum(r["success"] for r in baseline if r["task_id"] == task_id)
        table.append(f"| {task_id} | {task['success_episodes']}/50 | {old}/50 |")
    lines = "\n".join(table)
    REPORT.write_text(f"""# Object λ=0.0483 RGB 重建：完整 500-episode 配对控制

固定 HPCM `0.0483.pth.tar` 和完整 Object-Pro，使用同一 raw 256 RGB 数值路径进行真实 y/z 熵编码、解码与 RGB 重建，再由 Object-Pro 原生 JPEG/Lanczos3、center crop、processor 处理。LIBERO-Object 官方 DEFAULT 初始状态、10 tasks × 50 episodes、seed 7、10 dummy steps、8-action queue 与[已有高码率 HPCM3 正式结果](hpcm3_object_highrate_libero4_full_500.md)一致；模型权重未修改。逐 episode 的 state SHA 500/500 一致，首查询双视图的 RGB SHA **1000/1000 一致**。首查询 y/z bitstream 长度逐视图完全一致 **{summary['first_query_bitstream_byte_matches_hpcm3']}/1000**；y_bytes 差值范围 {summary['first_query_y_byte_delta_range']}、z_bytes 差值范围 {summary['first_query_z_byte_delta_range']}。这表明两种策略加载上下文中真实 codec 编码并非保证逐字节相同；不能把相同 RGB/权重误写为完全同一 bitstream。后续轨迹因策略分支不同不逐帧配对。

| 路径 | Success | Agent / wrist 真实 bpp | Agent / wrist PSNR |
| --- | ---: | ---: | ---: |
| 本轮 RGB Reconstruction→完整 Object-Pro | **{summary['success_episodes']}/500 ({summary['success_rate']:.1%})** | {summary['mean_actual_bpp']['agent']:.4f} / {summary['mean_actual_bpp']['wrist']:.4f} | {summary['mean_psnr_db']['agent']:.2f} / {summary['mean_psnr_db']['wrist']:.2f} dB |
| 已有 HPCM3→Object-Pro 后半视觉塔 | **{sum(r['success'] for r in baseline)}/500** | 0.5553 / 0.5020 | 未在历史 500-episode 运行中记录；不得填入本轮 PSNR |

同一初始状态上的 Reconstruction 独赢 **{b_only}** 个、HPCM3 独赢 **{c_only}** 个。成功率不能单靠码率或 PSNR 解释；若差距很小，不能声称一条路径优于另一条。bpp 为逐查询 y/z bitstream 字节数计算的全帧均值，不是 entropy model 估计。PSNR 是本轮重建闭环轨迹上的逐帧均值，不是两条策略的同帧 PSNR。

| Task | Reconstruction | HPCM3 |
| ---: | ---: | ---: |
{lines}

完整[配置](../results/hpcm3_object_highrate_reconstruction_full_eval/config.yaml)、[逐 episode](../results/hpcm3_object_highrate_reconstruction_full_eval/episode_results.jsonl)、[逐查询码率与 PSNR](../results/hpcm3_object_highrate_reconstruction_full_eval/rate.jsonl)、[汇总](../results/hpcm3_object_highrate_reconstruction_full_eval/summary.json)和[provenance](../results/hpcm3_object_highrate_reconstruction_full_eval/provenance.json)已保存。HPCM SHA `{summary['checkpoints']['hpcm']['sha256']}`，Object-Pro model SHA `{summary['checkpoints']['object_pro']['sha256']}`，HPCM3 Adapter SHA `{summary['checkpoints']['adapter']['sha256']}`。本轮抽检 20 个真实熵解码 latent 与 HPCM3 forward latent 逐元素相等（max abs=0），不将抽样核对表述为全帧证明。训练图像与 rollout raw RGB exact SHA 重合 **{summary['exact_train_frame_matches']}**；但 RLDS PNG 没有 demo 完整 simulator initial state，训练 demo 与正式初始状态是否等值仍无法确认。完整逐帧 SHA audit 留在机械盘 `{audit}`，SHA `{sha(audit)}`，不上传 Git。
""")
    print(f"Published {REPORT} and {DEST}")


if __name__ == "__main__":
    main()
