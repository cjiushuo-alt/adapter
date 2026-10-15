"""Publish the new raw256/trajectory-split low-rate Spatial HPCM3 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
DEST = REPO / "results/hpcm3_spatial_lowrate_retrain"
REPORT = REPO / "reports/hpcm3_spatial_lowrate_retrain_500.md"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_queries(path: Path) -> dict[tuple[int, int, str], dict]:
    first = {}
    for row in records(path):
        if row["policy_query_index"] == 0:
            key = (row["task_id"], row["episode_index"], row["view"])
            if key in first:
                raise ValueError(f"Duplicate first-query view: {key}")
            first[key] = row
    if len(first) != 1000:
        raise ValueError("Expected 1000 first-query views")
    return first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    if DEST.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite low-rate Spatial publication")
    source = args.eval_dir
    summary = json.loads((source / "summary.json").read_text())
    train = json.loads((args.train_root / "training_summary.json").read_text())
    episodes = records(source / "episode_results.jsonl")
    rates = records(source / "rate.jsonl")
    tasks = records(source / "task_results.jsonl")
    recon_root = REPO / "results/hpcm3_spatial_bitrate/lambda_0.0018_reconstruction"
    recon = records(recon_root / "episode_results.jsonl")
    recon_summary = json.loads((recon_root / "summary.json").read_text())
    if len(episodes) != 500 or len(recon) != 500 or len(tasks) != 10 or summary["total_episodes"] != 500:
        raise ValueError("Incomplete Spatial 10x50 result")
    if summary["lambda"] != "0.0018" or summary["arm"] != "hpcm3" or \
            summary["entropy_latent_checked_views"] != 20 or summary["entropy_latent_max_abs"] != 0 or \
            summary["exact_train_frame_matches"] != 0 or len(rates) != summary["compressed_views"]:
        raise ValueError("Low-rate result protocol/audit mismatch")
    if sum(r["success"] for r in episodes) != summary["success_episodes"]:
        raise ValueError("Success summary mismatch")
    if summary["checkpoints"]["adapter"]["sha256"] != sha(args.train_root / "adapter-best.pt"):
        raise ValueError("Best Adapter SHA differs from evaluation")
    if summary["checkpoints"]["hpcm"]["sha256"] != recon_summary["checkpoints"]["hpcm"]["sha256"]:
        raise ValueError("HPCM SHA differs between low-rate arms")
    for task_id in range(10):
        subset = [r for r in episodes if r["task_id"] == task_id]
        if len(subset) != 50 or [r["episode_index"] for r in subset] != list(range(50)):
            raise ValueError(f"Task {task_id} incomplete")
    for current, old in zip(episodes, recon):
        if (current["task_id"], current["episode_index"], current["initial_state_sha256"]) != \
                (old["task_id"], old["episode_index"], old["initial_state_sha256"]):
            raise ValueError("Low-rate arm initial-state mismatch")
    left = first_queries(source / "rate.jsonl")
    right = first_queries(recon_root / "rate.jsonl")
    if left.keys() != right.keys():
        raise ValueError("First-query view keys differ")
    if any(left[key]["source_rgb_sha256"] != right[key]["source_rgb_sha256"] for key in left):
        raise ValueError("First-query RGB differs between low-rate arms")
    byte_matches = sum(left[key]["y_bytes"] == right[key]["y_bytes"] and
                       left[key]["z_bytes"] == right[key]["z_bytes"] for key in left)
    c_only = sum(a["success"] and not b["success"] for a, b in zip(episodes, recon))
    b_only = sum(b["success"] and not a["success"] for a, b in zip(episodes, recon))
    artifact = args.train_root / "preflight_teacher_parity.json"
    parity = json.loads(artifact.read_text())
    if len(parity["samples"]) != 4 or any(s["object_teacher_block2_max_abs"] != 0 or
                                        s["hpcm_train_vs_separate_max_abs"] != 0 for s in parity["samples"]):
        raise ValueError("Training preflight parity failed")

    DEST.mkdir(parents=True)
    for name in ("config.yaml", "provenance.json", "summary.json", "episode_results.jsonl",
                 "task_results.jsonl", "rate.jsonl"):
        shutil.copy2(source / name, DEST / name)
    for source_path, name in ((args.train_root / "training_summary.json", "training_summary.json"),
                              (args.train_root / "validation_curve.jsonl", "validation_curve.jsonl"),
                              (args.train_root / "sim_validation_metrics.jsonl", "sim_validation_curve.jsonl"),
                              (artifact, "teacher_parity.json"),
                              (REPO / "Align_adapter-HPCM3/code/configs/train_spatial_lowrate_libero4_full.yaml", "train_config.yaml")):
        shutil.copy2(source_path, DEST / name)
    audit_path = source / "audit.jsonl"
    (DEST / "large_audit_provenance.json").write_text(json.dumps({
        "path": str(audit_path), "sha256": sha(audit_path), "bytes": audit_path.stat().st_size,
        "note": "Complete frame audit retained on HDD; not uploaded due to size"}, indent=2) + "\n")
    task_table = "\n".join(
        f"| {task_id} | {recon_summary['per_task'][task_id]['success_episodes']}/50 | "
        f"{tasks[task_id]['success_episodes']}/50 |" for task_id in range(10))
    REPORT.write_text(f"""# Spatial λ=0.0018：raw256/trajectory-split HPCM3 重训及完整评测

本轮只替换高码率 Spatial 实验的 HPCM checkpoint 为冻结的 `0.0018.pth.tar`，从随机初始化训练独立 Adapter；四套全量 LIBERO 训练轨迹、trajectory-level split、raw256 HPCM 输入、Spatial-Pro teacher、2 ResBlocks、block-2 `SigLIP_MSE + 5×DINO_MSE`、batch 16、seed 42、验证频率/early stopping 等与高码率配置一致。历史低码率 `last-v1.pt` 权重**未复用**，也未作为严格对照。正式 rollout 使用同一 Spatial-Pro 和已有[低码率重建 RGB 完整结果](hpcm3_spatial_bitrate.md)的 DEFAULT 500 初始状态、seed 7、10 dummy steps、8-action queue。

| λ=0.0018 路径 | Success | Agent / wrist 真实 bpp | Agent / wrist PSNR |
| --- | ---: | ---: | ---: |
| 已有 Reconstruction→完整 Spatial-Pro | **{recon_summary['success_episodes']}/500 ({recon_summary['success_rate']:.1%})** | {recon_summary['mean_actual_bpp']['agent']:.4f} / {recon_summary['mean_actual_bpp']['wrist']:.4f} | {recon_summary['mean_psnr_db']['agent']:.2f} / {recon_summary['mean_psnr_db']['wrist']:.2f} dB |
| 新 HPCM3→Spatial-Pro 后半视觉塔 | **{summary['success_episodes']}/500 ({summary['success_rate']:.1%})** | {summary['mean_actual_bpp']['agent']:.4f} / {summary['mean_actual_bpp']['wrist']:.4f} | {summary['mean_psnr_db']['agent']:.2f} / {summary['mean_psnr_db']['wrist']:.2f} dB¹ |

¹ HPCM3 行的 PSNR 是对其实际观察帧做**诊断性重建**，重建 RGB 未输入 HPCM3 策略。两条闭环轨迹不同，整段均值不是逐帧配对画质。500 个 initial-state SHA 与首查询 1000 个双视图原始 RGB SHA 均一致；首查询 y/z bitstream 长度逐视图一致 {byte_matches}/1000。配对状态中 HPCM3 独赢 **{c_only}** 个，重建独赢 **{b_only}** 个；不能将很小的差异解读为稳健优势。

| Task | Reconstruction | HPCM3 |
| ---: | ---: | ---: |
{task_table}

训练实际 **{train['global_step']}** optimizer steps，最佳 clean trajectory-held-out val loss **{train['best_val_loss']:.6f}**；[训练曲线](../results/hpcm3_spatial_lowrate_retrain/validation_curve.jsonl)和[独立 simulator 曲线](../results/hpcm3_spatial_lowrate_retrain/sim_validation_curve.jsonl)分开保存，后者未用于选模。训练前四套 held-out 图像上的 teacher block-2 与 HPCM 输入数值核对误差为 0。新 Adapter SHA `{summary['checkpoints']['adapter']['sha256']}`，HPCM SHA `{summary['checkpoints']['hpcm']['sha256']}`；其他权重 SHA 与运行 commit 见[provenance](../results/hpcm3_spatial_lowrate_retrain/provenance.json)。逐 episode 结果、实际 y/z bitstream bpp/PSNR、完整[汇总 JSON](../results/hpcm3_spatial_lowrate_retrain/summary.json)已保存；抽查 20 张视图的真实熵解码 latent 与前向 latent max abs=0，并非所有帧的证明。训练 raw RGB 与本轮 rollout 精确 SHA 重合 0；但 RLDS PNG 无 demo 完整 simulator initial state，**不能排除 demo 与正式初始状态等值**。正式评测状态未用于本轮训练或选模。逐帧 audit 保留在机械盘 `{audit_path}`，SHA `{sha(audit_path)}`，不进 Git。
""")
    print(f"Published {REPORT} and {DEST}")


if __name__ == "__main__":
    main()
