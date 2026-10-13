"""Export small, reproducible result files and a report after unattended run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

REPO = Path(__file__).resolve().parents[3]
DEST = REPO / "results/hpcm3_object_highrate_libero4_full"
REPORT = REPO / "reports/hpcm3_object_highrate_libero4_full.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def value(item: dict, name: str) -> str:
    entry = item.get(name)
    return f"{entry:.6f}" if isinstance(entry, (int, float)) else "—"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = args.output_root
    audit = json.loads((root / "data_audit/summary.json").read_text())
    parity = json.loads((root / "preflight_parity.json").read_text())
    training = json.loads((root / "training_summary.json").read_text())
    eval_summary = json.loads((args.eval_dir / "summary.json").read_text())
    curve = rows(root / "validation_curve.jsonl")
    sim = rows(root / "sim_validation_metrics.jsonl")
    if not curve or not sim:
        raise ValueError("Missing clean/simulator validation records")
    if audit["train_protected_exact_rgb_hash_overlap"] or audit["train_val_exact_rgb_hash_overlap"]:
        raise ValueError("Leakage audit failed")
    if eval_summary["exact_train_frame_matches"]:
        raise ValueError("Rollout contained an exact train frame")
    if not all(row["hpcm_train_vs_separate_max_abs"] < 1e-5 and
               row["object_teacher_block2_max_abs"] < 1e-4 for row in parity["samples"]):
        raise ValueError("Preprocessing/teacher parity failed")

    DEST.mkdir(parents=True, exist_ok=True)
    sources = {
        "train_config.yaml": args.config,
        "eval_config.yaml": args.eval_dir / "config.yaml",
        "data_audit.json": root / "data_audit/summary.json",
        "preflight_parity.json": root / "preflight_parity.json",
        "validation_curve.jsonl": root / "validation_curve.jsonl",
        "sim_validation_curve.jsonl": root / "sim_validation_metrics.jsonl",
        "training_summary.json": root / "training_summary.json",
        "eval_summary.json": args.eval_dir / "summary.json",
        "episode_results.jsonl": args.eval_dir / "episode_results.jsonl",
        "audit.jsonl": args.eval_dir / "audit.jsonl",
        "rate.jsonl": args.eval_dir / "rate.jsonl",
    }
    for name, path in sources.items():
        shutil.copy2(path, DEST / name)
    adapter = Path(training["adapter_checkpoint"])
    best = Path(training["best_checkpoint"])
    hpcm = Path(cfg["model"]["hpcm"]["checkpoint"])
    object_pro = Path(cfg["data"]["teacher_rollout_checkpoint"]) / "model.safetensors"
    checkpoint_shas = {"hpcm": sha256(hpcm), "adapter_best": sha256(adapter),
                       "best_full_state": sha256(best), "object_pro": sha256(object_pro)}
    (DEST / "checkpoint_sha256.json").write_text(json.dumps(checkpoint_shas, indent=2) + "\n")

    # A resumed run may repeat its last validation step or be interrupted
    # between the two callbacks. Keep the latest complete pair at each step.
    clean_by_step = {row["global_step"]: row for row in curve}
    sim_by_step = {row["global_step"]: row for row in sim}
    paired_steps = sorted(clean_by_step.keys() & sim_by_step.keys())
    if not paired_steps:
        raise ValueError("No paired clean/simulator validation steps")
    curve_lines = []
    for step in paired_steps:
        clean, simulator = clean_by_step[step], sim_by_step[step]
        curve_lines.append(
            f"| {clean['global_step']} | {value(clean, 'train/total_loss')} | "
            f"{value(clean, 'val_loss')} | "
            f"{value(simulator['views']['agent'], 'loss')} | "
            f"{value(simulator['views']['wrist'], 'loss')} | "
            f"{value(simulator['views']['all'], 'loss')} |"
        )
    suite_lines = []
    for suite, counts in audit["suite_counts"].items():
        suite_lines.append(f"| {suite} | {counts['train_trajectories']} | {counts['train_images']} | "
                           f"{counts['val_trajectories']} | {counts['val_images']} |")
    result_lines = []
    for episode in rows(args.eval_dir / "episode_results.jsonl"):
        result_lines.append(f"| {episode['episode_index']} | {int(episode['A_success'])} | "
                            f"{int(episode['B_success'])} | {int(episode['C_success'])} | "
                            f"{episode['exact_train_frame_matches']} | "
                            f"{episode['actual_bpp']['agent']:.4f} / {episode['actual_bpp']['wrist']:.4f} |")
    old_c = 1
    new_c = eval_summary["successes"]["C"]
    outcome = ("高于旧高码率 Object Adapter 的 1/5，但五个状态样本不足以证明 suite 级改善。"
               if new_c > old_c else "未超过旧高码率 Object Adapter 的 1/5；本组全量训练未显示五状态闭环改善。")
    text = f"""# Object 高码率 HPCM3：四套 LIBERO 全量轨迹图像联合训练

本实验只改变高码率 Object HPCM3 的训练语料规模：固定 λ=0.0483 HPCM、Object-Pro teacher、2 ResBlocks、block-2 边界蒸馏及 `SigLIP_MSE + 5×DINO_MSE`；没有 Prefix Head、full-visual loss 或旧 Adapter 初始化。Object task 0 五个配对状态的结果为原始 Object-Pro **{eval_summary['successes']['A']}/5**、高码率重建 RGB→Object-Pro **{eval_summary['successes']['B']}/5**、本次 Adapter **{new_c}/5**（旧高码率 Object Adapter 为 1/5）。{outcome}不把这五个曾用于历史实验的状态视为全新、无偏的论文正式测试集。

## 历史协议与本轮固定变量

历史 Spatial W&B 本地 run `loexl72x` 的 summary 记录 `global_step=48,829`、epoch 3、`val_loss=0.016296`；仓库历史配置为四套数据、batch 32、2 ResBlocks、同一边界损失。但历史 train/val 是逐帧随机划分，同 trajectory 可跨两侧，且剥离后的 Spatial `last-v1.pt` 不含可复核的训练 manifest，因此不能声称其四套全部图像已由该 checkpoint 自证。本轮用真实存在的四套 RLDS 导出 PNG，按 trajectory 分离 train/val，**训练侧不设 `train_max_samples`**。相对[旧高码率 Object 训练](hpcm3_object_highrate_boundary.md)，保留 batch 16、AdamW `1e-4`、cosine、FP16、Object-Pro 原生 teacher preprocessing 与 raw 256 HPCM 输入。为匹配历史 Spatial 约 156 万次图像曝光，预算上限 100,000 optimizer steps × batch 16；每 8,192 step 做一次验证，`min_delta=1e-4, patience=5` 早停。验证从 held-out 轨迹固定抽取 512 张，simulator 的 40 张只用于诊断，**不参与选模**。

| suite | train 轨迹 | train 图像 | val 轨迹 | val 图像 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(suite_lines)}
| **总计** | **{audit['split_manifest']['train']}** | **{audit['train_images']}** | **{audit['split_manifest']['val']}** | **{audit['val_images']}** |

split manifest SHA256 `{audit['split_manifest']['sha256']}`；train/val trajectory 交集 0，解码 RGB exact-frame 交集 **{audit['train_val_exact_rgb_hash_overlap']}**。历史 Object rollout audit 中 {audit['protected_rollout_rgb_hashes']} 个 RGB SHA 与训练图像精确重合 **{audit['train_protected_exact_rgb_hash_overlap']}**；独立 simulator 20 个 state 与正式默认 state 碰撞 **0**。但 RLDS PNG 导出缺少完整 demo simulator initial state，**无法排除训练 demo 与正式初始状态等值**；这仍是论文正式使用前必须补齐的审计。正式 rollout 图像只用于 SHA 审计和最终闭环测试，未进入训练、验证或选模。四套数据中各抽一张 held-out 图像做 HPCM 与 Object-Pro block-2 数值预检，逐图值见[预检结果](../results/hpcm3_object_highrate_libero4_full/preflight_parity.json)。

## 训练与独立 simulator 曲线

模型由随机初始化开始，clean trajectory-held-out `val_loss` 决定最佳权重；simulator fresh-reset 图像来自前次固定 manifest，非正式评测 initial states。

| Optimizer step | Train | Clean val | Sim agent | Sim wrist | Sim all |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(curve_lines)}

实际停止 epoch **{training['stopped_epoch']}**、optimizer steps **{training['global_step']}**，最佳 clean validation loss **{training['best_val_loss']:.8f}**。曲线完整记录：[clean](../results/hpcm3_object_highrate_libero4_full/validation_curve.jsonl)、[simulator 分视角及 DINO/SigLIP](../results/hpcm3_object_highrate_libero4_full/sim_validation_curve.jsonl)。训练日志与完整 checkpoint 留在机械盘 `{root}`；只有小体积配置/指标进 Git。

## Object task 0：五个配对 episode

沿用先前 task 0 默认 initial states 0–4、seed 7、10 dummy steps、8-action queue、双相机和 `libero_object_no_noops`。A/B 复用历史同状态结果，C 本轮独立执行；初始 state SHA 逐条核对。C 按每次策略查询的真实 HPCM bitstream 计码率，重建图不输入 C。后续轨迹因策略不同不逐帧配对。

| state index | A 原始 RGB | B 重建 RGB | C 本次 HPCM3 | C exact train-frame matches | C agent/wrist bpp |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(result_lines)}
| **合计** | **{eval_summary['successes']['A']}/5** | **{eval_summary['successes']['B']}/5** | **{new_c}/5** | **{eval_summary['exact_train_frame_matches']}** | **{eval_summary['mean_actual_bpp']['agent']:.4f} / {eval_summary['mean_actual_bpp']['wrist']:.4f}** |

[逐 episode](../results/hpcm3_object_highrate_libero4_full/episode_results.jsonl)、[frame audit](../results/hpcm3_object_highrate_libero4_full/audit.jsonl)、[真实码率](../results/hpcm3_object_highrate_libero4_full/rate.jsonl)与[完整 summary](../results/hpcm3_object_highrate_libero4_full/eval_summary.json)可复核。若 C 表现较好，仍需预先冻结方案后在未用于历次诊断的新初始状态和全 task 范围重复，不能据这五个样本直接宣称论文级收益。

## 权重及可复现性

| 权重 | SHA256 |
| --- | --- |
| HPCM λ=0.0483 | `{checkpoint_shas['hpcm']}` |
| Object-Pro model.safetensors | `{checkpoint_shas['object_pro']}` |
| 本轮最佳 Adapter | `{checkpoint_shas['adapter_best']}` |
| 本轮最佳完整训练状态 | `{checkpoint_shas['best_full_state']}` |

[训练配置](../results/hpcm3_object_highrate_libero4_full/train_config.yaml)、[评测配置](../results/hpcm3_object_highrate_libero4_full/eval_config.yaml)、[数据审计](../results/hpcm3_object_highrate_libero4_full/data_audit.json)和上述结果随报告提交；大型权重、原图、W&B 离线二进制与训练日志不上传。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    print(f"Wrote {REPORT} and small result exports to {DEST}", flush=True)


if __name__ == "__main__":
    main()
