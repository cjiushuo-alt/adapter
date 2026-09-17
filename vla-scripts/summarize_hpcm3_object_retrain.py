#!/usr/bin/env python3
"""Create the final Object-retrain training/evaluation report from durable artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN_ID = "shared-spatialpro-hpcm3-object-nonvision-r64-b2-ga8-seed7-20260917-2000"
MECHANICAL_ROOT = Path("/media/cjs/shared/linux/VLA-Adapter/shared_vision_policy_ckpts")
TRAIN_RUN = MECHANICAL_ROOT / "runs" / RUN_ID
TRAIN_LOG = MECHANICAL_ROOT / "logs" / f"{RUN_ID}.log"
CHECKPOINT = MECHANICAL_ROOT / "runs" / f"{RUN_ID}--200000_chkpt"
RESULT_ROOT = REPO / "results" / "hpcm3_object_retrain_eval"
SPATIAL_RESULT_ROOT = REPO / "results" / "hpcm3_libero4" / "spatial"
WANDB_URL = "https://wandb.ai/cjs838237678-nanjing-university/vla-adapter-shared-vision/runs/9tx7sqhs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def component_files() -> list[Path]:
    files = [CHECKPOINT / "model.safetensors"]
    files.extend(sorted(CHECKPOINT.glob("action_head--*_checkpoint.pt")))
    files.extend(sorted(CHECKPOINT.glob("proprio_projector--*_checkpoint.pt")))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def per_task_rows(summary: dict) -> list[str]:
    rows = []
    for item in summary.get("per_task", []):
        task_id = item.get("task_id", "")
        description = str(item.get("task_description", item.get("description", ""))).replace("|", "\\|")
        episodes = item.get("episodes", item.get("total_episodes", 0))
        successes = item.get("successes", item.get("success_episodes", 0))
        rate = item.get("success_rate", successes / episodes if episodes else 0.0)
        rows.append(f"| {task_id} | {description} | {successes}/{episodes} | {rate:.1%} |")
    return rows


def validate_formal_summary(summary: dict, expected_suite: str) -> None:
    if summary.get("suite") != expected_suite:
        raise ValueError(f"Expected suite {expected_suite}, got {summary.get('suite')}")
    if summary.get("task_count") != 10:
        raise ValueError(f"Expected 10 tasks for {expected_suite}")
    if summary.get("total_episodes") != 500:
        raise ValueError(f"Expected 500 episodes for {expected_suite}")
    per_task = summary.get("per_task", [])
    if len(per_task) != 10 or any(item.get("episodes") != 50 for item in per_task):
        raise ValueError(f"Expected 50 episodes for every task in {expected_suite}")


def main() -> None:
    object_summary = load_json(RESULT_ROOT / "object" / "summary.json")
    spatial_summary = load_json(SPATIAL_RESULT_ROOT / "summary.json")
    validate_formal_summary(object_summary, "libero_object")
    validate_formal_summary(spatial_summary, "libero_spatial")
    vision_metadata = load_json(TRAIN_RUN / "shared_vision_metadata.json")
    log_text = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")
    losses = [float(value) for value in re.findall(r"curr:\s+([0-9.]+)", log_text)]
    if "Max step 200005 reached" not in log_text:
        raise RuntimeError("Training log does not contain the max-step completion marker")

    checkpoint_hashes = {path.name: sha256(path) for path in component_files()}
    checkpoint_time = datetime.fromtimestamp((CHECKPOINT / "model.safetensors").stat().st_mtime).astimezone()
    report = [
        "# HPCM3 Shared-Vision Object Retraining Experiment",
        "",
        "## Conclusion",
        "",
        (
            "This experiment froze the Spatial-Pro vision tower, HPCM codec, and HPCM3 adapter, "
            "then retrained only the non-vision Object policy components. Object uses that newly "
            "trained policy checkpoint; Spatial uses the retained Spatial-Pro policy with the same "
            "frozen HPCM/HPCM3 visual path. This is a shared-vision, suite-specific-policy comparison, "
            "not cross-suite use of the Object policy."
        ),
        "",
        "| Evaluation suite | Success | Episodes | Exact HPCM3 train-frame matches |",
        "| --- | ---: | ---: | ---: |",
        f"| Object | {object_summary['success_rate']:.1%} ({object_summary['success_episodes']}) | {object_summary['total_episodes']} | {object_summary['exact_train_frame_matches']} |",
        f"| Spatial | {spatial_summary['success_rate']:.1%} ({spatial_summary['success_episodes']}) | {spatial_summary['total_episodes']} | {spatial_summary['exact_train_frame_matches']} |",
        "",
        "## Training",
        "",
        f"- Run ID: `{RUN_ID}`",
        f"- W&B: {WANDB_URL}",
        "- Dataset: `libero_object_no_noops`",
        "- Seed: `7`",
        "- Optimizer steps requested/completed: `200005`",
        "- Saved evaluation checkpoint: step `200000` (the final configured save boundary)",
        "- Batch size / gradient accumulation / effective batch: `2 / 8 / 16`",
        "- Learning rate / LoRA rank: `2e-4 / 64`",
        "- Trainable modules: non-vision LoRA, action head, proprio projector",
        "- Frozen modules: HPCM, HPCM3 adapter, Spatial-Pro posterior vision tower",
        f"- Sampled current-action L1 loss: first `{losses[0]:.6f}`, last `{losses[-1]:.6f}`, minimum `{min(losses):.6f}`",
        f"- Checkpoint timestamp: `{checkpoint_time.isoformat()}`",
        "",
        "### Frozen component identity",
        "",
        f"- Spatial-Pro vision SHA256: `{vision_metadata['vision_backbone_sha256']}`",
        f"- HPCM SHA256: `{vision_metadata['hpcm3']['hpcm_checkpoint_sha256']}`",
        f"- HPCM3 adapter SHA256: `{vision_metadata['hpcm3']['adapter_checkpoint_sha256']}`",
        "",
        "### Policy checkpoint identity",
        "",
    ]
    report.extend(f"- `{name}`: `{digest}`" for name, digest in checkpoint_hashes.items())
    report.extend(
        [
            "",
            "## Evaluation protocol",
            "",
            "- 10 tasks per suite and 50 episodes per task (500 episodes per suite).",
            "- Fixed seed 7 and official default LIBERO initial states.",
            "- HPCM3 rollout frame hashing enabled; hashes are compared with the cached HPCM3 training split.",
            "- Object uses Object action/proprio normalization; Spatial uses Spatial normalization from Spatial-Pro.",
            "- Spatial is the completed formal Spatial-Pro/HPCM3 run (491/500), retained rather than needlessly rerun.",
            "- No rollout videos were saved; JSONL episode and frame-audit records are retained.",
            "",
            "## Object per-task results",
            "",
            "| Task | Description | Success | Rate |",
            "| ---: | --- | ---: | ---: |",
            *per_task_rows(object_summary),
            "",
            "## Spatial-Pro per-task results",
            "",
            "| Task | Description | Success | Rate |",
            "| ---: | --- | ---: | ---: |",
            *per_task_rows(spatial_summary),
            "",
            "## Artifact locations",
            "",
            f"- Mechanical checkpoint: `{CHECKPOINT}`",
            "- Object evidence: `results/hpcm3_object_retrain_eval/object/`",
            "- Spatial evidence: `results/hpcm3_libero4/spatial/`",
            "- Per-frame `audit.jsonl` files remain local because they are large; exact-match totals are retained in each `summary.json`.",
            "",
        ]
    )
    (RESULT_ROOT / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
