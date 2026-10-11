"""Read-only Object Epoch-29 raw-HPCM-input A/B on static LIBERO initial states.

The Object-Pro teacher receives identical policy-processed pixels in both arms.
No environment step, action, training, or checkpoint write is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import yaml

from fullvisual_offline_diagnostic import (
    build_hf_tower,
    build_training_model,
    digest,
    load_adapter,
    rgb_hash,
    sample_simulator,
)
from experiments.robot.openvla_utils import prepare_images_for_vla
from inference.model.combined_vision_model import CombinedVisionModel
from inference.model.vision_truncation_2 import skip_vision_backbone_layers
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor


def aggregate(rows: list[dict], metric: str, arm: str, view: str) -> float:
    selected = [row[metric] for row in rows if row["arm"] == arm and (view == "all" or row["view"] == view)]
    return float(np.mean(selected))


def clustered_interval(rows: list[dict], metric: str, view: str, seed: int, repeats: int) -> dict:
    tasks = sorted({row["task_id"] for row in rows})
    paired = []
    for task in tasks:
        a = [row[metric] for row in rows if row["task_id"] == task and row["arm"] == "A" and (view == "all" or row["view"] == view)]
        b = [row[metric] for row in rows if row["task_id"] == task and row["arm"] == "B" and (view == "all" or row["view"] == view)]
        assert len(a) == len(b) == (6 if view == "all" else 3)
        paired.append(float(np.mean(b) - np.mean(a)))
    rng = np.random.default_rng(seed)
    values = np.asarray(paired)
    draws = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {
        "mean_B_minus_A": float(values.mean()),
        "task_bootstrap_95pct": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "tasks_B_better": int((values < 0).sum()),
        "tasks_B_worse": int((values > 0).sum()),
        "task_differences_B_minus_A": dict(zip(map(str, tasks), map(float, values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--vla-dir", type=Path, required=True)
    parser.add_argument("--previous-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    tf.config.set_visible_devices([], "GPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for full Object-Pro visual tower")

    old = json.loads(args.previous_json.read_text())
    assert old["sampling"]["sim_state_indices"] == [7, 23, 41]
    assert old["sampling"]["sim_task_limit"] == 10
    samples = sample_simulator([7, 23, 41], 10)
    old_samples = [item for item in old["sampling"]["samples"] if item["source"] == "simulator_initial_state"]
    assert len(samples) == len(old_samples) == 60
    sample_metadata = []
    matching_previous_rgb = 0
    for current, previous in zip(samples, old_samples):
        assert (current["task_id"], current["state_index"], current["view"], current["state_sha256"]) == (
            previous["task_id"], previous["state_index"], previous["view"], previous["state_sha256"]
        )
        current_hash = rgb_hash(current["rgb"])
        same_rgb = current_hash == previous["rgb_sha256"]
        matching_previous_rgb += int(same_rgb)
        sample_metadata.append({key: value for key, value in current.items() if key != "rgb"} | {
            "rgb_sha256": current_hash, "matches_previous_rgb_sha256": same_rgb,
        })
    print(f"Verified same 60 task/state/view keys; {matching_previous_rgb}/60 exact RGB hashes match previous render", flush=True)

    config = yaml.safe_load(args.train_config.read_text())
    model = build_training_model(config, device)
    adapter = copy.deepcopy(load_adapter(model, args.adapter)).half().to(device).eval()
    hf, weight_count = build_hf_tower(args.vla_dir, device)
    hf.to(torch.bfloat16)
    processor = PrismaticImageProcessor.from_pretrained(args.vla_dir, local_files_only=True)
    policy_cfg = SimpleNamespace(center_crop=True)

    # Teacher uses the intact Object-Pro tower. Only afterwards truncate the
    # same loaded tower for the actual HPCM3 inference posterior path.
    teachers = []
    processed_inputs = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for sample in samples:
            pil = prepare_images_for_vla([sample["rgb"]], policy_cfg)[0]
            pixels = processor.apply_transform(pil).unsqueeze(0).to(device, dtype=torch.bfloat16)
            processed_inputs.append(pixels)
            teachers.append(hf(pixels).float().cpu())
    assert skip_vision_backbone_layers(hf, 3)
    hf.to(torch.float16)
    proxy = SimpleNamespace(hpcm_encoder=model.hpcm_encoder, adapter=adapter, vision_backbone=hf)

    rows = []
    with torch.inference_mode():
        for index, (sample, pixels, teacher) in enumerate(zip(samples, processed_inputs, teachers)):
            raw = torch.from_numpy(sample["rgb"].copy()).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(127.5).sub_(1)
            processed = F.interpolate(pixels[:, 3:].float(), size=(256, 256), mode="bicubic", align_corners=False)
            for arm, hpcm_input in (("A", processed), ("B", raw)):
                output = CombinedVisionModel._forward_with_hpcm(proxy, hpcm_input)
                dino = output["vision_dino_features"][0, 0].float().cpu()
                siglip = output["vision_siglip_features"][0, 0].float().cpu()
                dino_mse = float((dino - teacher[0, :, :1024]).square().mean())
                siglip_mse = float((siglip - teacher[0, :, 1024:]).square().mean())
                rows.append({
                    "arm": arm, "task_id": sample["task_id"], "state_index": sample["state_index"],
                    "view": sample["view"], "rgb_sha256": sample_metadata[index]["rgb_sha256"],
                    "dino_mse": dino_mse, "siglip_mse": siglip_mse,
                    "full_visual_loss": siglip_mse + 5 * dino_mse,
                })
            if (index + 1) % 10 == 0:
                print(f"Evaluated {index + 1}/{len(samples)} images", flush=True)

    metrics = {}
    for view in ("all", "agent", "wrist"):
        metrics[view] = {}
        for metric in ("full_visual_loss", "dino_mse", "siglip_mse"):
            a = aggregate(rows, metric, "A", view)
            b = aggregate(rows, metric, "B", view)
            metrics[view][metric] = {
                "A": a, "B": b, "B_minus_A": b - a, "relative_change_pct": (b / a - 1) * 100,
                **clustered_interval(rows, metric, view, args.seed, 20000),
            }

    result = {
        "schema_version": 1,
        "seed": args.seed,
        "protocol": {
            "teacher": "Object-Pro intact HF visual tower, policy JPEG/Lanczos3 224px + 0.9 center crop + HF image processor, BF16 weights/FP32 execution",
            "A": "hpcm_raw_input_bypass=false: processor SigLIP channels bicubic-resized to 256px for HPCM",
            "B": "hpcm_raw_input_bypass=true: original simulator RGB 256px scaled to [-1,1] for HPCM",
            "student": "Epoch-29 adapter FP16, HF posterior blocks[3:-1], batch=1",
            "loss": "SigLIP token MSE + 5*DINO token MSE, per-image mean then image average",
        },
        "checkpoints": {
            "adapter": {"path": str(args.adapter), "sha256": digest(args.adapter)},
            "vla": {"path": str(args.vla_dir / "model.safetensors"), "sha256": digest(args.vla_dir / "model.safetensors"), "visual_weight_tensors": weight_count},
            "hpcm": {"path": config["model"]["hpcm"]["checkpoint"], "sha256": digest(Path(config["model"]["hpcm"]["checkpoint"]))},
        },
        "sampling": {"suite": "libero_object", "task_count": 10, "initial_state_indices": [7, 23, 41], "views": ["agent", "wrist"], "image_count": 60, "env_steps": 0, "previous_rgb_hash_match_count": matching_previous_rgb, "samples": sample_metadata},
        "metrics": metrics,
        "per_image_metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
