"""Read-only feature diagnostic for Object full-visual HPCM3 adapters.

Renders static LIBERO initial states only: no env.step(), policy action, training,
or checkpoint writes. The output contains metrics and hashes, never RGB tensors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import yaml
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for root in (REPO, CODE):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
)
from experiments.robot.openvla_utils import prepare_images_for_vla  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from model.model import AlignModel  # noqa: E402
from inference.model.combined_vision_model import CombinedVisionModel  # noqa: E402
from inference.model.vision_truncation_2 import skip_vision_backbone_layers  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor  # noqa: E402


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def rgb_hash(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def sample_clean(manifest: Path, episodes_per_suite: int, seed: int) -> list[dict]:
    split = json.loads(manifest.read_text())
    assert split["split_unit"] == "trajectory"
    assert not (set(split["train_episodes"]) & set(split["val_episodes"]))
    val_episodes = [Path(p) for p in split["val_episodes"]]
    result = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        episodes = sorted(p for p in val_episodes if p.parent.name == f"{suite}_no_noops")
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{suite}".encode()).digest()[:8], "little"))
        assert len(episodes) >= episodes_per_suite, (suite, len(episodes))
        for episode in rng.sample(episodes, episodes_per_suite):
            mains = sorted(p for p in episode.glob("step_*_image.png") if "_wrist_image" not in p.name)
            assert mains, episode
            main = rng.choice(mains)
            wrist = main.with_name(main.name.replace("_image.png", "_wrist_image.png"))
            assert wrist.is_file(), wrist
            for view, path in (("agent", main), ("wrist", wrist)):
                result.append({"source": "clean", "suite": suite, "view": view, "path": str(path)})
    return result


def sample_exact_validation(manifest: Path, max_samples: int, seed: int) -> list[dict]:
    """Mirror LiberoImageDataset's val_max_samples selection exactly."""
    split = json.loads(manifest.read_text())
    assert split["split_unit"] == "trajectory"
    assert not (set(split["train_episodes"]) & set(split["val_episodes"]))
    paths = sorted({
        path
        for episode in split["val_episodes"]
        for path in Path(episode).glob("step_*_image.png")
    })
    indices = list(range(len(paths)))
    random.Random(seed + 1).shuffle(indices)
    selected = [paths[index] for index in sorted(indices[:max_samples])]
    assert len(selected) == max_samples
    result = []
    for path in selected:
        view = "wrist" if path.name.endswith("_wrist_image.png") else "agent"
        result.append({
            "source": "clean", "suite": path.parent.parent.name.removesuffix("_no_noops"),
            "view": view, "path": str(path),
        })
    return result


def sample_simulator(state_indices: list[int], task_limit: int) -> list[dict]:
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    assert 1 <= task_limit <= suite.n_tasks
    result = []
    for task_id in range(task_limit):
        states = suite.get_task_init_states(task_id)
        assert max(state_indices) < len(states), (task_id, len(states))
        env, description = get_libero_env(suite.get_task(task_id), "openvla", resolution=256)
        try:
            for state_index in state_indices:
                # Static initial-state render only; deliberately never call env.step().
                obs = env.set_init_state(states[state_index])
                for view, rgb in (
                    ("agent", get_libero_image(obs)),
                    ("wrist", get_libero_wrist_image(obs)),
                ):
                    rgb = np.ascontiguousarray(rgb)
                    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
                    result.append({
                        "source": "simulator_initial_state", "suite": "libero_object",
                        "task_id": task_id, "task_description": description,
                        "state_index": state_index, "view": view, "rgb": rgb,
                        "state_sha256": hashlib.sha256(np.ascontiguousarray(states[state_index]).tobytes()).hexdigest(),
                    })
        finally:
            env.close()
    return result


def load_rgb(sample: dict) -> np.ndarray:
    if "rgb" in sample:
        return sample["rgb"]
    with Image.open(sample["path"]) as image:
        image = image.convert("RGB")
        side = min(image.size)
        left = (image.width - side) / 2
        top = (image.height - side) / 2
        image = image.crop((left, top, left + side, top + side))
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def build_training_model(config: dict, device: torch.device) -> AlignModel:
    m = config["model"]
    model = AlignModel(
        hpcm_root=m["hpcm"]["hpcm_root"],
        hpcm_checkpoint=m["hpcm"]["checkpoint"],
        hpcm_model_name=m["hpcm"]["model_name"],
        hpcm_scale_table_levels=m["hpcm"]["scale_table_levels"],
        freeze_hpcm_encoder=True,
        vision_backbone_id=m["vision"]["backbone_id"],
        vision_backbone_checkpoint=Path(m["vision"]["checkpoint"]),
        vision_backbone_max_layer=m["vision"]["max_layer"],
        image_resize_strategy=m["vision"]["image_resize_strategy"],
        image_sequence_len=m["vision"]["image_sequence_len"],
        adapter_config={key: value for key, value in m["adapter"].items() if key != "train_adapter"},
        freeze_vision_backbone=True,
        train_adapter=False,
        loss_config=config["loss"],
        augmentation_config={"enabled": False},
    )
    assert model.loss_scope == "full_visual" and model.posterior_start_layer == 3
    return model.to(device).eval()


def load_adapter(model: AlignModel, path: Path) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    adapter = copy.deepcopy(model.adapter)
    adapter.load_state_dict(state, strict=True)
    return adapter.eval()


def build_hf_tower(vla_dir: Path, device: torch.device) -> tuple[PrismaticVisionBackbone, int]:
    cfg = json.loads((vla_dir / "config.json").read_text())
    tower = PrismaticVisionBackbone(
        use_fused_vision_backbone=cfg["use_fused_vision_backbone"],
        image_sizes=cfg["image_sizes"],
        timm_model_ids=cfg["timm_model_ids"],
        timm_override_act_layers=cfg["timm_override_act_layers"],
    )
    state = tower.state_dict()
    with safe_open(str(vla_dir / "model.safetensors"), framework="pt", device="cpu") as source:
        source_keys = {key.removeprefix("vision_backbone.") for key in source.keys() if key.startswith("vision_backbone.")}
        missing = set(state) - source_keys
        unexpected = source_keys - set(state)
        if missing or unexpected:
            raise RuntimeError(f"HF tower weight mismatch: missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}")
        with torch.no_grad():
            for key, target in state.items():
                target.copy_(source.get_tensor(f"vision_backbone.{key}"))
    return tower.to(device).eval(), len(state)


def add_loss(rows: list[dict], sample: dict, epoch: str, protocol: str,
             dino: torch.Tensor, siglip: torch.Tensor,
             teacher_dino: torch.Tensor, teacher_siglip: torch.Tensor) -> None:
    dino_mse = float((dino.float() - teacher_dino.to(dino.device).float()).square().mean())
    siglip_mse = float((siglip.float() - teacher_siglip.to(siglip.device).float()).square().mean())
    rows.append({
        "source": sample["source"], "suite": sample["suite"], "view": sample["view"],
        "task_id": sample.get("task_id"), "epoch": epoch, "protocol": protocol,
        "dino_mse": dino_mse, "siglip_mse": siglip_mse,
        "full_visual_loss": siglip_mse + 5.0 * dino_mse,
    })


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["epoch"], row["protocol"], row["source"], row["suite"], row["view"])].append(row)
        groups[(row["epoch"], row["protocol"], row["source"], row["suite"], "all")].append(row)
    result = []
    for (epoch, protocol, source, suite, view), items in sorted(groups.items()):
        metrics = {}
        for key in ("full_visual_loss", "dino_mse", "siglip_mse"):
            values = np.asarray([x[key] for x in items], dtype=np.float64)
            metrics[key] = float(values.mean())
            metrics[f"{key}_std"] = float(values.std())
            metrics[f"{key}_p90"] = float(np.quantile(values, 0.9))
        result.append({
            "epoch": epoch, "protocol": protocol, "source": source,
            "suite": suite, "view": view, "n": len(items),
            **metrics,
        })
    return result


def summarize_simulator_tasks(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["source"] == "simulator_initial_state":
            groups[(row["epoch"], row["protocol"], row["task_id"], row["view"])].append(row)
    return [
        {"epoch": epoch, "protocol": protocol, "task_id": task_id, "view": view,
         "n": len(items), "full_visual_loss": float(np.mean([x["full_visual_loss"] for x in items]))}
        for (epoch, protocol, task_id, view), items in sorted(groups.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter10", type=Path, required=True)
    parser.add_argument("--adapter29", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=CODE / "configs/train_object_fullvisual_aug_libero4.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vla-dir", type=Path, default=REPO / "outputs/LIBERO-Object-Pro")
    parser.add_argument("--clean-episodes-per-suite", type=int, default=16)
    parser.add_argument("--exact-validation-subset", action="store_true")
    parser.add_argument("--validation-max-samples", type=int, default=512)
    parser.add_argument("--sim-state-indices", nargs="+", type=int, default=[7, 23, 41])
    parser.add_argument("--sim-task-limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tf.config.set_visible_devices([], "GPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This diagnostic expects CUDA for the full visual towers")
    config = yaml.safe_load(args.train_config.read_text())
    clean = (sample_exact_validation(args.manifest, args.validation_max_samples, 42)
             if args.exact_validation_subset
             else sample_clean(args.manifest, args.clean_episodes_per_suite, args.seed))
    simulator = sample_simulator(args.sim_state_indices, args.sim_task_limit)
    samples = clean + simulator
    print(f"Selected {len(clean)} clean and {len(simulator)} static simulator images", flush=True)

    model = build_training_model(config, device)
    adapters = {
        "10": load_adapter(model, args.adapter10),
        "29": load_adapter(model, args.adapter29),
    }
    hf, hf_weight_count = build_hf_tower(args.vla_dir, device)
    processor = PrismaticImageProcessor.from_pretrained(args.vla_dir, local_files_only=True)
    policy_cfg = SimpleNamespace(center_crop=True)
    rows: list[dict] = []
    preprocessing = defaultdict(list)
    hf_targets: dict[int, torch.Tensor] = {}
    parity: dict[str, object] = {}
    train_outputs: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    parity_rows: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            rgb = [load_rgb(item) for item in batch]
            raw = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).to(device).float().div_(127.5).sub_(1.0)
            train_pixels = model._prepare_vision_inputs(raw)
            with torch.autocast("cuda", dtype=torch.float16):
                train_teacher = model.vision_backbone(train_pixels)
            if start == 0:
                with torch.autocast("cuda", dtype=torch.float16):
                    hf_teacher = hf(torch.cat([train_pixels["dino"], train_pixels["siglip"]], dim=1))
                parity["teacher_same_pixels_mae"] = float((hf_teacher.float() - train_teacher.float()).abs().mean())
                parity["teacher_same_pixels_max_abs"] = float((hf_teacher.float() - train_teacher.float()).abs().max())
                parity["teacher_shape"] = list(train_teacher.shape)
                hf.to(torch.bfloat16)
            hpcm = model.extract_hpcm_features(raw)
            if start == 0:
                direct_hpcm = model.hpcm_encoder(raw[:1].add(1.0).mul(0.5))
                parity["hpcm_same_rgb_max_abs"] = float((hpcm[:1] - direct_hpcm).abs().max())
            with torch.autocast("cuda", dtype=torch.float16):
                for epoch, adapter in adapters.items():
                    boundary_siglip, boundary_dino = adapter(hpcm)
                    dino = model._run_frozen_posterior(model.vision_backbone.dino_featurizer, boundary_dino)
                    siglip = model._run_frozen_posterior(model.vision_backbone.siglip_featurizer, boundary_siglip)
                    for offset, sample in enumerate(batch):
                        add_loss(rows, sample, epoch, "train_canonical", dino[offset], siglip[offset],
                                 train_teacher[offset, :, :1024], train_teacher[offset, :, 1024:])
                    for offset in range(len(batch)):
                        train_outputs[(epoch, start + offset)] = (
                            dino[offset].detach().cpu(), siglip[offset].detach().cpu()
                        )

            policy_pils = [prepare_images_for_vla([image], policy_cfg)[0] for image in rgb]
            policy_pixels = torch.stack([processor.apply_transform(image) for image in policy_pils]).to(device, dtype=torch.bfloat16)
            with torch.autocast("cuda", enabled=False):
                policy_teacher = hf(policy_pixels)
            for offset, sample in enumerate(batch):
                hf_targets[start + offset] = policy_teacher[offset].cpu()
            resized_siglip = torch.nn.functional.interpolate(
                policy_pixels[:, 3:].float(), size=(256, 256), mode="bicubic", align_corners=False
            )
            for offset, sample in enumerate(batch):
                key = (sample["source"], sample["suite"], sample["view"])
                preprocessing[(key, "hpcm_input_mae")].append(float((resized_siglip[offset] - raw[offset]).abs().mean()))
                preprocessing[(key, "dino_teacher_input_mae")].append(
                    float((policy_pixels[offset, :3].float() - train_pixels["dino"][offset]).abs().mean()))
                preprocessing[(key, "siglip_teacher_input_mae")].append(
                    float((policy_pixels[offset, 3:].float() - train_pixels["siglip"][offset]).abs().mean()))
            if (start // args.batch_size + 1) % 10 == 0:
                print(f"Canonical/teacher pass: {min(start + args.batch_size, len(samples))}/{len(samples)}", flush=True)

        # Exact inference function, with the same layer skip and half-precision
        # quantization used by get_custom_vla_model(). No VLA action is called.
        assert skip_vision_backbone_layers(hf, 3)
        hf.to(torch.float16)
        inf_adapters = {epoch: copy.deepcopy(adapter).half().eval() for epoch, adapter in adapters.items()}
        proxies = {epoch: SimpleNamespace(hpcm_encoder=model.hpcm_encoder, adapter=adapter,
                                          vision_backbone=hf) for epoch, adapter in inf_adapters.items()}
        parity["inference_posterior_blocks"] = {
            "dino": len(hf.featurizer.blocks), "siglip": len(hf.fused_featurizer.blocks)
        }
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            rgb = [load_rgb(item) for item in batch]
            raw = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).to(device).float().div_(127.5).sub_(1.0)
            policy_pils = [prepare_images_for_vla([image], policy_cfg)[0] for image in rgb]
            policy_pixels = torch.stack([processor.apply_transform(image) for image in policy_pils]).to(device, dtype=torch.bfloat16)
            hpcm_input = torch.nn.functional.interpolate(
                policy_pixels[:, 3:], size=(256, 256), mode="bicubic", align_corners=False
            )
            for epoch, proxy in proxies.items():
                output = CombinedVisionModel._forward_with_hpcm(proxy, hpcm_input)
                for offset, sample in enumerate(batch):
                    teacher = hf_targets[start + offset]
                    add_loss(rows, sample, epoch, "rollout_effective", output["vision_dino_features"][offset, 0],
                             output["vision_siglip_features"][offset, 0], teacher[:, :1024], teacher[:, 1024:])
                canonical = CombinedVisionModel._forward_with_hpcm(proxy, raw)
                for offset, sample in enumerate(batch):
                    train_dino, train_siglip = train_outputs[(epoch, start + offset)]
                    for branch, train_feature, inference_feature in (
                        ("dino", train_dino, canonical["vision_dino_features"][offset, 0]),
                        ("siglip", train_siglip, canonical["vision_siglip_features"][offset, 0]),
                    ):
                        delta = inference_feature.float().cpu() - train_feature.float()
                        parity_rows.append({
                            "epoch": epoch, "branch": branch, "source": sample["source"],
                            "view": sample["view"], "mae": float(delta.abs().mean()),
                            "relative_l2": float(delta.norm() / train_feature.float().norm().clamp_min(1e-12)),
                        })
                if start == 0:
                    parity[f"epoch{epoch}_fused_dino_first"] = bool(torch.equal(
                        canonical["fused_features"][0, 0, :, :1024], canonical["vision_dino_features"][0, 0]
                    ))
                    parity[f"epoch{epoch}_fused_siglip_second"] = bool(torch.equal(
                        canonical["fused_features"][0, 0, :, 1024:], canonical["vision_siglip_features"][0, 0]
                    ))
            if (start // args.batch_size + 1) % 10 == 0:
                print(f"Inference-path pass: {min(start + args.batch_size, len(samples))}/{len(samples)}", flush=True)

    metadata = []
    for sample in samples:
        record = {key: value for key, value in sample.items() if key != "rgb"}
        record["rgb_sha256"] = rgb_hash(load_rgb(sample))
        metadata.append(record)
    parity_summary = []
    for epoch in ("10", "29"):
        for branch in ("dino", "siglip"):
            for source in ("clean", "simulator_initial_state"):
                values = [row for row in parity_rows if row["epoch"] == epoch and row["branch"] == branch
                          and row["source"] == source]
                parity_summary.append({
                    "epoch": epoch, "branch": branch, "source": source, "n": len(values),
                    "mae_mean": float(np.mean([row["mae"] for row in values])),
                    "mae_p90": float(np.quantile([row["mae"] for row in values], 0.9)),
                    "mae_max": float(np.max([row["mae"] for row in values])),
                    "mae_over_0p02": sum(row["mae"] > 0.02 for row in values),
                    "relative_l2_mean": float(np.mean([row["relative_l2"] for row in values])),
                })
    parity["student_same_rgb_summary"] = parity_summary
    output = {
        "schema_version": 1, "seed": args.seed,
        "protocols": {
            "train_canonical": "256px decoded RGB to HPCM; bilinear 224px teacher; standalone full tower; FP16 autocast",
            "rollout_effective": "JPEG/Lanczos3 224px + 0.9 center crop + HF processor; SigLIP pixels bicubic back to 256px for HPCM; HF posterior with FP16 weights/FP32 execution",
        },
        "checkpoints": {
            "adapter10": {"path": str(args.adapter10), "sha256": digest(args.adapter10)},
            "adapter29": {"path": str(args.adapter29), "sha256": digest(args.adapter29)},
            "vla_vision": {"path": str(args.vla_dir / "model.safetensors"), "sha256": digest(args.vla_dir / "model.safetensors"), "weight_tensors": hf_weight_count},
            "hpcm": {"path": config["model"]["hpcm"]["checkpoint"], "sha256": digest(Path(config["model"]["hpcm"]["checkpoint"]))},
            "manifest": {"path": str(args.manifest), "sha256": digest(args.manifest)},
        },
        "sampling": {"clean_episodes_per_suite": args.clean_episodes_per_suite,
                     "exact_validation_subset": args.exact_validation_subset,
                     "validation_max_samples": args.validation_max_samples if args.exact_validation_subset else None,
                     "sim_state_indices": args.sim_state_indices, "sim_task_limit": args.sim_task_limit,
                     "simulator_actions_taken": 0, "samples": metadata},
        "parity": parity,
        "preprocessing": [
            {"source": source, "suite": suite, "view": view, "metric": metric,
             "mean": float(np.mean(values)), "n": len(values)}
            for ((source, suite, view), metric), values in sorted(preprocessing.items())
        ],
        "metrics": summarize(rows),
        "simulator_task_metrics": summarize_simulator_tasks(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
