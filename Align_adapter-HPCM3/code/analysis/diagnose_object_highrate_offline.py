"""Read-only, paired Object-Pro/HPCM3 feature and oracle-injection diagnostic.

Uses no formal LIBERO evaluation initial states, simulator steps, training, or
checkpoint writes. The existing fresh-reset simulator RGB files are read only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for root in (REPO, CODE, REPO / "LIBERO"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head, get_processor, get_proprio_projector, get_vla,
    get_vla_action, prepare_images_for_vla,
)
from inference.model.model import get_custom_vla_model  # noqa: E402


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def rgb_from(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    assert rgb.shape == (256, 256, 3), (path, rgb.shape)
    return rgb


def select_samples(config: dict, training: dict) -> tuple[list[dict], dict]:
    split = json.loads(Path(config["split_manifest"]).read_text())
    assert split["split_unit"] == "trajectory"
    train_eps, val_eps = set(split["train_episodes"]), set(split["val_episodes"])
    assert not train_eps & val_eps and len(train_eps) == 408 and len(val_eps) == 46
    seed = int(training["data"]["seed"])
    all_train = sorted(p for ep in train_eps for p in Path(ep).glob("step_*_image.png"))
    assert len(all_train) == 120552
    indices = list(range(len(all_train)))
    random.Random(seed).shuffle(indices)
    selected = {all_train[index] for index in indices[:training["data"]["train_max_samples"]]}
    assert len(selected) == 16384
    n = int(config["rl_ds_trajectories_per_group"])
    rng = random.Random(int(config["seed"]))
    samples = []
    counts = {}
    for view in ("agent", "wrist"):
        suffix = "_wrist_image.png" if view == "wrist" else "_image.png"
        def candidates(episodes: set[str], want_selected: bool | None) -> dict[str, list[Path]]:
            result = {}
            for episode in sorted(episodes):
                paths = sorted(p for p in Path(episode).glob(f"step_*{suffix}")
                               if want_selected is None or (p in selected) == want_selected)
                if paths:
                    result[episode] = paths
            return result
        inside = candidates(train_eps, True)
        outside = candidates(train_eps, False)
        paired_eps = sorted(set(inside) & set(outside))
        val = candidates(val_eps, None)
        assert len(paired_eps) >= n and len(val) >= n
        chosen_train = rng.sample(paired_eps, n)
        chosen_val = rng.sample(sorted(val), n)
        for group, episodes, pool in (
            ("train_selected", chosen_train, inside),
            ("train_unsampled", chosen_train, outside),
            ("trajectory_held_out", chosen_val, val),
        ):
            for episode in episodes:
                path = rng.choice(pool[episode])
                samples.append({"group": group, "view": view, "episode": Path(episode).name,
                                "path": str(path)})
        counts[view] = {"train_selected_available": sum(len(v) for v in inside.values()),
                        "train_unsampled_available": sum(len(v) for v in outside.values()),
                        "held_out_available": sum(len(v) for v in val.values()),
                        "paired_train_episodes_available": len(paired_eps)}

    sim = json.loads(Path(config["sim_manifest"]).read_text())
    assert sim["protected_state_collisions"] == 0 and len(sim["samples"]) == 40
    for item in sim["samples"]:
        samples.append({"group": "simulator_fresh_reset", "view": item["view"],
                        "task_id": item["task_id"], "reset_index": item["reset_index"],
                        "state_sha256": item["state_sha256"],
                        "path": str(Path(config["sim_image_root"]) / item["path"]),
                        "expected_rgb_sha256": item["rgb_sha256"]})
    assert len(samples) == 6 * n + 40
    return samples, counts


def policy_cfg(eval_config: dict) -> SimpleNamespace:
    raw = dict(eval_config["model"])
    raw["preserve_checkpoint_files"] = True
    return SimpleNamespace(**raw)


def pixels(rgb: np.ndarray, image_processor, cfg) -> torch.Tensor:
    pil = prepare_images_for_vla([rgb], cfg)[0]
    return image_processor.apply_transform(pil).to("cuda", dtype=torch.bfloat16).unsqueeze(0)


def raw_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to("cuda").float().div_(127.5).sub_(1.0)


def action_observation(agent: np.ndarray, wrist: np.ndarray, raw: bool) -> dict:
    obs = {"full_image": agent.copy(), "wrist_image": wrist.copy(),
           "state": np.zeros(8, dtype=np.float32)}
    if raw:
        obs["_hpcm_raw_images"] = [agent.copy(), wrist.copy()]
    return obs


def action(cfg, vla, processor, action_head, proprio_projector,
           agent: np.ndarray, wrist: np.ndarray, description: str, raw: bool) -> np.ndarray:
    cfg.hpcm_raw_input_bypass = raw
    torch.manual_seed(9191)
    np.random.seed(9191)
    values = get_vla_action(
        cfg, vla, processor, action_observation(agent, wrist, raw), description,
        action_head=action_head, proprio_projector=proprio_projector,
        use_film=cfg.use_film, use_minivlm=cfg.use_minivlm,
    )
    return np.asarray(values, dtype=np.float64)


def feature_row(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[dict, np.ndarray, np.ndarray]:
    ref = reference.to(candidate.device).float().reshape(256, -1)
    cand = candidate.float().reshape(256, -1)
    assert ref.shape == cand.shape
    token_mse = (cand - ref).square().mean(dim=-1)
    token_cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=-1)
    row = {"mse": float(token_mse.mean()), "cosine": float(token_cosine.mean()),
           "relative_l2": float((cand - ref).norm() / ref.norm().clamp_min(1e-12)),
           "max_abs": float((cand - ref).abs().max()),
           "p95_token_mse": float(torch.quantile(token_mse, 0.95)),
           "worst_token": int(torch.argmax(token_mse))}
    return row, token_mse.cpu().numpy(), token_cosine.cpu().numpy()


def posterior_with_prefix(reference: dict, combined) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the actual suffix blocks with teacher patch AND prefix tokens."""
    restored = []
    for prefix_key, patch_key, suffix in (
        ("dino_prefix", "dino", combined.vision_backbone.featurizer),
        ("siglip_prefix", "siglip", combined.vision_backbone.fused_featurizer),
    ):
        prefix = reference[prefix_key].to("cuda", dtype=torch.float32)
        patch = reference[patch_key].to("cuda", dtype=torch.float32)
        full_tokens = torch.cat([prefix, patch], dim=1)
        original_dtype = next(suffix.parameters()).dtype
        suffix.to(torch.float32)
        try:
            for block in list(suffix.blocks)[:-1]:
                full_tokens = block(full_tokens)
        finally:
            suffix.to(original_dtype)
        restored.append(full_tokens[:, prefix.shape[1]:].to(torch.bfloat16))
    return restored[0], restored[1]


def emit_metrics(rows: list[dict], tokens: dict, sample: dict, stage: str, branch: str,
                 reference: torch.Tensor, candidate: torch.Tensor, kind: str = "student") -> None:
    row, mse, cosine = feature_row(reference, candidate)
    key = (kind, sample["group"], sample["view"], stage, branch)
    tokens[key].append((mse, cosine))
    rows.append({"kind": kind, "group": sample["group"], "view": sample["view"],
                 "stage": stage, "branch": branch, "rgb_sha256": sample["rgb_sha256"],
                 **row})


def summarize(rows: list[dict], tokens: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], row["group"], row["view"], row["stage"], row["branch"])].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        matrix = tokens[key]
        mean_mse = np.mean([pair[0] for pair in matrix], axis=0)
        mean_cos = np.mean([pair[1] for pair in matrix], axis=0)
        frequent = Counter(index for mse, _ in matrix for index in np.argsort(mse)[-5:])
        top = np.argsort(mean_mse)[-10:][::-1]
        result.append({
            "kind": key[0], "group": key[1], "view": key[2], "stage": key[3], "branch": key[4],
            "n": len(items),
            **{name: float(np.mean([x[name] for x in items]))
               for name in ("mse", "cosine", "relative_l2", "max_abs", "p95_token_mse")},
            "mse_median": float(np.median([x["mse"] for x in items])),
            "mse_p90": float(np.quantile([x["mse"] for x in items], 0.9)),
            "token_mse_mean": mean_mse.tolist(), "token_cosine_mean": mean_cos.tolist(),
            "highest_error_tokens": [{"index": int(i), "mean_mse": float(mean_mse[i]),
                                      "mean_cosine": float(mean_cos[i]),
                                      "top5_frequency": int(frequent[int(i)])} for i in top],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    training = yaml.safe_load(Path(config["train_config"]).read_text())
    evaluation = yaml.safe_load(Path(config["eval_config"]).read_text())
    assert training["loss"]["scope"] == "boundary"
    assert evaluation["model"]["hpcm_raw_input_bypass"]
    assert Path(config["adapter_checkpoint"]).resolve() == Path(evaluation["model"]["adapter"]["checkpoint"]).resolve()
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    samples, coverage = select_samples(config, training)
    for sample in samples:
        rgb = rgb_from(Path(sample["path"]))
        sample["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        if "expected_rgb_sha256" in sample:
            assert sample["rgb_sha256"] == sample["expected_rgb_sha256"]
    assert len({(s["path"], s["view"]) for s in samples}) == len(samples)
    sim_map = {(s["task_id"], s["reset_index"], s["view"]): s for s in samples
               if s["group"] == "simulator_fresh_reset"}
    action_pairs = sorted({(task, reset) for task, reset, _ in sim_map})[:int(config["action_pairs"])]
    cfg = policy_cfg(evaluation)
    intact = get_vla(cfg)
    processor = get_processor(cfg)
    image_processor = processor.image_processor
    # The teacher pass evaluates one view at a time; policy action checks
    # below restore the checkpoint's two-view setting.
    intact.vision_backbone.set_num_images_in_input(1)
    action_head = get_action_head(cfg, intact.llm_dim)
    proprio_projector = get_proprio_projector(cfg, intact.llm_dim, proprio_dim=8)
    teacher = {}
    baseline_actions = {}
    rows, tokens = [], defaultdict(list)
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            rgb = rgb_from(Path(sample["path"]))
            pix = pixels(rgb, image_processor, cfg)
            backbone = intact.vision_backbone
            dino, dino_prefix = backbone.featurizer.get_intermediate_layers(
                pix[:, :3], n={2}, return_prefix_tokens=True)[0]
            siglip, siglip_prefix = backbone.fused_featurizer.get_intermediate_layers(
                pix[:, 3:], n={2}, return_prefix_tokens=True)[0]
            final = backbone(pix)
            projected = intact.projector(final)
            teacher[index] = {"dino": dino.cpu(), "siglip": siglip.cpu(),
                              "dino_prefix": dino_prefix.cpu(), "siglip_prefix": siglip_prefix.cpu(),
                              "final": final.cpu(), "projector": projected.cpu()}
            if (index + 1) % 24 == 0:
                print(f"Teacher {index + 1}/{len(samples)}", flush=True)
        intact.vision_backbone.set_num_images_in_input(2)
        for task, reset in action_pairs:
            agent = rgb_from(Path(sim_map[(task, reset, "agent")]["path"]))
            wrist = rgb_from(Path(sim_map[(task, reset, "wrist")]["path"]))
            description = next(item["task_description"] for item in json.loads(Path(config["sim_manifest"]).read_text())["samples"]
                               if item["task_id"] == task and item["reset_index"] == reset)
            baseline_actions[(task, reset)] = action(cfg, intact, processor, action_head, proprio_projector,
                                                     agent, wrist, description, False)
    del intact
    gc.collect()
    torch.cuda.empty_cache()

    model = evaluation["model"]
    adapter_config = dict(model["adapter"])
    adapter_config.pop("checkpoint", None)
    spliced = get_custom_vla_model(
        openvla_path=model["vla_path"], vision_backbone_max_layer=model["vision_backbone"]["max_layer"],
        hpcm_root=model["hpcm"]["hpcm_root"], hpcm_checkpoint=model["hpcm"]["checkpoint"],
        adapter_weights_path=config["adapter_checkpoint"], adapter_config=adapter_config,
        load_in_8bit=False, load_in_4bit=False, use_film=False,
        num_images_in_input=2, device="cuda",
    )
    combined = spliced.vision_backbone.model
    oracle_per_group = int(config["oracle_images_per_group"])
    oracle_counts = Counter()
    parity = []
    action_rows = []
    original_adapter_forward = combined.adapter.forward
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            rgb = rgb_from(Path(sample["path"]))
            raw = raw_tensor(rgb)
            result = combined(image=raw)
            ref = teacher[index]
            aligned_dino = result["aligned_dino_features"][0, 0]
            aligned_siglip = result["aligned_siglip_features"][0, 0]
            final = result["fused_features"][0, 0]
            projected = spliced.projector(final.unsqueeze(0))[0]
            for stage, branch, reference, candidate in (
                ("block2", "dino", ref["dino"][0], aligned_dino),
                ("block2", "siglip", ref["siglip"][0], aligned_siglip),
                ("final", "dino", ref["final"][0, :, :1024], final[:, :1024]),
                ("final", "siglip", ref["final"][0, :, 1024:], final[:, 1024:]),
                ("final", "fused", ref["final"][0], final),
                ("projector", "fused", ref["projector"][0], projected),
            ):
                emit_metrics(rows, tokens, sample, stage, branch, reference, candidate)
            if index == 0:
                pix = pixels(rgb, image_processor, cfg)
                spliced.vision_backbone.set_raw_images([rgb])
                staged = spliced.vision_backbone._pending_raw_images.to("cuda")
                parity.append({"test": "staged_raw_vs_direct_raw", "max_abs": float((staged - raw).abs().max())})
                wrapped = spliced.vision_backbone(pix)
                parity.append({"test": "wrapper_vs_combined_same_rgb", "max_abs": float((wrapped.float() - final.unsqueeze(0).float()).abs().max())})
                repeat = combined(image=raw)["fused_features"][0, 0]
                parity.append({"test": "combined_repeat_same_rgb", "max_abs": float((repeat.float() - final.float()).abs().max())})
                parity.append({"test": "fused_dino_first", "equal": bool(torch.equal(final[:, :1024], result["vision_dino_features"][0, 0]))})
                parity.append({"test": "fused_siglip_second", "equal": bool(torch.equal(final[:, 1024:], result["vision_siglip_features"][0, 0]))})
            group_view = (sample["group"], sample["view"])
            if oracle_counts[group_view] < oracle_per_group:
                oracle_counts[group_view] += 1
                target_dino = ref["dino"].to("cuda", dtype=torch.float16)
                target_siglip = ref["siglip"].to("cuda", dtype=torch.float16)
                def oracle_forward(_self, _latent):
                    return target_siglip, target_dino
                combined.adapter.forward = MethodType(oracle_forward, combined.adapter)
                try:
                    injected = combined(image=raw)
                finally:
                    combined.adapter.forward = original_adapter_forward
                oracle_final = injected["fused_features"][0, 0]
                oracle_projected = spliced.projector(oracle_final.unsqueeze(0))[0]
                emit_metrics(rows, tokens, sample, "final", "dino", ref["final"][0, :, :1024],
                             oracle_final[:, :1024], "oracle")
                emit_metrics(rows, tokens, sample, "final", "siglip", ref["final"][0, :, 1024:],
                             oracle_final[:, 1024:], "oracle")
                emit_metrics(rows, tokens, sample, "final", "fused", ref["final"][0], oracle_final, "oracle")
                emit_metrics(rows, tokens, sample, "projector", "fused", ref["projector"][0], oracle_projected, "oracle")
                # A diagnostic upper bound: restore the intact tower's hidden
                # prefix/register tokens, which the 256-token Adapter omits.
                restored = posterior_with_prefix(ref, combined)
                for branch, patches in zip(("dino", "siglip"), restored):
                    emit_metrics(rows, tokens, sample, "final", branch,
                                 ref["final"][0, :, :1024] if branch == "dino" else ref["final"][0, :, 1024:],
                                 patches[0], "oracle_with_prefix")
                restored_final = torch.cat(restored, dim=-1)[0]
                emit_metrics(rows, tokens, sample, "final", "fused", ref["final"][0],
                             restored_final, "oracle_with_prefix")
                emit_metrics(rows, tokens, sample, "projector", "fused", ref["projector"][0],
                             spliced.projector(restored_final.unsqueeze(0))[0], "oracle_with_prefix")
            if (index + 1) % 24 == 0:
                print(f"Student {index + 1}/{len(samples)}", flush=True)

        # The action test uses only independent simulator fresh-reset RGB and a
        # fixed synthetic zero proprio vector, identical for all three calls.
        for task, reset in action_pairs:
            agent_sample = sim_map[(task, reset, "agent")]
            wrist_sample = sim_map[(task, reset, "wrist")]
            agent = rgb_from(Path(agent_sample["path"]))
            wrist = rgb_from(Path(wrist_sample["path"]))
            description = next(item["task_description"] for item in json.loads(Path(config["sim_manifest"]).read_text())["samples"]
                               if item["task_id"] == task and item["reset_index"] == reset)
            student_action = action(cfg, spliced, processor, action_head, proprio_projector,
                                    agent, wrist, description, True)
            teacher_dino = torch.cat([teacher[samples.index(agent_sample)]["dino"],
                                      teacher[samples.index(wrist_sample)]["dino"]]).to("cuda", dtype=torch.float16)
            teacher_siglip = torch.cat([teacher[samples.index(agent_sample)]["siglip"],
                                        teacher[samples.index(wrist_sample)]["siglip"]]).to("cuda", dtype=torch.float16)
            def oracle_forward(_self, _latent):
                return teacher_siglip, teacher_dino
            combined.adapter.forward = MethodType(oracle_forward, combined.adapter)
            try:
                oracle_action = action(cfg, spliced, processor, action_head, proprio_projector,
                                       agent, wrist, description, True)
            finally:
                combined.adapter.forward = original_adapter_forward
            restored_views = []
            for sample in (agent_sample, wrist_sample):
                restored_dino, restored_siglip = posterior_with_prefix(teacher[samples.index(sample)], combined)
                restored_views.append(torch.cat([restored_dino, restored_siglip], dim=-1))
            restored_features = torch.cat(restored_views, dim=1)
            wrapper = spliced.vision_backbone
            original_wrapper_forward = wrapper.forward
            def fixed_feature_forward(_self, _pixel_values, **_kwargs):
                return restored_features
            wrapper.forward = MethodType(fixed_feature_forward, wrapper)
            try:
                corrected_action = action(cfg, spliced, processor, action_head, proprio_projector,
                                          agent, wrist, description, False)
            finally:
                wrapper.forward = original_wrapper_forward
            baseline = baseline_actions[(task, reset)]
            assert baseline.shape == oracle_action.shape == student_action.shape == corrected_action.shape
            action_rows.append({"task_id": task, "reset_index": reset,
                                "state_sha256": agent_sample["state_sha256"],
                                "baseline_vs_oracle_mae": float(np.abs(baseline - oracle_action).mean()),
                                "baseline_vs_oracle_max_abs": float(np.abs(baseline - oracle_action).max()),
                                "baseline_vs_oracle_with_prefix_mae": float(np.abs(baseline - corrected_action).mean()),
                                "baseline_vs_oracle_with_prefix_max_abs": float(np.abs(baseline - corrected_action).max()),
                                "baseline_vs_student_mae": float(np.abs(baseline - student_action).mean()),
                                "baseline_vs_student_max_abs": float(np.abs(baseline - student_action).max()),
                                "action_shape": list(baseline.shape)})
            print(f"Action parity task {task} reset {reset}: {action_rows[-1]}", flush=True)

    output = {"schema_version": 1, "seed": int(config["seed"]),
              "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "protocol": {"training_cap_reconstruction": "sorted training paths, random.Random(42).shuffle, first 16384 indices",
                           "teacher": "Object-Pro native JPEG/Lanczos3/crop/processor pixels; BF16 intact tower",
                           "student": "raw decoded 256 RGB to frozen HPCM and FP16 Adapter; actual spliced FP32 posterior and BF16 output",
                           "oracle": "replace Adapter block-2 output in memory with intact teacher block-2 patch tokens; actual spliced posterior/wrapper",
                           "oracle_with_prefix": "offline-only restoration of intact block-2 prefix/register tokens before the same posterior; not a deployable Adapter path",
                           "action": "three independent simulator fresh-reset pairs, fixed zero proprio; no environment step or formal state"},
              "checkpoints": {"hpcm": {"path": model["hpcm"]["checkpoint"], "sha256": digest(Path(model["hpcm"]["checkpoint"]))},
                              "adapter": {"path": config["adapter_checkpoint"], "sha256": digest(Path(config["adapter_checkpoint"]))},
                              "object_pro": {"path": model["pretrained_checkpoint"], "sha256": digest(Path(model["pretrained_checkpoint"]) / "model.safetensors")},
                              "action_head_sha256": digest(Path(model["pretrained_checkpoint"]) / "action_head--checkpoint.pt"),
                              "proprio_projector_sha256": digest(Path(model["pretrained_checkpoint"]) / "proprio_projector--checkpoint.pt")},
              "manifests": {"split_sha256": digest(Path(config["split_manifest"])),
                            "sim_sha256": digest(Path(config["sim_manifest"]))},
              "coverage": coverage, "samples": samples, "parity": parity,
              "action_parity": action_rows, "metrics": summarize(rows, tokens), "sample_metrics": rows}
    output_dir.mkdir(parents=True)
    (output_dir / "config.yaml").write_text(args.config.read_text())
    (output_dir / "diagnostic.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {output_dir / 'diagnostic.json'}", flush=True)


if __name__ == "__main__":
    main()
