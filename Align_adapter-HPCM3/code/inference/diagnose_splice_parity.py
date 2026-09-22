"""Compare an intact VLA with the HPCM3-spliced VLA on one LIBERO state.

This diagnostic intentionally uses the same processor/preprocessing path as the
successful historical Spatial evaluation.  It reports vision-feature error and
the first predicted action chunk without advancing the environment.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
INFERENCE_ROOT = Path(__file__).resolve().parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
    prepare_images_for_vla,
)
from libero.libero import benchmark
from model.model import get_custom_vla_model
from run_libero_eval import prepare_observation, set_seed_everywhere


def _namespace(mapping: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**mapping)


def _feature_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    reference = reference.float().reshape(-1, reference.shape[-1])
    candidate = candidate.float().reshape(-1, candidate.shape[-1])
    delta = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=-1)
    ref_norm = torch.linalg.vector_norm(reference, dim=-1).mean()
    err_norm = torch.linalg.vector_norm(delta, dim=-1).mean()
    return {
        "mse": float(delta.square().mean()),
        "mae": float(delta.abs().mean()),
        "mean_token_cosine": float(cosine.mean()),
        "relative_l2": float(err_norm / ref_norm.clamp_min(1e-12)),
    }


def _processor_pixels(processor, cfg, observation, task_description):
    images = [observation["full_image"], observation["wrist_image"]]
    images = prepare_images_for_vla(images, cfg)
    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n<|im_start|>user\nWhat action should the robot take to "
        f"{task_description.lower()}?<|im_end|>\n<|im_start|>assistant\n"
    )
    encoded = [processor(prompt, image).to("cuda", dtype=torch.bfloat16) for image in images]
    return torch.cat([item["pixel_values"] for item in encoded], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    model_raw = dict(raw["model"])
    model_raw["hpcm"] = _namespace(model_raw["hpcm"])
    model_raw["adapter"] = _namespace(model_raw["adapter"])
    model_raw["vision_backbone"] = _namespace(model_raw["vision_backbone"])
    model_raw.setdefault("hpcm_raw_input_bypass", False)
    cfg = _namespace(model_raw)
    cfg.preserve_checkpoint_files = True
    cfg.pretrained_checkpoint = str(Path(cfg.pretrained_checkpoint).resolve())
    cfg.vla_path = str(Path(cfg.vla_path).resolve())
    cfg.hpcm_raw_input_bypass = False

    set_seed_everywhere(raw.get("seed", 7))
    suite = benchmark.get_benchmark_dict()[raw["libero"]["task_suite_name"]]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    env, task_description = get_libero_env(
        task, cfg.model_family, resolution=raw["libero"].get("env_img_res", 256)
    )
    observation, _, _ = prepare_observation(
        env.set_init_state(states[args.episode_index]), 224
    )
    env.close()

    processor = get_processor(cfg)
    intact = get_vla(cfg)
    action_head = get_action_head(cfg, intact.llm_dim)
    proprio_projector = get_proprio_projector(cfg, intact.llm_dim, proprio_dim=8)
    pixel_values = _processor_pixels(processor, cfg, observation, task_description)
    with torch.inference_mode():
        intact_features = intact.vision_backbone(pixel_values).detach().cpu()
        intact_actions = np.asarray(
            get_vla_action(
                cfg, intact, processor, dict(observation), task_description,
                action_head=action_head, proprio_projector=proprio_projector,
                use_film=cfg.use_film, use_minivlm=cfg.use_minivlm,
            )
        )

    del intact
    gc.collect()
    torch.cuda.empty_cache()

    adapter_cfg = vars(cfg.adapter).copy()
    adapter_cfg.pop("checkpoint", None)
    spliced = get_custom_vla_model(
        openvla_path=cfg.vla_path,
        vision_backbone_max_layer=cfg.vision_backbone.max_layer,
        hpcm_root=cfg.hpcm.hpcm_root,
        hpcm_checkpoint=cfg.hpcm.checkpoint,
        adapter_weights_path=cfg.adapter.checkpoint,
        adapter_config=adapter_cfg,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        use_film=cfg.use_film,
        num_images_in_input=cfg.num_images_in_input,
        device="cuda",
    )
    with torch.inference_mode():
        spliced_features = spliced.vision_backbone(pixel_values).detach().cpu()
        spliced_actions = np.asarray(
            get_vla_action(
                cfg, spliced, processor, dict(observation), task_description,
                action_head=action_head, proprio_projector=proprio_projector,
                use_film=cfg.use_film, use_minivlm=cfg.use_minivlm,
            )
        )

    result = {
        "suite": raw["libero"]["task_suite_name"],
        "task_id": args.task_id,
        "episode_index": args.episode_index,
        "task_description": task_description,
        "preprocessing": "historical_spatial_processor_path",
        "feature_shape": list(intact_features.shape),
        "features": {
            "fused": _feature_metrics(intact_features, spliced_features),
            "dino": _feature_metrics(intact_features[..., :1024], spliced_features[..., :1024]),
            "siglip": _feature_metrics(intact_features[..., 1024:], spliced_features[..., 1024:]),
        },
        "actions": {
            "intact": intact_actions.tolist(),
            "spliced": spliced_actions.tolist(),
            "mae": float(np.mean(np.abs(intact_actions - spliced_actions))),
            "max_abs": float(np.max(np.abs(intact_actions - spliced_actions))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
