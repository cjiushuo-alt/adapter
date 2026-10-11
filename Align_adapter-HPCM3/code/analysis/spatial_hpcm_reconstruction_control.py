"""Paired Spatial-Pro A/B control using Object's real HPCM bitstream codec.

A keeps intact RGB; B replaces both policy-query views with HPCM decoded RGB.
Historical Spatial HPCM3 C is read-only and checked against the same states.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from object_hpcm_reconstruction_control import (  # noqa: E402
    ReconstructionControl, append_jsonl, config_from_yaml, sha256, sha256_array,
)
from inference import run_libero_eval as evaluator  # noqa: E402
from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    cfg = config_from_yaml(raw)
    spec = raw["experiment"]
    count = int(spec["episodes"])
    assert count == 5 and cfg.seed == 7
    assert cfg.libero.task_suite_name == "libero_spatial" and cfg.libero.initial_states_path == "DEFAULT"
    assert cfg.libero.env_img_res == 256 and cfg.libero.num_steps_wait == 10
    assert cfg.model.num_images_in_input == 2 and cfg.model.num_open_loop_steps == 8 and cfg.model.center_crop
    assert not cfg.model.use_hpcm_adapter and not cfg.model.hpcm_raw_input_bypass

    c_path = Path(spec["adapter_results"])
    c_audit_path = Path(spec["adapter_audit"])
    c_rows = read_jsonl(c_path)
    assert len(c_rows) >= count
    c_initial_rgb = {}
    for row in read_jsonl(c_audit_path):
        if row["task_id"] == 0 and row["episode_index"] < count and row["rollout_step"] == 0:
            c_initial_rgb[row["episode_index"]] = (row["main_rgb_sha256"], row["wrist_rgb_sha256"])
    assert len(c_initial_rgb) == count

    suite = benchmark.get_benchmark_dict()[cfg.libero.task_suite_name]()
    states = suite.get_task_init_states(0)
    for index in range(count):
        assert c_rows[index]["task_id"] == 0 and c_rows[index]["episode_index"] == index
        assert evaluator._sha256_array(states[index]) == c_rows[index]["initial_state_sha256"]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("config.yaml", "provenance.json", "historical_c_initial_frames.json", "episode_results.jsonl", "frame_metrics.jsonl", "summary.json", "run.log"):
        assert not (output / filename).exists(), f"Refusing to overwrite {output / filename}"
    (output / "config.yaml").write_text(args.config.read_text())
    hpcm_path = Path(cfg.model.hpcm.checkpoint)
    adapter_path = Path(cfg.model.adapter.checkpoint)
    vla_dir = Path(cfg.model.pretrained_checkpoint)
    provenance = {
        "git_commit": evaluator._git_metadata()["commit"], "seed": cfg.seed,
        "suite": cfg.libero.task_suite_name, "task_id": 0,
        "checkpoints": {
            "hpcm": {"path": str(hpcm_path), "sha256": sha256(hpcm_path)},
            "adapter_C_reference": {"path": str(adapter_path), "sha256": sha256(adapter_path)},
            "vla_model": {"path": str(vla_dir / "model.safetensors"), "sha256": sha256(vla_dir / "model.safetensors")},
            "action_head": {"path": str(vla_dir / "action_head--checkpoint.pt"), "sha256": sha256(vla_dir / "action_head--checkpoint.pt")},
            "proprio_projector": {"path": str(vla_dir / "proprio_projector--checkpoint.pt"), "sha256": sha256(vla_dir / "proprio_projector--checkpoint.pt")},
        },
        "historical_C": {
            "episode_results_path": str(c_path), "episode_results_sha256": sha256(c_path),
            "audit_path": str(c_audit_path), "audit_sha256": sha256(c_audit_path),
        },
        "protocol": "A: intact Spatial-Pro; B: raw 256px agent+wrist HPCM compress/decompress y+z bitstreams at each policy query, then native JPEG/Lanczos3 224px+crop Spatial-Pro processor; same 10 dummy steps and 8-action queue as Object control and historical C",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    c_snapshot = [
        {"episode_index": index, "initial_state_sha256": c_rows[index]["initial_state_sha256"],
         "main_rgb_sha256": c_initial_rgb[index][0], "wrist_rgb_sha256": c_initial_rgb[index][1],
         "success": bool(c_rows[index]["success"])}
        for index in range(count)
    ]
    (output / "historical_c_initial_frames.json").write_text(json.dumps(c_snapshot, ensure_ascii=False, indent=2) + "\n")

    set_seed_everywhere(cfg.seed)
    policy, action_head, proprio_projector, noisy_action_projector, processor = evaluator.initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    env, description = evaluator.get_libero_env(suite.get_task(0), cfg.model.model_family, resolution=256)
    original_prepare = evaluator.prepare_observation
    original_get_action = evaluator.get_action
    frame_rows: list[dict] = []
    episode_rows: list[dict] = []
    first_hashes: dict[tuple[str, int], tuple[str, str]] = {}
    current_arm = "A"
    current_episode = -1
    current_frames: list[dict] = []
    codec = None

    def prepare_with_raw(obs, resize):
        observation, agent, wrist = original_prepare(obs, resize)
        if (current_arm, current_episode) not in first_hashes:
            first_hashes[(current_arm, current_episode)] = (sha256_array(agent), sha256_array(wrist))
        if current_arm == "B":
            observation["_original_rgb_for_compression"] = (agent, wrist)
        return observation, agent, wrist

    def get_action_ab(config, model, observation, task_label, *action_args, **kwargs):
        if current_arm == "B":
            assert codec is not None
            source = observation.pop("_original_rgb_for_compression")
            reconstructed = []
            for view, rgb in zip(("agent", "wrist"), source):
                rgb_hat, stats = codec.reconstruct(rgb, check_forward=(current_episode == 0 and len(current_frames) < 2))
                stats.update({"arm": "B", "episode_index": current_episode, "view": view,
                              "policy_query_index": len(current_frames) // 2})
                current_frames.append(stats)
                frame_rows.append(stats)
                append_jsonl(output / "frame_metrics.jsonl", stats)
                reconstructed.append(resize_image_for_policy(rgb_hat, resize_size))
            observation["full_image"], observation["wrist_image"] = reconstructed
        return original_get_action(config, model, observation, task_label, *action_args, **kwargs)

    evaluator.prepare_observation = prepare_with_raw
    evaluator.get_action = get_action_ab
    try:
        with (output / "run.log").open("x", encoding="utf-8") as run_log:
            for arm in ("A", "B"):
                if arm == "B":
                    codec = ReconstructionControl(hpcm_path, Path(cfg.model.hpcm.hpcm_root), cfg.model.hpcm.scale_table_levels)
                # Reset at the start of each arm, after policy/codec loading,
                # so A and B follow the same seeded simulator episode stream.
                set_seed_everywhere(cfg.seed)
                current_arm = arm
                for index in range(count):
                    current_episode = index
                    current_frames = []
                    print(f"Starting Spatial {arm} episode {index}/4", flush=True)
                    episode_log = io.StringIO()
                    success, _, _ = evaluator.run_episode(
                        cfg, env, description, policy, resize_size, processor,
                        action_head, proprio_projector, noisy_action_projector,
                        initial_state=states[index], log_file=episode_log,
                    )
                    run_log.write(episode_log.getvalue())
                    run_log.write(f"arm={arm} episode={index} success={bool(success)} policy_queries={len(current_frames)//2 if arm == 'B' else 'not_counted'}\n")
                    run_log.flush()
                    assert "Episode error:" not in episode_log.getvalue(), f"{arm} episode {index} failed with exception"
                    if arm == "B":
                        assert current_frames and len(current_frames) % 2 == 0
                    record = {
                        "suite": cfg.libero.task_suite_name, "task_id": 0, "arm": arm,
                        "episode_index": index, "initial_state_sha256": evaluator._sha256_array(states[index]),
                        "first_agent_rgb_sha256": first_hashes[(arm, index)][0],
                        "first_wrist_rgb_sha256": first_hashes[(arm, index)][1],
                        "first_rgb_matches_historical_C": first_hashes[(arm, index)] == c_initial_rgb[index],
                        "success": bool(success), "C_historical_success": bool(c_rows[index]["success"]),
                    }
                    if arm == "B":
                        record.update({
                            "policy_queries": len(current_frames) // 2, "compressed_frames": len(current_frames),
                            **{f"{view}_{metric}": float(np.mean([r[metric] for r in current_frames if r["view"] == view]))
                               for view in ("agent", "wrist") for metric in ("psnr_db", "bpp_actual")},
                        })
                    episode_rows.append(record)
                    append_jsonl(output / "episode_results.jsonl", record)
                    print(f"Spatial {arm} episode {index}: success={bool(success)}", flush=True)
    finally:
        evaluator.prepare_observation = original_prepare
        evaluator.get_action = original_get_action
        env.close()

    summary = {
        **provenance, "task_description": description, "episodes_per_arm": count,
        "successes": {arm: sum(r["success"] for r in episode_rows if r["arm"] == arm)
                      if arm != "C" else sum(bool(r["success"]) for r in c_rows[:count])
                      for arm in ("A", "B", "C")},
        "first_rgb_hash_matches_historical_C": {arm: sum(first_hashes[(arm, i)] == c_initial_rgb[i] for i in range(count))
                                                 for arm in ("A", "B")},
        "first_rgb_hash_matches_A_vs_B": sum(first_hashes[("A", i)] == first_hashes[("B", i)] for i in range(count)),
        "compressed_frames": len(frame_rows),
        "mean_psnr_db": {view: float(np.mean([r["psnr_db"] for r in frame_rows if r["view"] == view])) for view in ("agent", "wrist")},
        "mean_actual_bpp": {view: float(np.mean([r["bpp_actual"] for r in frame_rows if r["view"] == view])) for view in ("agent", "wrist")},
        "forward_vs_decode_max_abs": max(float(r.get("forward_vs_decoded_max_abs", 0)) for r in frame_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("successes", "first_rgb_hash_matches_A_vs_B", "first_rgb_hash_matches_historical_C", "mean_psnr_db", "mean_actual_bpp")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
