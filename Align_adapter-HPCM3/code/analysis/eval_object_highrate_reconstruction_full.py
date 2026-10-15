"""Frozen Object-Pro + real HPCM bitstream RGB reconstruction, 10x50.

The existing high-rate HPCM3 rollout is the paired protocol reference. This
script never changes model weights or trains, and refuses to overwrite output.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for root in (REPO, CODE, REPO / "LIBERO"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analysis.eval_object_highrate_boundary_task0 import config_from_yaml  # noqa: E402
from analysis.object_hpcm_reconstruction_control import ReconstructionControl, sha256  # noqa: E402
from analysis.eval_spatial_bitrate_full import append, encode_reconstruct  # noqa: E402
from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from inference import run_libero_eval as evaluator  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def load_reference_first_query(path: Path) -> dict[tuple[int, int, str], dict]:
    first = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["policy_query_index"] == 0:
            key = (row["task_id"], row["episode_index"], row["view"])
            if key in first:
                raise ValueError(f"Duplicate reference first query: {key}")
            first[key] = row
    if len(first) != 1000:
        raise ValueError("Expected two first-query views for 500 Object episodes")
    return first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    raw = yaml.safe_load(args.config.read_text())
    cfg = config_from_yaml(raw)
    reference_path = Path(raw["experiment"]["hpcm3_reference_summary"])
    reference = json.loads(reference_path.read_text())
    hpcm = Path(cfg.model.hpcm.checkpoint)
    policy_dir = Path(cfg.model.pretrained_checkpoint)
    if not (cfg.libero.task_suite_name == "libero_object" and cfg.libero.num_trials_per_task == 50 and
            cfg.libero.initial_states_path == "DEFAULT" and cfg.libero.env_img_res == 256 and
            cfg.libero.num_steps_wait == 10 and cfg.model.num_open_loop_steps == 8 and
            cfg.model.unnorm_key == "libero_object_no_noops" and cfg.model.center_crop and
            cfg.model.num_images_in_input == 2 and cfg.seed == 7 and
            not cfg.model.use_hpcm_adapter and not cfg.model.hpcm_raw_input_bypass):
        raise ValueError("Wrong Object reconstruction 10x50 protocol")
    if reference["success_episodes"] != 472 or reference["total_episodes"] != 500 or \
            sha256(hpcm) != reference["checkpoints"]["hpcm"]["sha256"] or \
            sha256(policy_dir / "model.safetensors") != reference["checkpoints"]["object_pro"]["sha256"]:
        raise ValueError("Frozen Object-Pro/HPCM or reference changed")
    for name in ("action_head--checkpoint.pt", "proprio_projector--checkpoint.pt"):
        if sha256(policy_dir / name) != reference["checkpoints"][name.split("--")[0]]["sha256"]:
            raise ValueError(f"Frozen Object-Pro component changed: {name}")
    reference_episodes_path = Path(raw["experiment"]["hpcm3_reference_episodes"])
    reference_rates_path = Path(raw["experiment"]["hpcm3_reference_rates"])
    ref_episodes = [json.loads(line) for line in reference_episodes_path.read_text().splitlines() if line.strip()]
    if len(ref_episodes) != 500:
        raise ValueError("Incomplete HPCM3 reference episode results")
    ref_first = load_reference_first_query(reference_rates_path)
    data_audit_path = Path(raw["experiment"]["data_audit_summary"])
    data_audit = json.loads(data_audit_path.read_text())
    train_cache = Path(cfg.audit.train_hash_cache)
    if sha256(train_cache) != data_audit["train_hash_cache_sha256"] or \
            data_audit["train_val_exact_rgb_hash_overlap"] != 0 or \
            data_audit["train_protected_exact_rgb_hash_overlap"] != 0:
        raise ValueError("Training data audit mismatch")
    train_hashes = set(train_cache.read_text().splitlines())
    if len(train_hashes) != data_audit["train_unique_rgb_hashes"]:
        raise ValueError("Training RGB cache count changed")

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.yaml").write_text(args.config.read_text())
    provenance = {"git_commit": evaluator._git_metadata()["commit"], "seed": cfg.seed,
                  "suite": "libero_object", "arm": "reconstruction", "lambda": "0.0483",
                  "checkpoints": reference["checkpoints"],
                  "hpcm3_reference_summary_sha256": sha256(reference_path),
                  "hpcm3_reference_episodes_sha256": sha256(reference_episodes_path),
                  "hpcm3_reference_rates_sha256": sha256(reference_rates_path),
                  "data_audit_summary_sha256": sha256(data_audit_path),
                  "protocol": "Raw 256 RGB -> real y/z entropy bitstream -> decoded RGB -> Object-Pro native image path; DEFAULT states 0-49/task, seed 7, 10 dummy steps, 8-action queue"}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    set_seed_everywhere(cfg.seed)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    if suite.n_tasks != 10:
        raise ValueError("Expected ten Object tasks")
    first_env, first_description = evaluator.get_libero_env(suite.get_task(0), cfg.model.model_family, resolution=256)
    policy, action_head, proprio_projector, noisy_action_projector, processor = evaluator.initialize_model(cfg)
    codec = ReconstructionControl(hpcm, Path(cfg.model.hpcm.hpcm_root), cfg.model.hpcm.scale_table_levels)
    resize_size = get_image_resize_size(cfg)
    original_get_action = evaluator.get_action
    active = {"task_id": -1, "episode_index": -1, "rates": []}

    def reconstructed_get_action(config, model, observation, task_label, *action_args, **kwargs):
        query = len(active["rates"]) // 2
        decoded_views = []
        for view, rgb in zip(("agent", "wrist"), observation["_hpcm_raw_images"]):
            decoded, metrics = encode_reconstruct(codec, rgb,
                check_latent=active["episode_index"] == 0 and query == 0)
            row = {"suite": "libero_object", "task_id": active["task_id"],
                   "episode_index": active["episode_index"], "policy_query_index": query,
                   "view": view, **metrics}
            if query == 0:
                key = (active["task_id"], active["episode_index"], view)
                old = ref_first[key]
                if row["source_rgb_sha256"] != old["source_rgb_sha256"]:
                    raise RuntimeError(f"First-query RGB differs from HPCM3 at {key}")
                # Exact entropy bytes can differ slightly between intact-VLA
                # and spliced-VLA process contexts despite identical RGB and
                # frozen codec weights. Record rather than conceal this.
                row["first_query_hpcm3_y_bytes_delta"] = row["y_bytes"] - old["y_bytes"]
                row["first_query_hpcm3_z_bytes_delta"] = row["z_bytes"] - old["z_bytes"]
                row["first_query_hpcm3_byte_match"] = (
                    row["y_bytes"] == old["y_bytes"] and row["z_bytes"] == old["z_bytes"])
            active["rates"].append(row)
            append(args.output_dir / "rate.jsonl", row)
            decoded_views.append(resize_image_for_policy(decoded, resize_size))
        observation["full_image"], observation["wrist_image"] = decoded_views
        return original_get_action(config, model, observation, task_label, *action_args, **kwargs)

    evaluator.get_action = reconstructed_get_action
    episodes, tasks = [], []
    start = time.monotonic()
    run_log = (args.output_dir / "run.log").open("x", encoding="utf-8")
    audit_file = (args.output_dir / "audit.jsonl").open("x", encoding="utf-8")
    try:
        for task_id in range(10):
            env, description = (first_env, first_description) if task_id == 0 else evaluator.get_libero_env(
                suite.get_task(task_id), cfg.model.model_family, resolution=256)
            states = suite.get_task_init_states(task_id)
            if len(states) < 50:
                raise RuntimeError(f"Task {task_id} has fewer than 50 DEFAULT states")
            current = []
            try:
                for episode_index in range(50):
                    active.update(task_id=task_id, episode_index=episode_index, rates=[])
                    reference_episode = ref_episodes[task_id * 50 + episode_index]
                    state_sha = evaluator._sha256_array(states[episode_index])
                    if (reference_episode["task_id"], reference_episode["episode_index"],
                        reference_episode["initial_state_sha256"]) != (task_id, episode_index, state_sha):
                        raise ValueError("Official initial-state ordering differs from HPCM3 reference")
                    episode_log = io.StringIO()
                    success, _, matches = evaluator.run_episode(
                        cfg, env, description, policy, resize_size, processor,
                        action_head, proprio_projector, noisy_action_projector,
                        initial_state=states[episode_index], log_file=episode_log,
                        audit_file=audit_file,
                        audit_context={"suite": "libero_object", "task_id": task_id,
                                       "episode_index": episode_index, "initial_state_source": "DEFAULT"},
                        training_frame_hashes=train_hashes)
                    run_log.write(f"TASK {task_id} EPISODE {episode_index}\n" + episode_log.getvalue() + "\n")
                    run_log.flush()
                    if "Episode error:" in episode_log.getvalue() or not active["rates"]:
                        raise RuntimeError(f"Evaluation error at task {task_id} episode {episode_index}")
                    rates = active["rates"]
                    row = {"suite": "libero_object", "arm": "reconstruction", "lambda": "0.0483",
                           "task_id": task_id, "task_description": description,
                           "episode_index": episode_index, "initial_state_source": "DEFAULT",
                           "initial_state_sha256": state_sha, "success": bool(success),
                           "failure_type": None if success else "horizon_without_success",
                           "exact_train_frame_matches": matches, "policy_queries": len(rates) // 2,
                           "bpp_actual_mean": {view: float(np.mean([r["bpp_actual"] for r in rates if r["view"] == view]))
                                               for view in ("agent", "wrist")},
                           "psnr_db_mean": {view: float(np.mean([r["psnr_db"] for r in rates if r["view"] == view]))
                                            for view in ("agent", "wrist")}}
                    append(args.output_dir / "episode_results.jsonl", row)
                    episodes.append(row)
                    current.append(row)
                    print(f"Object reconstruction task {task_id}/9 episode {episode_index}/49: success={success}", flush=True)
            finally:
                env.close()
            task_result = {"task_id": task_id, "task_description": description, "episodes": 50,
                           "success_episodes": sum(r["success"] for r in current),
                           "success_rate": sum(r["success"] for r in current) / 50,
                           "exact_train_frame_matches": sum(r["exact_train_frame_matches"] for r in current)}
            tasks.append(task_result)
            append(args.output_dir / "task_results.jsonl", task_result)
    finally:
        evaluator.get_action = original_get_action
        audit_file.close()
        run_log.close()
    if len(episodes) != 500 or len(tasks) != 10:
        raise RuntimeError("Incomplete Object 10x50 reconstruction result")
    rates = [json.loads(line) for line in (args.output_dir / "rate.jsonl").read_text().splitlines() if line.strip()]
    summary = {**provenance, "task_count": 10, "total_episodes": 500,
               "success_episodes": sum(r["success"] for r in episodes),
               "success_rate": sum(r["success"] for r in episodes) / 500,
               "exact_train_frame_matches": sum(r["exact_train_frame_matches"] for r in episodes),
               "compressed_views": len(rates),
               "mean_actual_bpp": {view: float(np.mean([r["bpp_actual"] for r in rates if r["view"] == view]))
                                   for view in ("agent", "wrist")},
               "mean_psnr_db": {view: float(np.mean([r["psnr_db"] for r in rates if r["view"] == view]))
                                for view in ("agent", "wrist")},
               "first_query_rgb_matches_hpcm3": sum(r.get("first_query_hpcm3_byte_match") is not None for r in rates),
               "first_query_bitstream_byte_matches_hpcm3": sum(r.get("first_query_hpcm3_byte_match", False) for r in rates),
               "first_query_y_byte_delta_range": [min(r.get("first_query_hpcm3_y_bytes_delta", 0) for r in rates),
                                                   max(r.get("first_query_hpcm3_y_bytes_delta", 0) for r in rates)],
               "first_query_z_byte_delta_range": [min(r.get("first_query_hpcm3_z_bytes_delta", 0) for r in rates),
                                                   max(r.get("first_query_hpcm3_z_bytes_delta", 0) for r in rates)],
               "entropy_latent_checked_views": sum("entropy_latent_max_abs" in r for r in rates),
               "entropy_latent_max_abs": max(r.get("entropy_latent_max_abs", 0.0) for r in rates),
               "per_task": tasks, "elapsed_seconds": time.monotonic() - start,
               "demo_initial_state_overlap": "unresolved: RLDS PNG export lacks full demo simulator states"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"success_episodes": summary["success_episodes"],
                      "mean_actual_bpp": summary["mean_actual_bpp"],
                      "mean_psnr_db": summary["mean_psnr_db"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
