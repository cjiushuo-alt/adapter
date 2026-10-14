"""Frozen high-rate Object HPCM3: ten official tasks, 50 states each."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for location in (REPO, CODE, REPO / "LIBERO"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from libero.libero import benchmark  # noqa: E402
from inference import run_libero_eval as evaluator  # noqa: E402
from analysis.eval_object_highrate_boundary_task0 import config_from_yaml  # noqa: E402
from analysis.object_hpcm_reconstruction_control import ReconstructionControl, sha256  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.inference_mode()
def encode_query_image(codec: ReconstructionControl, rgb: np.ndarray, check_latent: bool) -> dict:
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Expected raw 256 RGB, got {rgb.shape}/{rgb.dtype}")
    rgb = np.ascontiguousarray(rgb)
    # Match raw bypass: CPU uint8 -> FP32 [-1,1], then GPU -> HPCM [0,1].
    raw = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(127.5).sub_(1.0).cuda()
    image01 = raw.add(1.0).mul(0.5)
    bitstream = codec.codec.compress(image01)
    y_bytes, z_bytes = map(len, bitstream["strings"])
    record = {"source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "y_bytes": y_bytes, "z_bytes": z_bytes,
              "bpp_actual": 8.0 * (y_bytes + z_bytes) / (256 * 256)}
    if check_latent:
        forward = codec.encoder(image01)
        decoded_latent = []
        hook = codec.codec.g_s.register_forward_hook(
            lambda _module, inputs, _output: decoded_latent.append(inputs[0].detach().clone()))
        try:
            codec.codec.decompress(bitstream["strings"], bitstream["shape"])
        finally:
            hook.remove()
        if len(decoded_latent) != 1 or decoded_latent[0].shape != forward.shape:
            raise RuntimeError("Could not capture entropy-decoded latent")
        diff = (decoded_latent[0] - forward).float().abs()
        record["entropy_latent_max_abs"] = float(diff.max())
        record["entropy_latent_nonzero_elements"] = int(torch.count_nonzero(diff))
        if record["entropy_latent_max_abs"] != 0:
            raise RuntimeError("Forward latent differs from entropy-decoded latent")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    cfg = config_from_yaml(raw)
    exp = raw["experiment"]
    if not (cfg.model.use_hpcm_adapter and cfg.model.hpcm_raw_input_bypass and cfg.model.center_crop and
            cfg.model.num_open_loop_steps == 8 and cfg.model.unnorm_key == "libero_object_no_noops" and
            cfg.libero.task_suite_name == "libero_object" and cfg.libero.num_trials_per_task == 50 and
            cfg.libero.initial_states_path == "DEFAULT" and cfg.libero.num_steps_wait == 10 and
            cfg.libero.env_img_res == 256 and cfg.seed == 7):
        raise ValueError("Wrong 10x50 Object evaluation protocol")
    adapter = Path(cfg.model.adapter.checkpoint)
    hpcm = Path(cfg.model.hpcm.checkpoint)
    object_pro = Path(cfg.model.pretrained_checkpoint)
    if sha256(adapter) != exp["adapter_sha256"] or sha256(hpcm) != exp["hpcm_sha256"] or \
       sha256(object_pro / "model.safetensors") != exp["object_pro_sha256"]:
        raise ValueError("Frozen checkpoint SHA mismatch")
    data_audit_path = Path(exp["data_audit_summary"])
    data_audit = json.loads(data_audit_path.read_text())
    train_hash_path = Path(cfg.audit.train_hash_cache)
    if sha256(train_hash_path) != data_audit["train_hash_cache_sha256"] or \
       data_audit["train_val_exact_rgb_hash_overlap"] or data_audit["train_protected_exact_rgb_hash_overlap"]:
        raise ValueError("Training RGB audit mismatch/overlap")
    train_hashes = set(train_hash_path.read_text().splitlines())
    if len(train_hashes) != data_audit["train_unique_rgb_hashes"]:
        raise ValueError("Training RGB cache count mismatch")
    parity_path = Path(exp["independent_entropy_latent_parity"])
    parity = json.loads(parity_path.read_text())
    if parity["checkpoint_sha256"] != exp["hpcm_sha256"] or parity["exactly_equal_samples"] != 8:
        raise ValueError("Independent entropy-latent parity did not pass")
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(args.config.read_text())
    provenance = {
        "git_commit": evaluator._git_metadata()["commit"], "seed": cfg.seed,
        "checkpoints": {
            "adapter": {"path": str(adapter), "sha256": sha256(adapter)},
            "hpcm": {"path": str(hpcm), "sha256": sha256(hpcm)},
            "object_pro": {"path": str(object_pro / "model.safetensors"),
                           "sha256": sha256(object_pro / "model.safetensors")},
            "action_head": {"path": str(object_pro / "action_head--checkpoint.pt"),
                            "sha256": sha256(object_pro / "action_head--checkpoint.pt")},
            "proprio_projector": {"path": str(object_pro / "proprio_projector--checkpoint.pt"),
                                  "sha256": sha256(object_pro / "proprio_projector--checkpoint.pt")},
        },
        "train_hash_cache_sha256": sha256(train_hash_path),
        "data_audit_summary_sha256": sha256(data_audit_path),
        "independent_entropy_parity_sha256": sha256(parity_path),
        "protocol": "Object-Pro, raw 256 HPCM bypass, DEFAULT 50 states/task, seed 7, 10 dummy steps, 8-action queue",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    set_seed_everywhere(cfg.seed)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    if suite.n_tasks != 10:
        raise ValueError("Expected 10 Object tasks")
    # Match the pilot's initialization order for task 0.
    first_env, first_description = evaluator.get_libero_env(suite.get_task(0), cfg.model.model_family, resolution=256)
    model, action_head, proprio_projector, noisy_action_projector, processor = evaluator.initialize_model(cfg)
    codec = ReconstructionControl(hpcm, Path(cfg.model.hpcm.hpcm_root), cfg.model.hpcm.scale_table_levels)
    resize_size = get_image_resize_size(cfg)
    original_get_action = evaluator.get_action
    active = {"task_id": -1, "episode_index": -1, "rates": [], "rate_path": out / "rate.jsonl"}

    def audited_get_action(config, policy, observation, task_label, *action_args, **kwargs):
        query_index = len(active["rates"]) // 2
        for view, rgb in zip(("agent", "wrist"), observation["_hpcm_raw_images"]):
            check_latent = active["episode_index"] == 0 and query_index == 0
            record = {"suite": "libero_object", "task_id": active["task_id"],
                      "episode_index": active["episode_index"], "policy_query_index": query_index,
                      "view": view, **encode_query_image(codec, rgb, check_latent)}
            active["rates"].append(record)
            append(active["rate_path"], record)
        return original_get_action(config, policy, observation, task_label, *action_args, **kwargs)

    evaluator.get_action = audited_get_action
    episode_rows, task_rows = [], []
    start = time.monotonic()
    run_log = (out / "run.log").open("x", encoding="utf-8")
    audit_file = (out / "audit.jsonl").open("x", encoding="utf-8")
    try:
        for task_id in range(10):
            env, description = (first_env, first_description) if task_id == 0 else evaluator.get_libero_env(
                suite.get_task(task_id), cfg.model.model_family, resolution=256)
            states = suite.get_task_init_states(task_id)
            if len(states) < 50:
                raise RuntimeError(f"Task {task_id} has only {len(states)} DEFAULT states")
            current_task = []
            try:
                for episode_index in range(50):
                    active.update(task_id=task_id, episode_index=episode_index, rates=[])
                    episode_log = io.StringIO()
                    success, _, matches = evaluator.run_episode(
                        cfg, env, description, model, resize_size, processor,
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
                    row = {
                        "suite": "libero_object", "task_id": task_id, "task_description": description,
                        "episode_index": episode_index, "initial_state_source": "DEFAULT",
                        "initial_state_sha256": evaluator._sha256_array(states[episode_index]),
                        "success": bool(success), "failure_type": None if success else "horizon_without_success",
                        "exact_train_frame_matches": matches, "policy_queries": len(rates) // 2,
                        "compressed_views": len(rates),
                        "bpp_actual_mean": {
                            view: float(np.mean([r["bpp_actual"] for r in rates if r["view"] == view]))
                            for view in ("agent", "wrist")},
                    }
                    append(out / "episode_results.jsonl", row)
                    episode_rows.append(row)
                    current_task.append(row)
                    print(f"Task {task_id}/9 episode {episode_index}/49: success={success}, "
                          f"bpp={row['bpp_actual_mean']}, exact_train={matches}", flush=True)
            finally:
                env.close()
            task_result = {"task_id": task_id, "task_description": description,
                           "episodes": len(current_task), "success_episodes": sum(r["success"] for r in current_task),
                           "success_rate": sum(r["success"] for r in current_task) / len(current_task),
                           "exact_train_frame_matches": sum(r["exact_train_frame_matches"] for r in current_task)}
            task_rows.append(task_result)
            append(out / "task_results.jsonl", task_result)
    finally:
        evaluator.get_action = original_get_action
        audit_file.close()
        run_log.close()
    if len(episode_rows) != 500 or len(task_rows) != 10:
        raise RuntimeError("Incomplete 10x50 evaluation")
    rate_rows = [json.loads(line) for line in (out / "rate.jsonl").read_text().splitlines() if line.strip()]
    summary = {
        **provenance, "suite": "libero_object", "task_count": 10,
        "total_episodes": 500, "success_episodes": sum(r["success"] for r in episode_rows),
        "success_rate": sum(r["success"] for r in episode_rows) / 500,
        "exact_train_frame_matches": sum(r["exact_train_frame_matches"] for r in episode_rows),
        "compressed_views": len(rate_rows),
        "mean_actual_bpp": {
            view: float(np.mean([r["bpp_actual"] for r in rate_rows if r["view"] == view]))
            for view in ("agent", "wrist")},
        "entropy_latent_checked_views": sum("entropy_latent_max_abs" in r for r in rate_rows),
        "entropy_latent_max_abs": max(r.get("entropy_latent_max_abs", 0.0) for r in rate_rows),
        "per_task": task_rows, "elapsed_seconds": time.monotonic() - start,
        "initial_state_overlap": "unresolved: RLDS PNG export lacks full demo simulator states",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"success_episodes": summary["success_episodes"],
                      "success_rate": summary["success_rate"],
                      "mean_actual_bpp": summary["mean_actual_bpp"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
