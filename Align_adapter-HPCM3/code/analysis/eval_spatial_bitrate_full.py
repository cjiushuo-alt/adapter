"""Paired-protocol Spatial 10x50 codec reconstruction or HPCM3 evaluation.

Each invocation writes one immutable arm/rate result. Inputs are official
DEFAULT states only at evaluation time, never used for training or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for location in (REPO, CODE, REPO / "LIBERO"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from analysis.eval_object_highrate_boundary_task0 import config_from_yaml  # noqa: E402
from analysis.object_hpcm_reconstruction_control import ReconstructionControl, sha256  # noqa: E402
from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from inference import run_libero_eval as evaluator  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.inference_mode()
def encode_reconstruct(codec: ReconstructionControl, rgb: np.ndarray,
                       check_latent: bool) -> tuple[np.ndarray, dict]:
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Expected raw 256 RGB, got {rgb.shape}/{rgb.dtype}")
    rgb = np.ascontiguousarray(rgb)
    # Exactly the HPCM3 raw bypass numerical path: CPU FP32 conversion first.
    raw = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(127.5).sub_(1.0).cuda()
    image01 = raw.add(1.0).mul(0.5)
    bitstream = codec.codec.compress(image01)
    y_bytes, z_bytes = map(len, bitstream["strings"])
    captured = []
    hook = codec.codec.g_s.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach().clone())) if check_latent else None
    try:
        decoded = codec.codec.decompress(bitstream["strings"], bitstream["shape"])["x_hat"]
    finally:
        if hook is not None:
            hook.remove()
    if decoded.shape != image01.shape or not torch.isfinite(decoded).all():
        raise RuntimeError("Invalid HPCM entropy-decoded RGB")
    rgb_hat = decoded.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    mse = float(np.square(rgb.astype(np.float64) - rgb_hat.astype(np.float64)).mean())
    record = {"source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "reconstructed_rgb_sha256": hashlib.sha256(rgb_hat.tobytes()).hexdigest(),
              "y_bytes": y_bytes, "z_bytes": z_bytes,
              "bpp_actual": 8.0 * (y_bytes + z_bytes) / (256 * 256),
              "psnr_db": 10.0 * math.log10(255.0**2 / mse) if mse else None}
    if check_latent:
        if len(captured) != 1:
            raise RuntimeError("Could not capture entropy-decoded latent")
        forward = codec.encoder(image01)
        if captured[0].shape != forward.shape:
            raise RuntimeError("HPCM latent shape mismatch")
        difference = (captured[0] - forward).float().abs()
        record["entropy_latent_max_abs"] = float(difference.max())
        record["entropy_latent_nonzero_elements"] = int(torch.count_nonzero(difference))
        if record["entropy_latent_nonzero_elements"]:
            raise RuntimeError("HPCM3 forward latent differs from actual entropy decoded latent")
    return rgb_hat, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("reconstruction", "hpcm3"), required=True)
    parser.add_argument("--lambda-value", choices=("0.0018", "0.0483"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    base_path = REPO / "results/hpcm3_libero4/spatial/config.yaml"
    raw = yaml.safe_load(base_path.read_text())
    hpcm = REPO / "HPCM/ckpt" / f"{args.lambda_value}.pth.tar"
    adapter_root = ("hpcm3_spatial_lowrate_libero4_full" if args.lambda_value == "0.0018"
                    else "hpcm3_spatial_highrate_libero4_full")
    adapter = Path("/media/cjs/shared/linux/Align_adapter") / adapter_root / "adapter-best.pt"
    raw["model"]["hpcm"]["checkpoint"] = str(hpcm)
    raw["model"]["adapter"]["checkpoint"] = str(adapter)
    raw["model"]["use_hpcm_adapter"] = args.arm == "hpcm3"
    raw["model"]["hpcm_raw_input_bypass"] = args.arm == "hpcm3"
    raw["model"]["save_version"] = f"spatial-bitrate-{args.lambda_value}-{args.arm}"
    raw["logging"]["local_log_dir"] = str(args.output_dir)
    raw["logging"]["episode_results_jsonl"] = str(args.output_dir / "episode_results.jsonl")
    raw["logging"]["summary_json"] = str(args.output_dir / "summary.json")
    raw["audit"]["output_jsonl"] = str(args.output_dir / "audit.jsonl")
    raw["audit"]["train_hash_cache"] = "/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_libero4_full/data_audit/train_rgb_sha256.txt"
    cfg = config_from_yaml(raw)
    if not (cfg.libero.task_suite_name == "libero_spatial" and cfg.libero.num_trials_per_task == 50 and
            cfg.libero.initial_states_path == "DEFAULT" and cfg.libero.env_img_res == 256 and
            cfg.libero.num_steps_wait == 10 and cfg.model.num_open_loop_steps == 8 and
            cfg.model.unnorm_key == "libero_spatial_no_noops" and cfg.model.center_crop and cfg.seed == 7):
        raise ValueError("Wrong Spatial 10x50 protocol")
    preflight = json.loads(Path("/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full/bitrate_preflight.json").read_text())
    rate_info = next(r for r in preflight["rates"] if r["lambda"] == args.lambda_value)
    if rate_info["checkpoint_sha256"] != sha256(hpcm):
        raise ValueError("HPCM checkpoint differs from independent preflight")
    audit = json.loads(Path("/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full/data_protocol_audit.v2.json").read_text())
    train_hash_path = Path(raw["audit"]["train_hash_cache"])
    if sha256(train_hash_path) != audit["train_hash_cache_sha256"] or audit["formal_state_count"] != 500:
        raise ValueError("Training hash/protected state audit changed")
    train_hashes = set(train_hash_path.read_text().splitlines())
    if len(train_hashes) != audit["train_unique_rgb_hashes"]:
        raise ValueError("Training RGB cache count changed")
    if args.arm == "hpcm3" and not adapter.is_file():
        raise FileNotFoundError("High-rate Spatial Adapter not trained")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    spatial = REPO / "outputs/LIBERO-Spatial-Pro"
    provenance = {"git_commit": evaluator._git_metadata()["commit"], "seed": 7,
                  "arm": args.arm, "lambda": args.lambda_value,
                  "checkpoints": {
                      "hpcm": {"path": str(hpcm), "sha256": sha256(hpcm)},
                      "adapter": {"path": str(adapter), "sha256": sha256(adapter)} if args.arm == "hpcm3" else None,
                      "spatial_pro": {name: sha256(spatial / name) for name in
                                      ("model.safetensors", "action_head--checkpoint.pt", "proprio_projector--checkpoint.pt")}},
                  "preflight_sha256": sha256(Path("/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full/bitrate_preflight.json")),
                  "data_audit_sha256": sha256(Path("/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full/data_protocol_audit.v2.json")),
                  "protocol": "Spatial-Pro, DEFAULT states 0-49/task, seed 7, 10 dummy steps, 8-action queue; same raw RGB entropy codec for both arms; native VLA processor for reconstructed RGB; raw 256 HPCM bypass for HPCM3"}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    set_seed_everywhere(cfg.seed)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if suite.n_tasks != 10:
        raise ValueError("Expected ten Spatial tasks")
    first_env, first_description = evaluator.get_libero_env(suite.get_task(0), cfg.model.model_family, resolution=256)
    policy, action_head, proprio_projector, noisy_action_projector, processor = evaluator.initialize_model(cfg)
    codec = ReconstructionControl(hpcm, REPO / "HPCM", 60)
    resize_size = get_image_resize_size(cfg)
    original_get_action = evaluator.get_action
    active = {"task_id": -1, "episode_index": -1, "rates": []}

    def audited_get_action(config, model, observation, task_label, *action_args, **kwargs):
        query = len(active["rates"]) // 2
        reconstructed = []
        for view, rgb in zip(("agent", "wrist"), observation["_hpcm_raw_images"]):
            decoded, metrics = encode_reconstruct(codec, rgb,
                check_latent=active["episode_index"] == 0 and query == 0)
            record = {"suite": "libero_spatial", "task_id": active["task_id"],
                      "episode_index": active["episode_index"], "policy_query_index": query,
                      "view": view, **metrics}
            active["rates"].append(record)
            append(args.output_dir / "rate.jsonl", record)
            reconstructed.append(resize_image_for_policy(decoded, resize_size))
        if args.arm == "reconstruction":
            observation["full_image"], observation["wrist_image"] = reconstructed
        return original_get_action(config, model, observation, task_label, *action_args, **kwargs)

    evaluator.get_action = audited_get_action
    episodes, task_rows = [], []
    start = time.monotonic()
    run_log = (args.output_dir / "run.log").open("x", encoding="utf-8")
    audit_file = (args.output_dir / "audit.jsonl").open("x", encoding="utf-8")
    try:
        for task_id in range(10):
            env, description = (first_env, first_description) if task_id == 0 else evaluator.get_libero_env(
                suite.get_task(task_id), cfg.model.model_family, resolution=256)
            states = suite.get_task_init_states(task_id)
            if len(states) < 50:
                raise RuntimeError("Fewer than 50 official states")
            current_task = []
            try:
                for episode_index in range(50):
                    active.update(task_id=task_id, episode_index=episode_index, rates=[])
                    episode_log = io.StringIO()
                    success, _, matches = evaluator.run_episode(
                        cfg, env, description, policy, resize_size, processor,
                        action_head, proprio_projector, noisy_action_projector,
                        initial_state=states[episode_index], log_file=episode_log,
                        audit_file=audit_file,
                        audit_context={"suite": "libero_spatial", "task_id": task_id,
                                       "episode_index": episode_index, "initial_state_source": "DEFAULT"},
                        training_frame_hashes=train_hashes)
                    run_log.write(f"TASK {task_id} EPISODE {episode_index}\n" + episode_log.getvalue() + "\n")
                    run_log.flush()
                    if "Episode error:" in episode_log.getvalue() or not active["rates"]:
                        raise RuntimeError(f"Evaluation error at task {task_id} episode {episode_index}")
                    rates = active["rates"]
                    row = {"suite": "libero_spatial", "arm": args.arm, "lambda": args.lambda_value,
                           "task_id": task_id, "task_description": description,
                           "episode_index": episode_index, "initial_state_source": "DEFAULT",
                           "initial_state_sha256": evaluator._sha256_array(states[episode_index]),
                           "success": bool(success), "failure_type": None if success else "horizon_without_success",
                           "exact_train_frame_matches": matches, "policy_queries": len(rates) // 2,
                           "bpp_actual_mean": {view: float(np.mean([r["bpp_actual"] for r in rates if r["view"] == view]))
                                               for view in ("agent", "wrist")},
                           "psnr_db_mean": {view: float(np.mean([r["psnr_db"] for r in rates if r["view"] == view]))
                                            for view in ("agent", "wrist")}}
                    append(args.output_dir / "episode_results.jsonl", row)
                    episodes.append(row)
                    current_task.append(row)
                    print(f"{args.lambda_value} {args.arm} task {task_id}/9 episode {episode_index}/49: success={success}", flush=True)
            finally:
                env.close()
            task_result = {"task_id": task_id, "task_description": description, "episodes": 50,
                           "success_episodes": sum(r["success"] for r in current_task),
                           "success_rate": sum(r["success"] for r in current_task) / 50,
                           "exact_train_frame_matches": sum(r["exact_train_frame_matches"] for r in current_task)}
            task_rows.append(task_result)
            append(args.output_dir / "task_results.jsonl", task_result)
    finally:
        evaluator.get_action = original_get_action
        audit_file.close()
        run_log.close()
    if len(episodes) != 500 or len(task_rows) != 10:
        raise RuntimeError("Incomplete 10x50 Spatial evaluation")
    rate_rows = [json.loads(line) for line in (args.output_dir / "rate.jsonl").read_text().splitlines() if line.strip()]
    summary = {**provenance, "suite": "libero_spatial", "task_count": 10, "total_episodes": 500,
               "success_episodes": sum(r["success"] for r in episodes),
               "success_rate": sum(r["success"] for r in episodes) / 500,
               "exact_train_frame_matches": sum(r["exact_train_frame_matches"] for r in episodes),
               "compressed_views": len(rate_rows),
               "mean_actual_bpp": {view: float(np.mean([r["bpp_actual"] for r in rate_rows if r["view"] == view]))
                                   for view in ("agent", "wrist")},
               "mean_psnr_db": {view: float(np.mean([r["psnr_db"] for r in rate_rows if r["view"] == view]))
                                for view in ("agent", "wrist")},
               "entropy_latent_checked_views": sum("entropy_latent_max_abs" in r for r in rate_rows),
               "entropy_latent_max_abs": max(r.get("entropy_latent_max_abs", 0.0) for r in rate_rows),
               "per_task": task_rows, "elapsed_seconds": time.monotonic() - start,
               "demo_initial_state_overlap": "unresolved: RLDS PNG export lacks full demo simulator states"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arm": args.arm, "lambda": args.lambda_value,
                      "success_episodes": summary["success_episodes"],
                      "mean_actual_bpp": summary["mean_actual_bpp"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
