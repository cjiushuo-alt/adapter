"""Protected five-state Object-Pro A/B/new HPCM3 task-0 comparison.

Run only after early-stopped training has saved adapter-best.pt. The A and B
arms are read-only historical controls; only the new C arm is rolled out.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for root in (REPO, CODE, REPO / "LIBERO"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from libero.libero import benchmark  # noqa: E402
from inference import run_libero_eval as evaluator  # noqa: E402
from analysis.object_hpcm_reconstruction_control import ReconstructionControl, sha256  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402


def config_from_yaml(raw: dict) -> evaluator.InferenceConfig:
    model = dict(raw["model"])
    model["hpcm"] = evaluator.HPCMConfig(**model["hpcm"])
    model["adapter"] = evaluator.AdapterConfig(**model["adapter"])
    model["vision_backbone"] = evaluator.VisionBackboneConfig(**model["vision_backbone"])
    return evaluator.InferenceConfig(
        model=evaluator.ModelConfig(**model),
        libero=evaluator.LiberoConfig(**raw["libero"]),
        logging=evaluator.LoggingConfig(**raw["logging"]),
        audit=evaluator.AuditConfig(**raw["audit"]),
        guard=evaluator.GuardConfig(enabled=False),
        seed=int(raw["seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    cfg = config_from_yaml(raw)
    assert cfg.model.use_hpcm_adapter and cfg.model.hpcm_raw_input_bypass
    assert cfg.libero.task_suite_name == "libero_object" and cfg.libero.num_trials_per_task == 5
    assert cfg.seed == 7 and cfg.model.num_open_loop_steps == 8 and cfg.libero.num_steps_wait == 10
    assert cfg.model.hpcm.checkpoint.endswith("0.0483.pth.tar")
    assert cfg.model.adapter.checkpoint.endswith("hpcm3_object_highrate_boundary/adapter-best.pt")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    adapter_path = Path(cfg.model.adapter.checkpoint)
    if not adapter_path.is_file():
        raise FileNotFoundError("Early-stopped high-rate Adapter is not ready")
    reference = {}
    for arm, key in (("A", "baseline_results"), ("B", "reconstructed_results")):
        path = Path(raw["experiment"][key])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) >= 5
        reference[arm] = (path, rows[:5])
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    states = suite.get_task_init_states(0)
    for i in range(5):
        state_hash = evaluator._sha256_array(states[i])
        assert reference["A"][1][i]["initial_state_sha256"] == state_hash
        assert reference["B"][1][i]["initial_state_sha256"] == state_hash
    train_hash_path = Path(cfg.audit.train_hash_cache)
    train_hashes = set(train_hash_path.read_text().splitlines())
    assert len(train_hashes) == 16384
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.yaml").write_text(args.config.read_text())
    provenance = {
        "git_commit": evaluator._git_metadata()["commit"], "seed": cfg.seed,
        "checkpoints": {
            "hpcm": {"path": cfg.model.hpcm.checkpoint, "sha256": sha256(Path(cfg.model.hpcm.checkpoint))},
            "adapter": {"path": str(adapter_path), "sha256": sha256(adapter_path)},
            "vla_model": {"path": str(Path(cfg.model.pretrained_checkpoint) / "model.safetensors"),
                          "sha256": sha256(Path(cfg.model.pretrained_checkpoint) / "model.safetensors")},
            "action_head": {"path": str(Path(cfg.model.pretrained_checkpoint) / "action_head--checkpoint.pt"),
                            "sha256": sha256(Path(cfg.model.pretrained_checkpoint) / "action_head--checkpoint.pt")},
            "proprio_projector": {"path": str(Path(cfg.model.pretrained_checkpoint) / "proprio_projector--checkpoint.pt"),
                                  "sha256": sha256(Path(cfg.model.pretrained_checkpoint) / "proprio_projector--checkpoint.pt")},
        },
        "train_hash_cache": {"path": str(train_hash_path), "sha256": sha256(train_hash_path),
                             "hash_count": len(train_hashes)},
        "references": {arm: {"path": str(path), "sha256": sha256(path)}
                       for arm, (path, _) in reference.items()},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    set_seed_everywhere(cfg.seed)
    env, description = evaluator.get_libero_env(suite.get_task(0), cfg.model.model_family, resolution=256)
    model, action_head, proprio_projector, noisy_action_projector, processor = evaluator.initialize_model(cfg)
    codec = ReconstructionControl(Path(cfg.model.hpcm.checkpoint), Path(cfg.model.hpcm.hpcm_root),
                                  cfg.model.hpcm.scale_table_levels)
    resize_size = get_image_resize_size(cfg)
    original_get_action = evaluator.get_action
    episode_rates = []
    episode_index = -1

    def audited_get_action(config, policy, observation, task_label, *args_, **kwargs):
        nonlocal episode_rates
        raw_images = observation["_hpcm_raw_images"]
        for view, rgb in zip(("agent", "wrist"), raw_images):
            _, metrics = codec.reconstruct(rgb)
            record = {"episode_index": episode_index, "view": view,
                      "policy_query_index": len(episode_rates) // 2,
                      "source_rgb_sha256": metrics["source_rgb_sha256"],
                      "bpp_actual": metrics["bpp_actual"],
                      "y_bytes": metrics["y_bytes"], "z_bytes": metrics["z_bytes"]}
            episode_rates.append(record)
            evaluator.write_audit_record(rate_file, record)
        return original_get_action(config, policy, observation, task_label, *args_, **kwargs)

    evaluator.get_action = audited_get_action
    episodes = []
    try:
        with (args.output_dir / "run.log").open("x") as run_log, \
             (args.output_dir / "audit.jsonl").open("x") as audit_file, \
             (args.output_dir / "rate.jsonl").open("x") as rate_file, \
             (args.output_dir / "episode_results.jsonl").open("x") as episode_file:
            for i in range(5):
                episode_index = i
                episode_rates = []
                episode_log = io.StringIO()
                success, _, exact_matches = evaluator.run_episode(
                    cfg, env, description, model, resize_size, processor,
                    action_head, proprio_projector, noisy_action_projector,
                    initial_state=states[i], log_file=episode_log,
                    audit_file=audit_file,
                    audit_context={"suite": "libero_object", "task_id": 0, "episode_index": i},
                    training_frame_hashes=train_hashes,
                )
                run_log.write(episode_log.getvalue())
                run_log.flush()
                assert "Episode error:" not in episode_log.getvalue() and episode_rates
                row = {"suite": "libero_object", "task_id": 0, "episode_index": i,
                       "initial_state_sha256": evaluator._sha256_array(states[i]),
                       "A_success": bool(reference["A"][1][i]["success"]),
                       "B_success": bool(reference["B"][1][i]["B_success"]),
                       "C_success": bool(success), "exact_train_frame_matches": exact_matches,
                       "policy_queries": len(episode_rates) // 2,
                       "actual_bpp": {view: float(np.mean([r["bpp_actual"] for r in episode_rates if r["view"] == view]))
                                      for view in ("agent", "wrist")}}
                episodes.append(row)
                evaluator.write_audit_record(episode_file, row)
                print(f"Episode {i}: C_success={success}, queries={row['policy_queries']}, exact_train_matches={exact_matches}", flush=True)
    finally:
        evaluator.get_action = original_get_action
        env.close()
    all_rates = [json.loads(line) for line in (args.output_dir / "rate.jsonl").read_text().splitlines()]
    summary = {
        **provenance, "suite": "libero_object", "task_id": 0, "task_description": description,
        "episodes": 5, "successes": {arm: sum(row[f"{arm}_success"] for row in episodes)
                                      for arm in ("A", "B", "C")},
        "success_rates": {arm: sum(row[f"{arm}_success"] for row in episodes) / 5
                          for arm in ("A", "B", "C")},
        "exact_train_frame_matches": sum(row["exact_train_frame_matches"] for row in episodes),
        "mean_actual_bpp": {view: float(np.mean([r["bpp_actual"] for r in all_rates if r["view"] == view]))
                            for view in ("agent", "wrist")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
