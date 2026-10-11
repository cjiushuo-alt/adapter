"""Object-Pro baseline with real HPCM entropy-coded RGB reconstruction.

Reuses the established LIBERO task-0 rollout loop and intact policy. Only the
two policy-query RGB images are replaced after HPCM compress/decompress.
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
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for path in (REPO, CODE, REPO / "LIBERO"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset.hpcm_encoder import HPCMEncoder  # noqa: E402
from inference import run_libero_eval as evaluator  # noqa: E402
from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class ReconstructionControl:
    def __init__(self, checkpoint: Path, hpcm_root: Path, scale_levels: int):
        self.encoder = HPCMEncoder(
            hpcm_root=str(hpcm_root), checkpoint_path=str(checkpoint),
            model_name="HPCM_Base", scale_table_levels=scale_levels, pad_multiple=256,
        ).cuda().eval()
        self.codec = self.encoder._hpcm_model
        assert self.codec is not None

    @torch.inference_mode()
    def reconstruct(self, rgb: np.ndarray, *, check_forward: bool = False) -> tuple[np.ndarray, dict]:
        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
        original = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).cuda().float().div_(255)
        assert original.shape == (1, 3, 256, 256)  # no padding is needed
        encoded = self.codec.compress(original)
        assert len(encoded["strings"]) == 2 and all(isinstance(s, bytes) and len(s) > 0 for s in encoded["strings"])
        decoded = self.codec.decompress(encoded["strings"], encoded["shape"])["x_hat"]
        assert decoded.shape == original.shape and torch.isfinite(decoded).all()
        decoded = decoded.clamp(0, 1)
        reconstructed = decoded.mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        mse = float(np.square(rgb.astype(np.float64) - reconstructed.astype(np.float64)).mean())
        stats = {
            "psnr_db": float(10 * math.log10(255.0**2 / mse)) if mse > 0 else None,
            "mse_u8": mse,
            "bpp_actual": float(8 * sum(len(s) for s in encoded["strings"]) / (256 * 256)),
            "y_bytes": len(encoded["strings"][0]), "z_bytes": len(encoded["strings"][1]),
            "source_rgb_sha256": sha256_array(rgb),
            "reconstructed_rgb_sha256": sha256_array(reconstructed),
        }
        if check_forward:
            # Quantization/reconstruction sanity check, not a replacement for
            # the real bitstream used above. HPCM forward returns x_hat.
            forward = self.codec(original, training=False)["x_hat"].clamp(0, 1)
            stats["forward_vs_decoded_mae"] = float((forward - decoded).abs().mean())
            stats["forward_vs_decoded_max_abs"] = float((forward - decoded).abs().max())
        return reconstructed, stats


def config_from_yaml(raw: dict) -> evaluator.InferenceConfig:
    model = dict(raw["model"])
    model["hpcm"] = evaluator.HPCMConfig(**model["hpcm"])
    model["adapter"] = evaluator.AdapterConfig(**model["adapter"])
    model["vision_backbone"] = evaluator.VisionBackboneConfig(**model["vision_backbone"])
    model["use_hpcm_adapter"] = False
    model["hpcm_raw_input_bypass"] = False
    return evaluator.InferenceConfig(
        model=evaluator.ModelConfig(**model),
        libero=evaluator.LiberoConfig(**raw["libero"]),
        logging=evaluator.LoggingConfig(**raw["logging"]),
        audit=evaluator.AuditConfig(enabled=False),
        guard=evaluator.GuardConfig(enabled=False),
        seed=int(raw["seed"]),
    )


def validate_controls(a_path: Path, c_path: Path, count: int) -> tuple[list[dict], list[dict]]:
    a = [json.loads(line) for line in a_path.read_text().splitlines() if line.strip()]
    c = [json.loads(line) for line in c_path.read_text().splitlines() if line.strip()]
    assert len(a) >= count and len(c) >= count, "A/C controls must cover all requested episodes"
    for index in range(count):
        assert a[index]["episode_index"] == c[index]["episode_index"] == index
        assert a[index]["task_id"] == c[index]["task_id"] == 0
        assert a[index]["initial_state_sha256"] == c[index]["initial_state_sha256"]
    return a, c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-images-dir", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    cfg = config_from_yaml(raw)
    count = int(raw["experiment"]["episodes"])
    assert count in (5, 20), "Supported screening sizes are five and twenty episodes"
    reference_count = min(count, 5)
    assert cfg.libero.task_suite_name == "libero_object" and cfg.libero.initial_states_path == "DEFAULT"
    assert cfg.libero.num_steps_wait == 10 and cfg.libero.env_img_res == 256
    assert cfg.model.center_crop and cfg.model.num_open_loop_steps == 8 and cfg.seed == 7
    paths = raw["experiment"]
    hpcm_path = Path(cfg.model.hpcm.checkpoint)
    adapter_path = Path(cfg.model.adapter.checkpoint)
    vla_dir = Path(cfg.model.pretrained_checkpoint)
    a, c = validate_controls(Path(paths["baseline_results"]), Path(paths["adapter_results"]), reference_count)
    assert all(bool(row["success"]) for row in a[:reference_count])
    assert not any(bool(row["success"]) for row in c[:reference_count])
    set_seed_everywhere(cfg.seed)
    codec = ReconstructionControl(hpcm_path, Path(cfg.model.hpcm.hpcm_root), cfg.model.hpcm.scale_table_levels)

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    states = suite.get_task_init_states(0)
    for index in range(reference_count):
        assert evaluator._sha256_array(states[index]) == a[index]["initial_state_sha256"]
    env, description = evaluator.get_libero_env(suite.get_task(0), cfg.model.model_family, resolution=256)
    try:
        if args.probe_only:
            env.reset()
            obs = env.set_init_state(states[0])
            for _ in range(cfg.libero.num_steps_wait):
                obs, _, _, _ = env.step(evaluator.get_libero_dummy_action(cfg.model.model_family))
            rows = []
            for view, image in (("agent", evaluator.get_libero_image(obs)), ("wrist", evaluator.get_libero_wrist_image(obs))):
                reconstructed, stats = codec.reconstruct(image, check_forward=True)
                rows.append({"view": view, **stats})
                if args.probe_images_dir is not None:
                    args.probe_images_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(args.probe_images_dir / f"{view}_original.png")
                    Image.fromarray(reconstructed).save(args.probe_images_dir / f"{view}_reconstructed.png")
            print(json.dumps({"probe": rows, "hpcm_sha256": sha256(hpcm_path)}, indent=2), flush=True)
            return

        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        for filename in ("episode_results.jsonl", "frame_metrics.jsonl", "summary.json", "provenance.json", "config.yaml", "run.log"):
            assert not (out / filename).exists(), f"Will not overwrite {out / filename}"
        (out / "config.yaml").write_text(args.config.read_text())
        provenance = {
            "git_commit": evaluator._git_metadata()["commit"], "seed": cfg.seed,
            "checkpoints": {
                "hpcm": {"path": str(hpcm_path), "sha256": sha256(hpcm_path)},
                "adapter_epoch29_C_reference": {"path": str(adapter_path), "sha256": sha256(adapter_path)},
                "vla_model": {"path": str(vla_dir / "model.safetensors"), "sha256": sha256(vla_dir / "model.safetensors")},
                "action_head": {"path": str(vla_dir / "action_head--checkpoint.pt"), "sha256": sha256(vla_dir / "action_head--checkpoint.pt")},
                "proprio_projector": {"path": str(vla_dir / "proprio_projector--checkpoint.pt"), "sha256": sha256(vla_dir / "proprio_projector--checkpoint.pt")},
            },
            "control_files": {
                "A": {"path": paths["baseline_results"], "sha256": sha256(Path(paths["baseline_results"]))},
                "C": {"path": paths["adapter_results"], "sha256": sha256(Path(paths["adapter_results"]))},
            },
            "protocol": "HPCM.compress/decompress actual y+z bitstreams on both raw 256px views at every policy query; then intact Object-Pro native JPEG/Lanczos3 224px+center-crop processor; 10 dummy steps; eight-action queue",
            "reference_A_C_episodes": reference_count,
        }
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        model, action_head, proprio_projector, noisy_action_projector, processor = evaluator.initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)
        original_prepare = evaluator.prepare_observation
        original_get_action = evaluator.get_action
        current_frames = []
        current_episode = -1

        def prepare_with_original(obs, resize):
            observation, agent, wrist = original_prepare(obs, resize)
            observation["_original_rgb_for_compression"] = (agent, wrist)
            return observation, agent, wrist

        def action_with_reconstruction(config, policy, observation, task_label, *action_args, **kwargs):
            nonlocal current_frames
            source = observation.pop("_original_rgb_for_compression")
            reconstructed = []
            for view, rgb in zip(("agent", "wrist"), source):
                rgb_hat, stats = codec.reconstruct(rgb, check_forward=(current_episode == 0 and not current_frames))
                stats.update({"episode_index": current_episode, "view": view, "policy_query_index": len(current_frames) // 2})
                current_frames.append(stats)
                append_jsonl(out / "frame_metrics.jsonl", stats)
                reconstructed.append(resize_image_for_policy(rgb_hat, resize_size))
            observation["full_image"], observation["wrist_image"] = reconstructed
            return original_get_action(config, policy, observation, task_label, *action_args, **kwargs)

        evaluator.prepare_observation = prepare_with_original
        evaluator.get_action = action_with_reconstruction
        episodes = []
        start_time = time.monotonic()
        run_log = (out / "run.log").open("x", encoding="utf-8")
        try:
            for index in range(count):
                current_episode = index
                current_frames = []
                print(f"Starting B episode {index}/{count - 1}", flush=True)
                episode_log = io.StringIO()
                success, _, _ = evaluator.run_episode(
                    cfg, env, description, model, resize_size, processor,
                    action_head, proprio_projector, noisy_action_projector,
                    initial_state=states[index],
                    log_file=episode_log,
                )
                run_log.write(episode_log.getvalue())
                run_log.flush()
                assert "Episode error:" not in episode_log.getvalue(), f"Episode {index} raised an evaluation exception"
                assert current_frames, f"No policy-query frames in episode {index}"
                record = {
                    "suite": "libero_object", "task_id": 0, "episode_index": index,
                    "initial_state_sha256": evaluator._sha256_array(states[index]),
                    "A_success": bool(a[index]["success"]) if index < reference_count else None,
                    "B_success": bool(success),
                    "C_success": bool(c[index]["success"]) if index < reference_count else None,
                    "policy_queries": len(current_frames) // 2, "frames": len(current_frames),
                    "agent_psnr_db": float(np.mean([r["psnr_db"] for r in current_frames if r["view"] == "agent"])),
                    "wrist_psnr_db": float(np.mean([r["psnr_db"] for r in current_frames if r["view"] == "wrist"])),
                    "agent_bpp_actual": float(np.mean([r["bpp_actual"] for r in current_frames if r["view"] == "agent"])),
                    "wrist_bpp_actual": float(np.mean([r["bpp_actual"] for r in current_frames if r["view"] == "wrist"])),
                }
                episodes.append(record)
                append_jsonl(out / "episode_results.jsonl", record)
                print(f"B episode {index}: success={success}, queries={record['policy_queries']}, agent PSNR={record['agent_psnr_db']:.2f}, wrist PSNR={record['wrist_psnr_db']:.2f}", flush=True)
        finally:
            run_log.close()
            evaluator.prepare_observation = original_prepare
            evaluator.get_action = original_get_action
        frames = [json.loads(line) for line in (out / "frame_metrics.jsonl").read_text().splitlines()]
        summary = {
            **provenance, "task_description": description, "episodes": len(episodes),
            "successes": {arm: sum(bool(row[f"{arm}_success"]) for row in episodes if row[f"{arm}_success"] is not None)
                          for arm in ("A", "B", "C")},
            "success_rates": {arm: sum(bool(row[f"{arm}_success"]) for row in episodes if row[f"{arm}_success"] is not None) /
                              sum(row[f"{arm}_success"] is not None for row in episodes)
                              for arm in ("A", "B", "C")},
            "reference_A_C_episodes": reference_count,
            "policy_queries": len(frames) // 2, "compressed_frames": len(frames),
            "mean_psnr_db": {view: float(np.mean([r["psnr_db"] for r in frames if r["view"] == view])) for view in ("agent", "wrist")},
            "mean_actual_bpp": {view: float(np.mean([r["bpp_actual"] for r in frames if r["view"] == view])) for view in ("agent", "wrist")},
            "elapsed_seconds": time.monotonic() - start_time,
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"successes": summary["successes"], "mean_psnr_db": summary["mean_psnr_db"], "mean_actual_bpp": summary["mean_actual_bpp"]}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
