"""Fail-closed four-suite trajectory/data audit before high-rate Object training.

The official rollout RGB hashes are read only for leakage detection, never fed
to training/validation/selection. RLDS image exports do not contain the full
simulator initial state, so same-initial-state equivalence remains unknown.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

REPO = Path(__file__).resolve().parents[3]
EXPECTED_SUITES = (
    "libero_10_no_noops", "libero_goal_no_noops",
    "libero_object_no_noops", "libero_spatial_no_noops",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_hash(path: str) -> str:
    with Image.open(path) as image:
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if rgb.shape != (256, 256, 3):
        raise ValueError(f"Expected native 256 RGB: {path}, got {rgb.shape}")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def paths_for(episodes: list[str]) -> list[str]:
    result = []
    for episode in episodes:
        files = sorted(path for path in Path(episode).glob("step_*_image.png")
                       if not path.name.endswith("_wrist_image.png"))
        files.extend(sorted(Path(episode).glob("step_*_wrist_image.png")))
        if not files:
            raise ValueError(f"Empty trajectory: {episode}")
        result.extend(str(path) for path in files)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    data = cfg["data"]
    if tuple(data["dataset_names"]) != EXPECTED_SUITES or data["split_unit"] != "trajectory":
        raise ValueError("Expected the four suites and trajectory split")
    if data.get("train_max_samples") is not None:
        raise ValueError("Full-data experiment must not have train_max_samples")
    manifest_path = Path(data["split_manifest_path"])
    split = json.loads(manifest_path.read_text())
    if split["split_unit"] != "trajectory" or split["seed"] != data["seed"]:
        raise ValueError("Wrong split manifest")
    train_episodes, val_episodes = split["train_episodes"], split["val_episodes"]
    if set(train_episodes) & set(val_episodes):
        raise ValueError("Train/val trajectories overlap")
    dataset_root = Path(data["dataset_dir"]).resolve()
    actual = {str(path.resolve()) for suite in EXPECTED_SUITES
              for path in (dataset_root / suite).glob("episode_*") if path.is_dir()}
    if set(train_episodes) | set(val_episodes) != actual:
        raise ValueError("Split manifest no longer covers all source trajectories")
    if not all(Path(episode).is_dir() for episode in actual):
        raise FileNotFoundError("An image trajectory is missing")

    sim_manifest = Path(cfg["simulator_validation"]["manifest_path"])
    sim = json.loads(sim_manifest.read_text())
    if sim.get("source") != "fresh_env_reset_not_default_pruned_init" or sim.get("protected_state_collisions") != 0:
        raise ValueError("Simulator validation is not protected-state independent")
    sim_state_hashes = {row["state_sha256"] for row in sim["samples"]}
    if len(sim["samples"]) != 40 or len(sim_state_hashes) != 20:
        raise ValueError("Expected the independent 40-image/20-state validation set")

    protected_hashes = set()
    protected_states = set()
    audit_files = sorted((REPO / "results").glob("**/audit.jsonl"))
    used_audit_files = []
    for path in audit_files:
        if "object" not in str(path).lower():
            continue
        found = False
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != "libero_object":
                continue
            found = True
            protected_states.add(row.get("initial_state_sha256"))
            protected_hashes.update(value for key, value in row.items()
                                    if key in {"main_rgb_sha256", "wrist_rgb_sha256"} and value)
        if found:
            used_audit_files.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path)})
    if not protected_hashes or not protected_states:
        raise ValueError("No historical formal Object audit hashes were found")
    if protected_states & sim_state_hashes:
        raise ValueError("Independent simulator validation collided with protected states")

    train_paths, val_paths = paths_for(train_episodes), paths_for(val_episodes)
    suite_counts = defaultdict(lambda: Counter())
    for split_name, episodes, paths in (
        ("train", train_episodes, train_paths), ("val", val_episodes, val_paths)
    ):
        for episode in episodes:
            suite_counts[Path(episode).parent.name][f"{split_name}_trajectories"] += 1
        for path in paths:
            suite_counts[Path(path).parent.parent.name][f"{split_name}_images"] += 1

    output.mkdir(parents=True)
    hashes = {}
    for name, paths in (("train", train_paths), ("val", val_paths)):
        cache = output / f"{name}_rgb_sha256.txt"
        unique = set()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, digest in enumerate(pool.map(rgb_hash, paths, chunksize=128), start=1):
                unique.add(digest)
                if index % 50000 == 0:
                    print(f"{name}: hashed {index}/{len(paths)} images", flush=True)
        with cache.open("x") as stream:
            stream.write("\n".join(sorted(unique)) + "\n")
        hashes[name] = unique
    protected_matches = hashes["train"] & protected_hashes
    train_val_matches = hashes["train"] & hashes["val"]
    report = {
        "config_sha256": sha256(args.config),
        "split_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path),
                           "unit": "trajectory", "train": len(train_episodes), "val": len(val_episodes),
                           "overlap": 0},
        "sim_manifest": {"path": str(sim_manifest), "sha256": sha256(sim_manifest),
                         "images": len(sim["samples"]), "protected_state_collisions": 0},
        "suite_counts": {name: dict(suite_counts[name]) for name in EXPECTED_SUITES},
        "train_images": len(train_paths), "val_images": len(val_paths),
        "train_unique_rgb_hashes": len(hashes["train"]),
        "val_unique_rgb_hashes": len(hashes["val"]),
        "train_val_exact_rgb_hash_overlap": len(train_val_matches),
        "protected_rollout_audit_files": used_audit_files,
        "protected_rollout_state_hashes": len(protected_states),
        "protected_rollout_rgb_hashes": len(protected_hashes),
        "train_protected_exact_rgb_hash_overlap": len(protected_matches),
        "initial_state_overlap": "unknown: RLDS PNG export lacks complete simulator initial states",
        "train_hash_cache_sha256": sha256(output / "train_rgb_sha256.txt"),
        "val_hash_cache_sha256": sha256(output / "val_rgb_sha256.txt"),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if protected_matches or train_val_matches:
        raise RuntimeError("Exact RGB leakage detected; refuse to train")


if __name__ == "__main__":
    main()
