"""Read-only split and protected-state audit for the Spatial bitrate study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split_path = Path("/media/cjs/shared/linux/Align_adapter/hpcm3_object_dist_libero4/data_split.json")
    train_cache = Path("/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_libero4_full/data_audit/train_rgb_sha256.txt")
    val_cache = Path("/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_libero4_full/data_audit/val_rgb_sha256.txt")
    spatial_audit = REPO / "results/hpcm3_libero4/spatial/audit.jsonl"
    split = json.loads(split_path.read_text())
    sim = json.loads(args.sim_manifest.read_text())
    if split["split_unit"] != "trajectory" or set(split["train_episodes"]) & set(split["val_episodes"]):
        raise ValueError("Trajectory train/val split is invalid")
    expected = {str(p.resolve()) for suite in ("libero_10_no_noops", "libero_goal_no_noops",
                                               "libero_object_no_noops", "libero_spatial_no_noops")
                for p in (REPO / "Align_adapter/dataset" / suite).glob("episode_*") if p.is_dir()}
    if set(split["train_episodes"]) | set(split["val_episodes"]) != expected:
        raise ValueError("Four-suite manifest does not cover the current dataset")
    train = set(train_cache.read_text().splitlines())
    val = set(val_cache.read_text().splitlines())
    if train & val:
        raise ValueError("Exact train/val RGB overlap")
    protected_rgb = set()
    protected_states = set()
    for line in spatial_audit.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_suite", row.get("suite")) == "libero_spatial":
            protected_states.add(row.get("initial_state_sha256"))
            protected_rgb.update(row[key] for key in ("main_rgb_sha256", "wrist_rgb_sha256") if row.get(key))
    sim_states = {row["state_sha256"] for row in sim["samples"]}
    sim_rgb = {row["rgb_sha256"] for row in sim["samples"]}
    if len(protected_states) != 500 or not protected_rgb:
        raise ValueError("Expected all 500 official Spatial states and their audited RGB hashes")
    if sim.get("protected_state_collisions") or sim_states & protected_states or sim_rgb & protected_rgb:
        raise ValueError("Simulator diagnostic set collided with official Spatial evaluation")
    if train & protected_rgb or train & sim_rgb:
        raise ValueError("Training RGB collides with protected or independent simulator RGB")
    result = {"split_manifest_sha256": sha(split_path), "train_hash_cache_sha256": sha(train_cache),
              "val_hash_cache_sha256": sha(val_cache), "spatial_formal_audit_sha256": sha(spatial_audit),
              "sim_manifest_sha256": sha(args.sim_manifest), "train_trajectories": len(split["train_episodes"]),
              "val_trajectories": len(split["val_episodes"]), "train_unique_rgb_hashes": len(train),
              "val_unique_rgb_hashes": len(val), "train_val_exact_rgb_overlap": 0,
              "train_formal_exact_rgb_overlap": 0, "train_sim_exact_rgb_overlap": 0,
              "sim_formal_initial_state_overlap": 0, "sim_formal_exact_rgb_overlap": 0,
              "formal_state_count": len(protected_states), "sim_state_count": len(sim_states),
              "demo_initial_state_overlap": "unknown: RLDS PNG export lacks complete simulator states"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
