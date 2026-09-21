"""Build the HPCM3 train-split decoded-RGB SHA256 cache once for all suites."""

import argparse
import hashlib
import multiprocessing as mp
import os
import random
import json
from pathlib import Path

from PIL import Image


def hash_decoded_rgb(path: str) -> str:
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="trajectory split manifest；提供时只 hash train_episodes 中的两个相机",
    )
    args = parser.parse_args()

    if args.split_manifest is not None:
        with args.split_manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("split_unit") != "trajectory":
            raise ValueError(f"Not a trajectory split manifest: {args.split_manifest}")
        paths = []
        for episode in manifest.get("train_episodes", []):
            episode_path = Path(episode)
            paths.extend(episode_path.glob("step_*_image.png"))
        paths = sorted(set(paths))
    else:
        paths = sorted(args.dataset_dir.glob("libero_*_no_noops/episode_*/step_*_image.png"))
    if not paths:
        raise FileNotFoundError(f"No training frames found under {args.dataset_dir}")
    if args.split_manifest is not None:
        train_paths = [str(path) for path in paths]
    else:
        indices = list(range(len(paths)))
        random.Random(args.seed).shuffle(indices)
        split_idx = int(len(indices) * args.train_ratio)
        train_paths = [str(paths[index]) for index in indices[:split_idx]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    hashes = set()
    print(f"Hashing {len(train_paths)} decoded RGB frames with {args.workers} workers", flush=True)
    with mp.Pool(processes=args.workers) as pool:
        for index, value in enumerate(pool.imap(hash_decoded_rgb, train_paths, chunksize=128), 1):
            hashes.add(value)
            if index % 50000 == 0:
                print(f"Hashed {index}/{len(train_paths)}", flush=True)

    with temporary.open("w", encoding="ascii") as handle:
        for value in sorted(hashes):
            handle.write(value + "\n")
    os.replace(temporary, args.output)
    print(f"Wrote {len(hashes)} unique hashes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
