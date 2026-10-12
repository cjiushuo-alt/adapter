"""Precompute exact Object-Pro teacher pixels outside the training process.

The training parent must not initialize TensorFlow before DataLoader workers
fork, otherwise TensorFlow's runtime can deadlock all workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for root in (REPO, CODE):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dataset.dataset import object_pro_rollout_pixels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object-pro", type=str, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["source"] != "fresh_env_reset_not_default_pruned_init":
        raise ValueError("Only independently generated simulator validation is allowed")
    output_root = args.manifest.parent / "teacher_pixels"
    output_root.mkdir(exist_ok=False)
    for row in manifest["samples"]:
        with Image.open(args.manifest.parent / row["path"]) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
            raise ValueError(f"RGB hash mismatch: {row['path']}")
        pixels = object_pro_rollout_pixels(rgb, args.object_pro).numpy()
        assert pixels.shape == (6, 224, 224) and pixels.dtype == np.float32
        np.save(output_root / row["path"].replace(".png", ".npy"), pixels)
    print(f"Saved {len(manifest['samples'])} teacher pixel tensors in {output_root}", flush=True)


if __name__ == "__main__":
    main()
