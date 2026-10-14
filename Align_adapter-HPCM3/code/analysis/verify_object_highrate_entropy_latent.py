"""Compare the HPCM3 inference latent with actual entropy-decoded y_hat.

Uses only the fixed, independent fresh-reset simulator images. No formal
initial states, policy inference, model mutation, or training are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from dataset.hpcm_encoder import HPCMEncoder  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def compare(encoder: HPCMEncoder, rgb: np.ndarray) -> dict:
    # Exactly the rollout bypass conversion: CPU uint8 -> FP32 [-1,1], then
    # transfer to CUDA; CombinedVisionModel maps this to [0,1] on CUDA.
    raw = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(127.5).sub_(1.0).cuda()
    image01 = raw.add(1.0).mul(0.5)
    forward_latent = encoder(image01)
    codec = encoder._hpcm_model
    compressed = codec.compress(image01)
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().clone())

    handle = codec.g_s.register_forward_hook(hook)
    try:
        _ = codec.decompress(compressed["strings"], compressed["shape"])
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != forward_latent.shape:
        raise RuntimeError("Could not capture entropy-decoded decoder input")
    difference = (captured[0] - forward_latent).float().abs()
    return {
        "latent_shape": list(forward_latent.shape),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "nonzero_elements": int(torch.count_nonzero(difference)),
        "latent_elements": forward_latent.numel(),
        "y_bytes": len(compressed["strings"][0]),
        "z_bytes": len(compressed["strings"][1]),
        "bpp_actual": 8.0 * sum(map(len, compressed["strings"])) / (256 * 256),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.manifest = args.manifest.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protected_state_collisions") != 0 or len(manifest["samples"]) != 40:
        raise ValueError("Expected prior independent 40-image simulator manifest")
    samples = [row for row in manifest["samples"] if row["task_id"] in (0, 1)]
    if len(samples) != 8:
        raise ValueError("Expected two tasks x two resets x two views")
    encoder = HPCMEncoder(
        hpcm_root=str(REPO / "HPCM"), checkpoint_path=str(args.checkpoint),
        model_name="HPCM_Base", scale_table_levels=60, pad_multiple=256,
    ).cuda().eval()
    rows = []
    for sample in samples:
        with Image.open(args.manifest.parent / sample["path"]) as image:
            rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        if rgb.shape != (256, 256, 3):
            raise ValueError(sample["path"])
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if rgb_sha != sample["rgb_sha256"]:
            raise ValueError("Simulator RGB changed")
        row = {"task_id": sample["task_id"], "reset_index": sample["reset_index"],
               "view": sample["view"], "rgb_sha256": rgb_sha, **compare(encoder, rgb)}
        rows.append(row)
        print(row, flush=True)
    result = {
        "checkpoint_sha256": sha(args.checkpoint),
        "manifest_sha256": sha(args.manifest),
        "source": "independent fresh-reset simulator images, not formal evaluation states",
        "samples": rows,
        "max_abs": max(row["max_abs"] for row in rows),
        "mean_abs": sum(row["mean_abs"] for row in rows) / len(rows),
        "exactly_equal_samples": sum(row["nonzero_elements"] == 0 for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
