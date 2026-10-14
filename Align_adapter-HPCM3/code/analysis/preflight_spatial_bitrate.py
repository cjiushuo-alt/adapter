"""Independent Spatial RGB codec/rate preflight; no official rollout states."""

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

from analysis.object_hpcm_reconstruction_control import ReconstructionControl, sha256  # noqa: E402
from analysis.verify_object_highrate_entropy_latent import compare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("source") != "fresh_env_reset_not_default_pruned_init" or \
            manifest.get("protected_state_collisions") != 0:
        raise ValueError("Non-independent simulator manifest")
    if len(manifest["samples"]) != 40 or any(row["suite"] != "libero_spatial" for row in manifest["samples"]):
        raise ValueError("Expected 40 independent Spatial views")
    rates = []
    for lam in ("0.0018", "0.0483"):
        checkpoint = REPO / "HPCM/ckpt" / f"{lam}.pth.tar"
        codec = ReconstructionControl(checkpoint, REPO / "HPCM", 60)
        rows = []
        for index, meta in enumerate(manifest["samples"]):
            with Image.open(args.manifest.parent / meta["path"]) as image:
                rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != meta["rgb_sha256"]:
                raise ValueError("Independent sample RGB hash changed")
            _, metrics = codec.reconstruct(rgb)
            parity = compare(codec.encoder, rgb) if index < 4 else None
            if parity is not None and parity["nonzero_elements"]:
                raise RuntimeError(f"Entropy latent mismatch for {lam} sample {index}")
            rows.append({"task_id": meta["task_id"], "view": meta["view"],
                         "rgb_sha256": meta["rgb_sha256"], "bpp_actual": metrics["bpp_actual"],
                         "psnr_db": metrics["psnr_db"], "entropy_latent_max_abs":
                         None if parity is None else parity["max_abs"]})
        rates.append({"lambda": lam, "checkpoint_sha256": sha256(checkpoint),
                      "samples": rows, "mean": {
                          view: {key: float(np.mean([r[key] for r in rows if r["view"] == view]))
                                 for key in ("bpp_actual", "psnr_db")}
                          for view in ("agent", "wrist")},
                      "entropy_latent_checked_views": 4})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_manifest_sha256": sha256(args.manifest),
                                       "source": "independent Spatial fresh-reset, non-official states",
                                       "rates": rates}, indent=2) + "\n")
    print(json.dumps([{"lambda": r["lambda"], "mean": r["mean"]} for r in rates], indent=2))


if __name__ == "__main__":
    main()
