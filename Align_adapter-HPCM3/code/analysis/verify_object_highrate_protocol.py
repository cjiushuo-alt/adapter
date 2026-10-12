"""Preflight numeric parity on held-out RLDS images, without eval states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
for root in (REPO, CODE, REPO / "LIBERO"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dataset.dataset import object_pro_rollout_pixels  # noqa: E402
from dataset.hpcm_encoder import HPCMEncoder  # noqa: E402
from model.model import AlignModel  # noqa: E402
from analysis.fullvisual_offline_diagnostic import build_hf_tower  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    m = cfg["model"]
    assert cfg["loss"]["scope"] == "boundary" and m["vision"]["max_layer"] == 2
    assert m["hpcm"]["freeze"] and m["vision"]["freeze_backbone"]
    manifest = json.loads(Path(cfg["data"]["split_manifest_path"]).read_text())
    assert not (set(manifest["train_episodes"]) & set(manifest["val_episodes"]))
    paths = []
    for episode in manifest["val_episodes"][:2]:
        selected = sorted(Path(episode).glob("step_*_image.png"))
        assert selected
        paths.append(selected[0])
    model = AlignModel(
        hpcm_root=m["hpcm"]["hpcm_root"], hpcm_checkpoint=m["hpcm"]["checkpoint"],
        hpcm_model_name=m["hpcm"]["model_name"],
        hpcm_scale_table_levels=m["hpcm"]["scale_table_levels"],
        freeze_hpcm_encoder=True, vision_backbone_id=m["vision"]["backbone_id"],
        vision_backbone_checkpoint=Path(m["vision"]["checkpoint"]),
        vision_backbone_max_layer=2, image_resize_strategy="letterbox",
        image_sequence_len=1,
        adapter_config={k: v for k, v in m["adapter"].items() if k != "train_adapter"},
        freeze_vision_backbone=True, train_adapter=False,
        loss_config=cfg["loss"], augmentation_config={"enabled": False},
    ).cuda().eval()
    separate_codec = HPCMEncoder(
        hpcm_root=m["hpcm"]["hpcm_root"], checkpoint_path=m["hpcm"]["checkpoint"],
        model_name="HPCM_Base", scale_table_levels=60, pad_multiple=256,
    ).cuda().eval()
    hf, _ = build_hf_tower(Path(cfg["data"]["teacher_rollout_checkpoint"]), torch.device("cuda"))
    rows = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
            assert rgb.shape == (256, 256, 3)
            raw = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda().float().div_(127.5).sub_(1.0)
            teacher = object_pro_rollout_pixels(rgb, cfg["data"]["teacher_rollout_checkpoint"]).unsqueeze(0).cuda()
            a = model.extract_hpcm_features(raw)
            b = separate_codec(raw.add(1.0).mul(0.5))
            target = model.vision_backbone({"dino": teacher[:, :3], "siglip": teacher[:, 3:]})
            reference_dino = hf.featurizer.get_intermediate_layers(teacher[:, :3], n={2})[0]
            reference_siglip = hf.fused_featurizer.get_intermediate_layers(teacher[:, 3:], n={2})[0]
            reference = torch.cat([reference_dino, reference_siglip], dim=-1)
            row = {
                "path": str(path), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "teacher_pixels_shape": list(teacher.shape),
                "latent_shape": list(a.shape), "target_shape": list(target.shape),
                "hpcm_train_vs_separate_max_abs": float((a - b).abs().max()),
                "object_teacher_block2_max_abs": float((target - reference).abs().max()),
                "object_teacher_block2_mae": float((target - reference).abs().mean()),
            }
            if row["hpcm_train_vs_separate_max_abs"] > 1e-5 or row["object_teacher_block2_max_abs"] > 1e-4:
                raise RuntimeError(f"Preflight parity failed: {row}")
            rows.append(row)
            print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "hpcm_sha256": sha(Path(m["hpcm"]["checkpoint"])),
        "partial_teacher_sha256": sha(Path(m["vision"]["checkpoint"])),
        "object_pro_sha256": sha(Path(cfg["data"]["teacher_rollout_checkpoint"]) / "model.safetensors"),
        "split_manifest_sha256": sha(Path(cfg["data"]["split_manifest_path"])),
        "samples": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
