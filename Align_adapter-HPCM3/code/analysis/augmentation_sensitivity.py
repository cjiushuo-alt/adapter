"""Measure HPCM3 boundary-feature sensitivity to VLA-Adapter image augmentation.

The probe applies the exact dlimp augmentation operators and parameters used by
``prismatic.vla.datasets.RLDSDataset``.  Each augmented RGB image is shared by
the frozen HPCM student path and the frozen vision teacher path, so the
reported distillation error measures representation robustness rather than an
input mismatch between the two branches.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import dlimp as dl
import numpy as np
from PIL import Image
import tensorflow as tf
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
HPCM3_CODE_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, HPCM3_CODE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from model.model import AlignModel  # noqa: E402


VLA_AUGMENTATION = {
    "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
    "random_brightness": [0.2],
    "random_contrast": [0.8, 1.2],
    "random_saturation": [0.8, 1.2],
    "random_hue": [0.05],
}

AUGMENT_ORDERS = {
    "clean": [],
    "crop": ["random_resized_crop"],
    "brightness": ["random_brightness"],
    "contrast": ["random_contrast"],
    "color": [
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
    "full": [
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
}

VIEW_PATTERNS = {
    "image": re.compile(r"^step_\d+_image\.png$"),
    "wrist_image": re.compile(r"^step_\d+_wrist_image\.png$"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--hpcm-root", type=Path, default=REPO_ROOT / "HPCM")
    parser.add_argument(
        "--hpcm-checkpoint", type=Path, default=REPO_ROOT / "HPCM/ckpt/0.0018.pth.tar"
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "Align_adapter/dataset")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial_no_noops", "libero_object_no_noops"],
    )
    parser.add_argument("--samples-per-view", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, path: Path, mode: str) -> tf.Tensor:
    payload = f"{base_seed}\0{path}\0{mode}".encode()
    raw = hashlib.sha256(payload).digest()
    first = int.from_bytes(raw[:4], "little") & 0x7FFFFFFF
    second = int.from_bytes(raw[4:8], "little") & 0x7FFFFFFF
    return tf.constant([first, second], dtype=tf.int32)


def load_rgb_256(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) / 2
    top = (height - side) / 2
    image = image.crop((left, top, left + side, top + side))
    if image.size != (256, 256):
        image = image.resize((256, 256), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def augment_rgb(image: np.ndarray, path: Path, mode: str, base_seed: int) -> np.ndarray:
    order = AUGMENT_ORDERS[mode]
    if not order:
        return image
    kwargs = {"augment_order": order}
    kwargs.update({name: VLA_AUGMENTATION[name] for name in order})
    augmented = dl.transforms.augment_image(
        tf.convert_to_tensor(image), seed=stable_seed(base_seed, path, mode), **kwargs
    )
    return augmented.numpy()


def collect_samples(
    manifest_path: Path,
    dataset_root: Path,
    suites: Sequence[str],
    samples_per_view: int,
    seed: int,
) -> Dict[str, List[Path]]:
    manifest = json.loads(manifest_path.read_text())
    val_episodes = [Path(path) for path in manifest["val_episodes"]]
    selected: Dict[str, List[Path]] = {}
    for suite in suites:
        suite_root = (dataset_root / suite).resolve()
        suite_episodes = [path for path in val_episodes if path.parent.resolve() == suite_root]
        if not suite_episodes:
            raise ValueError(f"manifest 中没有 {suite} validation trajectories")
        paths: List[Path] = []
        for view, pattern in VIEW_PATTERNS.items():
            candidates = sorted(
                path
                for episode in suite_episodes
                for path in episode.glob("*.png")
                if pattern.match(path.name)
            )
            if len(candidates) < samples_per_view:
                raise ValueError(
                    f"{suite}/{view} 只有 {len(candidates)} 张 validation 图，"
                    f"无法抽取 {samples_per_view} 张"
                )
            view_seed = int.from_bytes(hashlib.sha256(f"{seed}:{suite}:{view}".encode()).digest()[:8], "little")
            paths.extend(random.Random(view_seed).sample(candidates, samples_per_view))
        selected[suite] = sorted(paths)
    return selected


def load_adapter(model: AlignModel, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("adapter.") for key in state):
        state = {key.removeprefix("adapter."): value for key, value in state.items() if key.startswith("adapter.")}
    model.adapter.load_state_dict(state, strict=True)


def build_model(args: argparse.Namespace) -> AlignModel:
    model = AlignModel(
        hpcm_root=str(args.hpcm_root.resolve()),
        hpcm_checkpoint=str(args.hpcm_checkpoint.resolve()),
        hpcm_model_name="HPCM_Base",
        hpcm_scale_table_levels=60,
        freeze_hpcm_encoder=True,
        vision_backbone_id="dinosiglip-vit-so-224px",
        vision_backbone_checkpoint=args.teacher_checkpoint.resolve(),
        vision_backbone_max_layer=2,
        image_resize_strategy="letterbox",
        image_sequence_len=1,
        adapter_config={
            "hpcm_input_dim": 320,
            "pre_neck_output_dim": 320,
            "pre_neck_spatial_size": 16,
            "pre_neck_num_res_blocks": 2,
            "pre_neck_dropout": 0.0,
            "siglip_dim": 1152,
            "dino_dim": 1024,
            "neck_num_heads": 8,
            "neck_ffn_hidden_dim": None,
            "neck_dropout": 0.0,
            "neck_activation": "gelu",
        },
        freeze_vision_backbone=True,
        train_adapter=False,
        loss_config={
            "type": "dist",
            "siglip_loss_weight": 1.0,
            "dino_loss_weight": 5.0,
            "reduction": "mean",
        },
    )
    load_adapter(model, args.adapter_checkpoint)
    return model.to(torch.device(args.device)).eval()


def batched(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def summarize(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate(
    model: AlignModel,
    selected: Dict[str, List[Path]],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    aggregate: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    with torch.inference_mode():
        for suite, paths in selected.items():
            for mode in AUGMENT_ORDERS:
                for path_batch in batched(paths, batch_size):
                    images = [augment_rgb(load_rgb_256(path), path, mode, seed) for path in path_batch]
                    tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
                    tensor = tensor.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
                    result = model(tensor)
                    siglip = (result["aligned_siglip_features"].float() - result["vision_siglip_features"].float())
                    dino = (result["aligned_dino_features"].float() - result["vision_dino_features"].float())
                    siglip_mse = siglip.square().flatten(1).mean(1)
                    dino_mse = dino.square().flatten(1).mean(1)
                    total = siglip_mse + 5.0 * dino_mse
                    for metric, batch_values in (
                        ("siglip_mse", siglip_mse),
                        ("dino_mse", dino_mse),
                        ("weighted_loss", total),
                    ):
                        aggregate[suite][mode][metric].extend(batch_values.cpu().tolist())

    output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for suite, modes in aggregate.items():
        output[suite] = {}
        clean_mean = np.mean(modes["clean"]["weighted_loss"])
        for mode, metrics in modes.items():
            output[suite][mode] = {metric: summarize(values) for metric, values in metrics.items()}
            output[suite][mode]["weighted_loss_ratio_to_clean"] = float(
                np.mean(metrics["weighted_loss"]) / clean_mean
            )
    return output


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        # dlimp/TensorFlow is used only for augmentation; reserve the GPU for PyTorch.
        tf.config.set_visible_devices([], "GPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    selected = collect_samples(
        args.split_manifest,
        args.dataset_root,
        args.suites,
        args.samples_per_view,
        args.seed,
    )
    model = build_model(args)
    results = evaluate(model, selected, args.batch_size, args.seed, torch.device(args.device))

    payload = {
        "schema_version": 1,
        "label": args.label,
        "seed": args.seed,
        "samples_per_view": args.samples_per_view,
        "views": list(VIEW_PATTERNS),
        "sample_count_per_suite": {suite: len(paths) for suite, paths in selected.items()},
        "checkpoints": {
            "adapter": {"path": str(args.adapter_checkpoint.resolve()), "sha256": sha256_file(args.adapter_checkpoint)},
            "teacher": {"path": str(args.teacher_checkpoint.resolve()), "sha256": sha256_file(args.teacher_checkpoint)},
            "hpcm": {"path": str(args.hpcm_checkpoint.resolve()), "sha256": sha256_file(args.hpcm_checkpoint)},
        },
        "manifest": {"path": str(args.split_manifest.resolve()), "sha256": sha256_file(args.split_manifest)},
        "augmentation": {
            "implementation": "dlimp.transforms.augment_image",
            "parameters": VLA_AUGMENTATION,
            "modes": AUGMENT_ORDERS,
            "input_resolution": [256, 256],
            "note": "One augmented RGB tensor is shared by HPCM student and vision teacher.",
        },
        "selected_samples": {
            suite: [str(path.resolve().relative_to(args.dataset_root.resolve())) for path in paths]
            for suite, paths in selected.items()
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"label": args.label, "output": str(args.output), "results": results}, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
