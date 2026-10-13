"""Export small, reproducible per-epoch curves from local training records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+) (?P<split>训练|验证): Total=(?P<total>[\d.]+), "
    r"SigLIP=(?P<siglip>[\d.]+), DINO=(?P<dino>[\d.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--resume-console-log", type=Path)
    parser.add_argument("--resume-from-epoch", type=int)
    parser.add_argument("--sim-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = {}
    if bool(args.resume_console_log) != (args.resume_from_epoch is not None):
        raise ValueError("Provide both --resume-console-log and --resume-from-epoch")
    sources = [(args.console_log, 0)]
    if args.resume_console_log:
        sources.append((args.resume_console_log, args.resume_from_epoch))
    for source, first_epoch in sources:
        for match in PATTERN.finditer(source.read_text(errors="replace")):
            epoch = int(match["epoch"])
            if epoch < first_epoch:
                continue
            split = "train" if match["split"] == "训练" else "clean_val"
            # Lightning's short sanity validation logs the preceding epoch;
            # exclude it from the resumed source via first_epoch.
            metrics.setdefault(epoch, {})[split] = {
                "total": float(match["total"]), "siglip_mse": float(match["siglip"]),
                "dino_mse": float(match["dino"]),
            }
    for line in args.sim_metrics.read_text().splitlines():
        row = json.loads(line)
        epoch = int(row["epoch"])
        if "simulator" in metrics.setdefault(epoch, {}):
            raise ValueError(f"Duplicate simulator epoch {epoch}")
        metrics[epoch]["simulator"] = row["views"]
        metrics[epoch]["global_step"] = row["global_step"]
    if not metrics or sorted(metrics) != list(range(max(metrics) + 1)):
        raise ValueError("Missing or non-contiguous epoch metrics")
    for epoch, row in metrics.items():
        if not {"train", "clean_val", "simulator", "global_step"} <= row.keys():
            raise ValueError(f"Incomplete epoch {epoch}: {row.keys()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x") as stream:
        for epoch in sorted(metrics):
            stream.write(json.dumps({"epoch": epoch, **metrics[epoch]}) + "\n")
    print(f"Exported {len(metrics)} complete epochs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
