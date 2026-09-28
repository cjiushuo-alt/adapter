#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/cjs/study/VLA/VLA-Adapter"
code_root="${repo_root}/Align_adapter-HPCM3/code"
output_root="/media/cjs/shared/linux/Align_adapter/hpcm3_object_fullvisual_aug_libero4"
python_bin="/home/cjs/anaconda3/envs/align-adapter/bin/python"

if [[ "$(findmnt -n -o SOURCE --target /media/cjs/shared/linux)" != "/dev/sda1" ]]; then
  echo "Refusing to train: /media/cjs/shared/linux is not backed by /dev/sda1" >&2
  exit 1
fi

mkdir -p "${output_root}/logs" "${output_root}/wandb"
export WANDB_DIR="${output_root}/wandb"
export PYTHONUNBUFFERED=1

cd "${code_root}"
exec "${python_bin}" train_main.py \
  --config configs/train_object_fullvisual_aug_libero4.yaml \
  --wandb_mode online
