#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
output_root=/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_boundary
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python

if [[ "$(findmnt -n -o SOURCE --target "${output_root}")" != /dev/sda1 ]]; then
  echo "Refusing to train: output is not on the mechanical disk" >&2
  exit 1
fi
test -f "${output_root}/preflight_parity.json"
test -f "${output_root}/data_split.json"
test -f "${output_root}/sim_validation/manifest.json"
mkdir -p "${output_root}/wandb" "${output_root}/logs"
export WANDB_DIR="${output_root}/wandb"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/cjs/anaconda3/envs/adapter}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${code_root}"
exec "${python_bin}" train_main.py \
  --config configs/train_object_highrate_boundary.yaml \
  --wandb_mode online
