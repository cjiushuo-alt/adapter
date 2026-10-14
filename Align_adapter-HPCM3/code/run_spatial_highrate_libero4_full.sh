#!/usr/bin/env bash
# Single fixed-rate Spatial Adapter training; all large states stay on HDD.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
output_root=/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
config=${code_root}/configs/train_spatial_highrate_libero4_full.yaml

if [[ "$(findmnt -n -o SOURCE --target "${output_root}")" != /dev/sda1 ]]; then
  echo "Refusing to train: output is not mounted on the mechanical disk" >&2
  exit 1
fi
if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]]; then
  echo "Wrong branch" >&2
  exit 1
fi
mkdir -p "${output_root}/logs" "${output_root}/wandb"
exec 9>"${output_root}/training.lock"
flock -n 9 || { echo "Spatial training already running" >&2; exit 1; }
test -s "${output_root}/teacher/vision_backbone_partial_layer2.safetensors"
test -s "${output_root}/preflight_teacher_parity.json"
test -s "${output_root}/bitrate_preflight.json"
test -s "${output_root}/data_protocol_audit.v2.json"
test -s "${output_root}/sim_validation/teacher_pixels/task00_reset00_agent.npy"
"${python_bin}" -c 'import json,sys; root=sys.argv[1]; a=json.load(open(root+"/data_protocol_audit.v2.json")); p=json.load(open(root+"/preflight_teacher_parity.json")); b=json.load(open(root+"/bitrate_preflight.json")); assert a["formal_state_count"]==500 and a["train_formal_exact_rgb_overlap"]==a["train_val_exact_rgb_overlap"]==a["sim_formal_initial_state_overlap"]==0; assert len(p["samples"])==4 and all(x["object_teacher_block2_max_abs"]==0 and x["hpcm_train_vs_separate_max_abs"]==0 for x in p["samples"]); assert len(b["rates"])==2 and all(x["entropy_latent_checked_views"]==4 for x in b["rates"])' "${output_root}"

git -C "${repo_root}" rev-parse HEAD >"${output_root}/source_commit.txt"
export WANDB_DIR="${output_root}/wandb"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/cjs/anaconda3/envs/adapter}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -f "${output_root}/training_summary.json" ]]; then
  echo "Training already complete" >&2
  exit 0
fi
for attempt in 1 2 3; do
  echo "$(date -Is) training attempt ${attempt}" | tee "${output_root}/status.txt"
  if (cd "${code_root}" && "${python_bin}" train_main.py \
      --config configs/train_spatial_highrate_libero4_full.yaml --wandb_mode offline) \
      >"${output_root}/logs/train.attempt${attempt}.log" 2>&1; then
    break
  fi
  if [[ -f "${output_root}/training_summary.json" ]]; then
    break
  fi
  if [[ "${attempt}" -eq 3 ]]; then
    echo "Training failed three times; see HDD logs" | tee "${output_root}/status.txt"
    exit 1
  fi
  sleep 60
done
test -s "${output_root}/training_summary.json"
test -s "${output_root}/adapter-best.pt"
sha256sum "${output_root}/adapter-best.pt" >"${output_root}/adapter-best.sha256"
echo "$(date -Is) training complete" | tee "${output_root}/status.txt"
