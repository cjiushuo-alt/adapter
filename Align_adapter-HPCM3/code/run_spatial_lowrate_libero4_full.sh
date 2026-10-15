#!/usr/bin/env bash
# Single λ=0.0018 Spatial retrain; no change to architecture, loss or data split.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
output_root=/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_lowrate_libero4_full
high_root=/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
config=${code_root}/configs/train_spatial_lowrate_libero4_full.yaml

if [[ "$(findmnt -n -o SOURCE --target "$(dirname "${output_root}")")" != /dev/sda1 ]]; then
  echo "Refusing to train: output is not on the mechanical disk" >&2
  exit 1
fi
if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]]; then
  echo "Wrong branch" >&2
  exit 1
fi
mkdir -p "${output_root}/logs" "${output_root}/wandb"
exec 9>"${output_root}/training.lock"
flock -n 9 || { echo "Another low-rate Spatial training owns the lock" >&2; exit 1; }
test -s "${high_root}/teacher/vision_backbone_partial_layer2.safetensors"
test -s "${high_root}/sim_validation/teacher_pixels/task00_reset00_agent.npy"
test -s "${high_root}/data_protocol_audit.v2.json"
test -s "${high_root}/bitrate_preflight.json"

"${python_bin}" -c 'import copy,json,sys,yaml; lo,hi,high=sys.argv[1:]; a=yaml.safe_load(open(hi)); b=yaml.safe_load(open(lo)); assert a["model"]["hpcm"]["checkpoint"].endswith("0.0483.pth.tar") and b["model"]["hpcm"]["checkpoint"].endswith("0.0018.pth.tar"); a["model"]["hpcm"]["checkpoint"]=b["model"]["hpcm"]["checkpoint"]; a["training"]["resume"]=b["training"]["resume"]; a["training"]["metrics_jsonl"]=b["training"]["metrics_jsonl"]; a["wandb"]=b["wandb"]; a["simulator_validation"]["output_jsonl"]=b["simulator_validation"]["output_jsonl"]; a["output_dir"]=b["output_dir"]; assert a==b, "Unapproved high/low training protocol difference"; audit=json.load(open(high+"/data_protocol_audit.v2.json")); assert audit["formal_state_count"]==500 and audit["train_formal_exact_rgb_overlap"]==0 and audit["train_val_exact_rgb_overlap"]==0; bitrate=json.load(open(high+"/bitrate_preflight.json")); assert {r["lambda"] for r in bitrate["rates"]}=={"0.0018","0.0483"}; assert all(r["entropy_latent_checked_views"]==4 for r in bitrate["rates"])' \
  "${config}" "${code_root}/configs/train_spatial_highrate_libero4_full.yaml" "${high_root}"

export WANDB_DIR="${output_root}/wandb"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/cjs/anaconda3/envs/adapter}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
git -C "${repo_root}" rev-parse HEAD >"${output_root}/source_commit.txt"
if [[ ! -f "${output_root}/preflight_teacher_parity.json" ]]; then
  "${python_bin}" "${code_root}/analysis/verify_object_highrate_protocol.py" \
    --config "${config}" --output "${output_root}/preflight_teacher_parity.json" \
    >"${output_root}/logs/preflight_teacher_parity.log" 2>&1
fi
"${python_bin}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert len(p["samples"])==4 and all(x["hpcm_train_vs_separate_max_abs"]==0 and x["object_teacher_block2_max_abs"]==0 for x in p["samples"])' \
  "${output_root}/preflight_teacher_parity.json"
if [[ -f "${output_root}/training_summary.json" ]]; then
  echo "Low-rate Spatial training already complete" >&2
  exit 0
fi
for attempt in 1 2 3; do
  echo "$(date -Is) training attempt ${attempt}" | tee "${output_root}/status.txt"
  if (cd "${code_root}" && "${python_bin}" train_main.py \
      --config configs/train_spatial_lowrate_libero4_full.yaml --wandb_mode offline) \
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
