#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/cjs/study/VLA/VLA-Adapter"
code_root="${repo_root}/Align_adapter-HPCM3/code"
output_root="/media/cjs/shared/linux/Align_adapter/hpcm3_object_fullvisual_aug_libero4"
checkpoint_dir="${output_root}/checkpoints"
adapter_path="${output_root}/eval/adapter-epoch10.pt"
result_dir="${repo_root}/results/hpcm3_object_fullvisual_aug_epoch10/object"
config_path="${code_root}/configs/eval_object_fullvisual_aug_epoch10.yaml"
python_bin="/home/cjs/anaconda3/envs/align-adapter/bin/python"
training_unit="hpcm3-object-fullvisual-aug-resume.service"

if [[ "$(findmnt -n -o SOURCE --target /media/cjs/shared/linux)" != "/dev/sda1" ]]; then
  echo "Mechanical disk is not mounted at /media/cjs/shared/linux" >&2
  exit 1
fi
if [[ -e "${result_dir}/run.log" || -e "${result_dir}/episode_results.jsonl" ]]; then
  echo "Refusing to overwrite an existing rollout: ${result_dir}" >&2
  exit 1
fi

last_checkpoint="${checkpoint_dir}/last.ckpt"
initial_last_mtime="$(stat -c %Y "${last_checkpoint}")"
echo "[$(date --iso-8601=seconds)] waiting for completed Epoch 10 checkpoint"
while true; do
  epoch_checkpoint="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'epoch-epoch=10-*.ckpt' -print -quit)"
  if [[ -n "${epoch_checkpoint}" ]]; then
    checkpoint_path="${epoch_checkpoint}"
    break
  fi
  if [[ "$(stat -c %Y "${last_checkpoint}")" -gt "${initial_last_mtime}" ]]; then
    checkpoint_path="${last_checkpoint}"
    break
  fi
  if ! systemctl --user is-active --quiet "${training_unit}"; then
    echo "Training ended before Epoch 10 checkpoint appeared" >&2
    exit 1
  fi
  sleep 60
done

# Wait until Lightning's large checkpoint has stopped changing before reading it.
while true; do
  before="$(stat -c '%s:%Y' "${checkpoint_path}")"
  sleep 30
  after="$(stat -c '%s:%Y' "${checkpoint_path}")"
  [[ "${before}" == "${after}" ]] && break
done
"${python_bin}" -c 'import sys, torch; p=sys.argv[1]; epoch=torch.load(p, map_location="cpu", weights_only=False)["epoch"]; print(f"Checkpoint epoch: {epoch}"); assert epoch == 10, f"Expected epoch 10, got {epoch}"' "${checkpoint_path}"
echo "[$(date --iso-8601=seconds)] extracting ${checkpoint_path}"
mkdir -p "$(dirname "${adapter_path}")" "${result_dir}"
"${python_bin}" "${code_root}/inference/utils/extract_adapter_weights.py" \
  --input_ckpt "${checkpoint_path}" --output_path "${adapter_path}"
cp -n "${config_path}" "${result_dir}/config.yaml"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
echo "[$(date --iso-8601=seconds)] starting Object rollout"
"${python_bin}" "${code_root}/inference/run_libero_eval.py" \
  --config_path "${result_dir}/config.yaml"
echo "[$(date --iso-8601=seconds)] rollout finished: ${result_dir}/summary.json"
