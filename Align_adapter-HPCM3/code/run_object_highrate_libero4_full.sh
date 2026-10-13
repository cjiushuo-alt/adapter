#!/usr/bin/env bash
# Detached, retryable single-experiment workflow. All large files stay on HDD.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
output_root=/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_libero4_full
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
train_config=${code_root}/configs/train_object_highrate_libero4_full.yaml
eval_config=${code_root}/configs/eval_object_highrate_libero4_full_task0_ep5.yaml

if [[ "$(findmnt -n -o SOURCE --target "${output_root}")" != /dev/sda1 ]]; then
  echo "Refusing to run: output is not mounted on /dev/sda1" >&2
  exit 1
fi
mkdir -p "${output_root}/logs" "${output_root}/wandb"
exec 9>"${output_root}/workflow.lock"
if ! flock -n 9; then
  echo "Another high-rate four-suite workflow owns the lock" >&2
  exit 1
fi
if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]]; then
  echo "Wrong Git branch; refusing to start" >&2
  exit 1
fi
git -C "${repo_root}" rev-parse HEAD >"${output_root}/source_commit.txt"
export WANDB_DIR="${output_root}/wandb"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/cjs/anaconda3/envs/adapter}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_root}"

if [[ ! -f "${output_root}/data_audit/summary.json" ]]; then
  for attempt in 1 2 3; do
    audit_attempt="${output_root}/data_audit_attempt${attempt}"
    if [[ -e "${audit_attempt}" ]]; then
      continue
    fi
    echo "$(date -Is) audit attempt ${attempt}" | tee "${output_root}/status.txt"
    if "${python_bin}" "${code_root}/analysis/audit_object_highrate_libero4_full.py" \
      --config "${train_config}" --output-dir "${audit_attempt}" --workers 8 \
      >"${output_root}/logs/data_audit_attempt${attempt}.log" 2>&1; then
      mv "${audit_attempt}" "${output_root}/data_audit"
      break
    fi
    if [[ -f "${audit_attempt}/summary.json" ]] && \
       ! "${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["train_protected_exact_rgb_hash_overlap"]==0 and d["train_val_exact_rgb_hash_overlap"]==0' \
          "${audit_attempt}/summary.json"; then
      echo "Leakage audit detected an overlap; training stopped" | tee "${output_root}/status.txt"
      exit 1
    fi
  done
fi
test -f "${output_root}/data_audit/summary.json"
"${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["train_protected_exact_rgb_hash_overlap"]==0 and d["train_val_exact_rgb_hash_overlap"]==0' \
  "${output_root}/data_audit/summary.json"

if [[ ! -f "${output_root}/preflight_parity.json" ]]; then
  echo "$(date -Is) preprocessing/teacher parity" | tee "${output_root}/status.txt"
  "${python_bin}" "${code_root}/analysis/verify_object_highrate_protocol.py" \
    --config "${train_config}" --output "${output_root}/preflight_parity.json" \
    >"${output_root}/logs/preflight_parity.log" 2>&1
fi

if [[ ! -f "${output_root}/training_summary.json" ]]; then
  for attempt in 1 2 3; do
    echo "$(date -Is) training attempt ${attempt}" | tee "${output_root}/status.txt"
    if (cd "${code_root}" && "${python_bin}" train_main.py \
      --config configs/train_object_highrate_libero4_full.yaml --wandb_mode offline) \
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
fi
test -f "${output_root}/training_summary.json"
test -f "${output_root}/adapter-best.pt"

eval_dir=
for attempt in 1 2 3; do
  candidate="${output_root}/eval_task0_ep5_attempt${attempt}"
  if [[ -f "${candidate}/summary.json" ]]; then
    eval_dir=${candidate}
    break
  fi
  if [[ -e "${candidate}" ]]; then
    continue
  fi
  echo "$(date -Is) protected task-0 evaluation attempt ${attempt}" | tee "${output_root}/status.txt"
  if "${python_bin}" "${code_root}/analysis/eval_object_highrate_boundary_task0.py" \
    --config "${eval_config}" --output-dir "${candidate}" \
    >"${output_root}/logs/eval.attempt${attempt}.log" 2>&1; then
    eval_dir=${candidate}
    break
  fi
done
if [[ -z "${eval_dir}" ]]; then
  echo "Evaluation failed three times; see HDD logs" | tee "${output_root}/status.txt"
  exit 1
fi

echo "$(date -Is) finalizing report" | tee "${output_root}/status.txt"
"${python_bin}" "${code_root}/analysis/finalize_object_highrate_libero4_full.py" \
  --config "${train_config}" --output-root "${output_root}" --eval-dir "${eval_dir}" \
  >"${output_root}/logs/finalize.log" 2>&1

# Do not accidentally include unrelated worktree edits or any large weights.
if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]]; then
  echo "Report is ready, but branch changed; automatic Git push skipped" | tee "${output_root}/status.txt"
  exit 0
fi
if [[ -n "$(git -C "${repo_root}" diff --cached --name-only)" ]]; then
  echo "Report is ready, but unrelated staged files exist; automatic Git push skipped" | tee "${output_root}/status.txt"
  exit 0
fi
git -C "${repo_root}" add reports/hpcm3_object_highrate_libero4_full.md results/hpcm3_object_highrate_libero4_full
git -C "${repo_root}" diff --cached --check
git -C "${repo_root}" commit -m "Report full four-suite high-rate Object HPCM3 experiment"
for attempt in 1 2 3; do
  if git -C "${repo_root}" push personal exp/hpcm3-object-dist; then
    echo "$(date -Is) complete and pushed $(git -C "${repo_root}" rev-parse HEAD)" | tee "${output_root}/status.txt"
    exit 0
  fi
  sleep 60
done
echo "Report committed locally but Git push failed three times" | tee "${output_root}/status.txt"
exit 1
