#!/usr/bin/env bash
# Wait for the one authorized low-rate training, then run/publish its 10x50 control.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
train_root=/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_lowrate_libero4_full
eval_dir=${train_root}/eval/lambda_0.0018_hpcm3
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
train_pid=${1:?Pass the active training wrapper PID}

if [[ "$(findmnt -n -o SOURCE --target "${train_root}")" != /dev/sda1 ]]; then
  echo "Training output is not on the mechanical disk" >&2
  exit 1
fi
exec 9>"${train_root}/eval_publish.lock"
flock -n 9 || { echo "Another low-rate eval publisher is active" >&2; exit 1; }

# The wrapper is allowed to retry training. Its completed summary is the gate.
for minute in $(seq 1 4320); do
  if [[ -s "${train_root}/training_summary.json" && -s "${train_root}/adapter-best.pt" ]]; then
    break
  fi
  if ! kill -0 "${train_pid}" 2>/dev/null; then
    echo "Training wrapper exited before publishing a complete checkpoint" >&2
    exit 1
  fi
  sleep 60
done
test -s "${train_root}/training_summary.json"
test -s "${train_root}/adapter-best.pt"
if kill -0 "${train_pid}" 2>/dev/null; then
  while kill -0 "${train_pid}" 2>/dev/null; do sleep 10; done
fi

if [[ ! -e "${eval_dir}" ]]; then
  mkdir -p "$(dirname "${eval_dir}")"
  (cd "${repo_root}" && "${python_bin}" "${code_root}/analysis/eval_spatial_bitrate_full.py" \
    --arm hpcm3 --lambda-value 0.0018 --output-dir "${eval_dir}") \
    >"${train_root}/logs/eval.console.log" 2>&1
fi
test -s "${eval_dir}/summary.json"

cd "${repo_root}"
"${python_bin}" "${code_root}/analysis/finalize_spatial_lowrate_retrain.py" \
  --train-root "${train_root}" --eval-dir "${eval_dir}" \
  >"${train_root}/logs/finalize.log" 2>&1
if [[ "$(git branch --show-current)" != exp/hpcm3-object-dist ]] || \
   [[ -n "$(git diff --cached --name-only)" ]]; then
  echo "Report ready, but branch/staging changed; automatic commit skipped" >&2
  exit 0
fi
git add reports/hpcm3_spatial_lowrate_retrain_500.md results/hpcm3_spatial_lowrate_retrain
git diff --cached --check
git commit -m "Report protocol-matched low-rate Spatial HPCM3 500-episode evaluation"
for attempt in 1 2 3; do
  if git push personal exp/hpcm3-object-dist; then
    echo "Published $(git rev-parse HEAD)" | tee "${train_root}/published.txt"
    exit 0
  fi
  sleep 60
done
echo "Low-rate Spatial result committed locally but push failed" >&2
exit 1
