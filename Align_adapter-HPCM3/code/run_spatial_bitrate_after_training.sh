#!/usr/bin/env bash
# Wait for the single fixed Spatial training, then run 3 immutable 10x50 arms.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
output_root=/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_highrate_libero4_full
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
if [[ "$(findmnt -n -o SOURCE --target "${output_root}")" != /dev/sda1 ]]; then
  echo "Evaluation output is not on the mechanical disk" >&2
  exit 1
fi
if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]]; then
  echo "Wrong branch" >&2
  exit 1
fi
exec 9>"${output_root}/evaluation.lock"
flock -n 9 || { echo "Another Spatial bitrate evaluation owns the lock" >&2; exit 1; }
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/cjs/anaconda3/envs/adapter}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_root}"

# A training crash is not permission to choose another checkpoint or alter loss.
for minute in $(seq 1 2880); do
  if [[ -s "${output_root}/training_summary.json" && -s "${output_root}/adapter-best.pt" ]]; then
    break
  fi
  if [[ -f "${output_root}/status.txt" ]] && \
     grep -q "Training failed three times" "${output_root}/status.txt"; then
    echo "Spatial training failed; refusing evaluation" >&2
    exit 1
  fi
  sleep 60
done
test -s "${output_root}/training_summary.json"
test -s "${output_root}/adapter-best.pt"
mkdir -p "${output_root}/eval"

for specification in "0.0483 reconstruction" "0.0483 hpcm3" "0.0018 reconstruction"; do
  read -r rate arm <<<"${specification}"
  directory="${output_root}/eval/lambda_${rate}_${arm}"
  if [[ -s "${directory}/summary.json" ]]; then
    echo "Reusing complete ${rate}/${arm}" >&2
    continue
  fi
  if [[ -e "${directory}" ]]; then
    echo "Incomplete ${directory}; refusing RNG-inexact resume" >&2
    exit 1
  fi
  echo "$(date -Is) evaluate ${rate}/${arm}" | tee "${output_root}/status.txt"
  "${python_bin}" "${code_root}/analysis/eval_spatial_bitrate_full.py" \
    --arm "${arm}" --lambda-value "${rate}" --output-dir "${directory}" \
    >"${output_root}/logs/eval_${rate}_${arm}.log" 2>&1
done

echo "$(date -Is) finalize Spatial bitrate report" | tee "${output_root}/status.txt"
"${python_bin}" "${code_root}/analysis/finalize_spatial_bitrate.py" --output-root "${output_root}" \
  >"${output_root}/logs/finalize_bitrate.log" 2>&1
if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]] || \
   [[ -n "$(git -C "${repo_root}" diff --cached --name-only)" ]]; then
  echo "Report ready, but branch/staging changed; automatic Git push skipped" | tee "${output_root}/status.txt"
  exit 0
fi
git -C "${repo_root}" add reports/hpcm3_spatial_bitrate.md results/hpcm3_spatial_bitrate
git -C "${repo_root}" add -f results/hpcm3_spatial_bitrate/bitrate_success.png
git -C "${repo_root}" diff --cached --check
git -C "${repo_root}" commit -m "Report Spatial HPCM bitrate-success experiment"
for attempt in 1 2 3; do
  if git -C "${repo_root}" push personal exp/hpcm3-object-dist; then
    echo "$(date -Is) complete and pushed $(git -C "${repo_root}" rev-parse HEAD)" | tee "${output_root}/status.txt"
    exit 0
  fi
  sleep 60
done
echo "Report committed locally but Git push failed" >&2
exit 1
