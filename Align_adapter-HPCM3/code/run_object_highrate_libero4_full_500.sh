#!/usr/bin/env bash
# One frozen 10x50 evaluation. A failed run is preserved, never silently
# resumed with a different RNG stream or counted as policy failures.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
output_root=/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_libero4_full/full_eval
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
config=${code_root}/configs/eval_object_highrate_libero4_full_500.yaml

if [[ "$(findmnt -n -o SOURCE --target "$(dirname "${output_root}")")" != /dev/sda1 ]]; then
  echo "Refusing to evaluate: output is not on the mechanical disk" >&2
  exit 1
fi
if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]]; then
  echo "Wrong branch" >&2
  exit 1
fi
mkdir -p "$(dirname "${output_root}")"
exec 9>"$(dirname "${output_root}")/full_eval.lock"
flock -n 9 || { echo "Another full Object evaluation is running" >&2; exit 1; }
if [[ -e "${output_root}" ]]; then
  echo "Refusing to overwrite existing full evaluation: ${output_root}" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/cjs/anaconda3/envs/adapter}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_root}"

"${python_bin}" "${code_root}/analysis/eval_object_highrate_libero4_full_500.py" \
  --config "${config}" --output-dir "${output_root}" \
  >"$(dirname "${output_root}")/full_eval.console.log" 2>&1
"${python_bin}" "${code_root}/analysis/finalize_object_highrate_libero4_full_500.py" \
  --eval-dir "${output_root}" \
  >"$(dirname "${output_root}")/full_eval.finalize.log" 2>&1

if [[ "$(git -C "${repo_root}" branch --show-current)" != exp/hpcm3-object-dist ]] || \
   [[ -n "$(git -C "${repo_root}" diff --cached --name-only)" ]]; then
  echo "Evaluation complete, but Git branch/staging changed; automatic commit skipped" >&2
  exit 0
fi
git -C "${repo_root}" add reports/hpcm3_object_highrate_libero4_full_500.md \
  results/hpcm3_object_highrate_libero4_full_eval
git -C "${repo_root}" diff --cached --check
git -C "${repo_root}" commit -m "Report full 500-episode high-rate Object HPCM3 evaluation"
for attempt in 1 2 3; do
  if git -C "${repo_root}" push personal exp/hpcm3-object-dist; then
    echo "complete and pushed $(git -C "${repo_root}" rev-parse HEAD)"
    exit 0
  fi
  sleep 60
done
echo "Evaluation/report complete but Git push failed" >&2
exit 1
