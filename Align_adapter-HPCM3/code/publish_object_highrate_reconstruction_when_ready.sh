#!/usr/bin/env bash
# Publish task 2 as its own commit after the immutable 10x50 run completes.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
output_root=/media/cjs/shared/linux/Align_adapter/hpcm3_object_highrate_reconstruction_full
eval_dir=${output_root}/full_eval_attempt2
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
eval_pid=${1:?Pass the active evaluation PID}

if [[ "$(findmnt -n -o SOURCE --target "${output_root}")" != /dev/sda1 ]]; then
  echo "Object evaluation output is not on the mechanical disk" >&2
  exit 1
fi
exec 9>"${output_root}/publish.lock"
flock -n 9 || { echo "Another publisher is active" >&2; exit 1; }
for minute in $(seq 1 360); do
  if [[ -s "${eval_dir}/summary.json" ]]; then
    break
  fi
  if ! kill -0 "${eval_pid}" 2>/dev/null; then
    echo "Object reconstruction evaluation ended without summary" >&2
    exit 1
  fi
  sleep 60
done
test -s "${eval_dir}/summary.json"
cd "${repo_root}"
"${python_bin}" "${code_root}/analysis/finalize_object_highrate_reconstruction_full.py" \
  --eval-dir "${eval_dir}" >"${output_root}/finalize.log" 2>&1
if [[ "$(git branch --show-current)" != exp/hpcm3-object-dist ]] || \
   [[ -n "$(git diff --cached --name-only)" ]]; then
  echo "Report ready, but branch/staging changed; automatic commit skipped" >&2
  exit 0
fi
git add reports/hpcm3_object_highrate_reconstruction_full_500.md \
  results/hpcm3_object_highrate_reconstruction_full_eval
git diff --cached --check
git commit -m "Report paired full Object high-rate RGB reconstruction"
for attempt in 1 2 3; do
  if git push personal exp/hpcm3-object-dist; then
    echo "Published $(git rev-parse HEAD)" | tee "${output_root}/published.txt"
    exit 0
  fi
  sleep 60
done
echo "Object reconstruction report committed locally but push failed" >&2
exit 1
