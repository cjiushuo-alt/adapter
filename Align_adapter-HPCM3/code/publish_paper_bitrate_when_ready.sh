#!/usr/bin/env bash
# Assemble the audited paper artifacts only after the Spatial 0.0018 stage is pushed.
set -euo pipefail

repo_root=/home/cjs/study/VLA/VLA-Adapter
code_root=${repo_root}/Align_adapter-HPCM3/code
train_root=/media/cjs/shared/linux/Align_adapter/hpcm3_spatial_lowrate_libero4_full
python_bin=/home/cjs/anaconda3/envs/adapter/bin/python
stage_pid=${1:?Pass the active Spatial evaluation publisher PID}

exec 9>"${train_root}/paper_publish.lock"
flock -n 9 || { echo "Another paper publisher is active" >&2; exit 1; }
for minute in $(seq 1 4320); do
  if [[ -s "${train_root}/published.txt" ]]; then
    break
  fi
  if ! kill -0 "${stage_pid}" 2>/dev/null; then
    echo "Spatial stage ended without publishing the verified 500-episode result" >&2
    exit 1
  fi
  sleep 60
done
test -s "${train_root}/published.txt"
cd "${repo_root}"
if [[ "$(git branch --show-current)" != exp/hpcm3-object-dist ]] || \
   [[ -n "$(git diff --cached --name-only)" ]]; then
  echo "Branch/staging changed; paper build skipped" >&2
  exit 1
fi
"${python_bin}" "${code_root}/analysis/build_paper_bitrate_success.py" \
  >"${train_root}/logs/paper_build.log" 2>&1
git add reports/hpcm3_paper_bitrate_success.md results/hpcm3_paper_bitrate_success
# Repository-wide *.png ignore also catches the requested paper figure.
git add -f results/hpcm3_paper_bitrate_success/bitrate_success.png
git diff --cached --check
git commit -m "Publish audited Spatial and Object bitrate-success paper comparison"
for attempt in 1 2 3; do
  if git push personal exp/hpcm3-object-dist; then
    echo "Published $(git rev-parse HEAD)" | tee "${train_root}/paper_published.txt"
    exit 0
  fi
  sleep 60
done
echo "Paper summary committed locally but push failed" >&2
exit 1
