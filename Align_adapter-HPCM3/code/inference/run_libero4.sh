#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/cjs/study/VLA/VLA-Adapter"
PYTHON="/home/cjs/anaconda3/envs/align-adapter/bin/python"
EVAL_SCRIPT="$REPO_ROOT/Align_adapter-HPCM3/code/inference/run_libero_eval.py"
RESULT_ROOT="$REPO_ROOT/results/hpcm3_libero4"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

for suite in spatial object goal libero_10; do
    suite_dir="$RESULT_ROOT/$suite"
    if [[ -f "$suite_dir/summary.json" ]]; then
        echo "[$(date --iso-8601=seconds)] skip completed suite: $suite"
        continue
    fi
    if find "$suite_dir" -maxdepth 1 -type f ! -name config.yaml -print -quit | grep -q .; then
        echo "Refusing to overwrite incomplete suite output: $suite_dir" >&2
        exit 2
    fi
    echo "[$(date --iso-8601=seconds)] start suite: $suite"
    "$PYTHON" "$EVAL_SCRIPT" --config_path "$suite_dir/config.yaml" \
        >"/tmp/hpcm3-libero4-${suite}-console.log" 2>&1
    echo "[$(date --iso-8601=seconds)] completed suite: $suite"
done

if [[ ! -e "$RESULT_ROOT/summary.md" ]]; then
    "$PYTHON" "$REPO_ROOT/Align_adapter-HPCM3/code/inference/summarize_libero4.py"
fi
echo "[$(date --iso-8601=seconds)] all suites completed"
