#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:?usage: watch_object_dist_train_eval.sh TRAIN_PID}"
REPO_ROOT="/home/cjs/study/VLA/VLA-Adapter"
PYTHON="/home/cjs/anaconda3/envs/align-adapter/bin/python"
TRAIN_ROOT="/media/cjs/shared/linux/Align_adapter/hpcm3_object_dist"
RESULT_ROOT="$REPO_ROOT/results/hpcm3_object_dist/object"
CONFIG_SOURCE="$REPO_ROOT/Align_adapter-HPCM3/code/configs/eval_object_dist.yaml"
CONFIG_COPY="$RESULT_ROOT/config.yaml"
EVAL_SCRIPT="$REPO_ROOT/Align_adapter-HPCM3/code/inference/run_libero_eval.py"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

echo "[$(date --iso-8601=seconds)] waiting for training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done

if [[ ! -s "$TRAIN_ROOT/training_summary.json" || ! -s "$TRAIN_ROOT/adapter-best.pt" ]]; then
    echo "[$(date --iso-8601=seconds)] training ended without required outputs" >&2
    exit 2
fi

echo "[$(date --iso-8601=seconds)] building train-frame hash cache"
"$PYTHON" "$REPO_ROOT/Align_adapter-HPCM3/code/inference/build_train_frame_hash_cache.py" \
    --dataset-dir "$REPO_ROOT/Align_adapter/dataset" \
    --split-manifest "$TRAIN_ROOT/data_split.json" \
    --output "$TRAIN_ROOT/train_rgb_sha256.txt" \
    --workers 8

mkdir -p "$RESULT_ROOT"
if find "$RESULT_ROOT" -maxdepth 1 -type f -print -quit | grep -q .; then
    echo "Refusing to overwrite existing evaluation output: $RESULT_ROOT" >&2
    exit 3
fi
cp "$CONFIG_SOURCE" "$CONFIG_COPY"

echo "[$(date --iso-8601=seconds)] Object rollout startup check"
"$PYTHON" "$EVAL_SCRIPT" --config_path "$CONFIG_COPY" --startup_check_only True

echo "[$(date --iso-8601=seconds)] starting Object formal evaluation"
"$PYTHON" "$EVAL_SCRIPT" --config_path "$CONFIG_COPY" \
    >"$RESULT_ROOT/console.log" 2>&1
echo "[$(date --iso-8601=seconds)] Object formal evaluation complete"
