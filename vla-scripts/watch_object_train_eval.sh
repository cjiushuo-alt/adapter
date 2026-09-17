#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/cjs/study/VLA/VLA-Adapter"
PYTHON="/home/cjs/anaconda3/envs/align-adapter/bin/python"
RUN_ID="shared-spatialpro-hpcm3-object-nonvision-r64-b2-ga8-seed7-20260917-2000"
MECHANICAL_ROOT="/media/cjs/shared/linux/VLA-Adapter/shared_vision_policy_ckpts"
TRAIN_LOG="${MECHANICAL_ROOT}/logs/${RUN_ID}.log"
CHECKPOINT="${MECHANICAL_ROOT}/runs/${RUN_ID}--200000_chkpt"
RESULT_ROOT="${REPO_ROOT}/results/hpcm3_object_retrain_eval"
MONITOR_LOG="${MECHANICAL_ROOT}/logs/${RUN_ID}.monitor.log"
EVAL_SCRIPT="${REPO_ROOT}/Align_adapter-HPCM3/code/inference/run_libero_eval.py"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() {
    echo "[$(date --iso-8601=seconds)] $*" >> "${MONITOR_LOG}"
}

log "monitor started"
while true; do
    if grep -q "Max step 200005 reached" "${TRAIN_LOG}"; then
        log "training completion marker found"
        break
    fi
    if ! pgrep -f "vla-scripts/finetune.py.*${RUN_ID}" >/dev/null; then
        log "ERROR: training process exited before completion"
        exit 10
    fi
    step="$(tr '\r' '\n' < "${TRAIN_LOG}" | sed -n 's/.*| *\([0-9][0-9]*\)\/200005.*/\1/p' | tail -1)"
    log "training healthy step=${step:-unknown}"
    sleep 600
done

for required in model.safetensors dataset_statistics.json; do
    if [[ ! -f "${CHECKPOINT}/${required}" ]]; then
        log "ERROR: missing final checkpoint artifact ${CHECKPOINT}/${required}"
        exit 11
    fi
done
if [[ "$(findmnt -n -o SOURCE -T "${CHECKPOINT}")" != "/dev/sda1" ]]; then
    log "ERROR: checkpoint is not on /dev/sda1"
    exit 12
fi

cd "${REPO_ROOT}"
for suite in object spatial; do
    suite_dir="${RESULT_ROOT}/${suite}"
    config="${suite_dir}/config.yaml"
    if [[ -f "${suite_dir}/summary.json" ]]; then
        log "skip completed evaluation suite=${suite}"
        continue
    fi
    log "startup check suite=${suite}"
    "${PYTHON}" "${EVAL_SCRIPT}" --config_path "${config}" --startup_check_only True \
        >> "${MONITOR_LOG}" 2>&1
    log "formal evaluation started suite=${suite}"
    "${PYTHON}" "${EVAL_SCRIPT}" --config_path "${config}" \
        > "${suite_dir}/console.log" 2>&1
    log "formal evaluation completed suite=${suite}"
done

"${PYTHON}" "${REPO_ROOT}/vla-scripts/summarize_hpcm3_object_retrain.py" \
    >> "${MONITOR_LOG}" 2>&1

git add -- \
    results/hpcm3_object_retrain_eval/report.md \
    results/hpcm3_object_retrain_eval/object/config.yaml \
    results/hpcm3_object_retrain_eval/object/summary.json \
    results/hpcm3_object_retrain_eval/object/episode_results.jsonl \
    results/hpcm3_object_retrain_eval/object/audit.jsonl \
    results/hpcm3_object_retrain_eval/object/run.log \
    results/hpcm3_object_retrain_eval/spatial/config.yaml \
    results/hpcm3_object_retrain_eval/spatial/summary.json \
    results/hpcm3_object_retrain_eval/spatial/episode_results.jsonl \
    results/hpcm3_object_retrain_eval/spatial/audit.jsonl \
    results/hpcm3_object_retrain_eval/spatial/run.log
git commit -m "report Object retraining and Object-Spatial evaluation"

for attempt in 1 2 3 4 5; do
    if git push personal exp/hpcm3-libero4 >> "${MONITOR_LOG}" 2>&1; then
        log "report committed and pushed"
        exit 0
    fi
    log "push attempt ${attempt} failed"
    sleep 60
done
log "ERROR: report commit created but push failed after retries"
exit 20
