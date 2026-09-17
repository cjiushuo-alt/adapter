#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MECHANICAL_ROOT="${MECHANICAL_ROOT:-/media/cjs/shared/linux/VLA-Adapter/shared_vision_policy_ckpts}"
RUN_ID="${RUN_ID:-shared-spatialpro-vision-object-policy-r64-b2-ga8-$(date +%Y%m%d-%H%M%S)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/cjs/anaconda3/envs/align-adapter/bin/torchrun}"
RUN_ROOT="${MECHANICAL_ROOT}/runs"
LOG_ROOT="${MECHANICAL_ROOT}/logs"
WANDB_ROOT="${MECHANICAL_ROOT}/wandb"
LOG_FILE="${LOG_ROOT}/${RUN_ID}.log"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${WANDB_ROOT}"

mount_source="$(findmnt -n -o SOURCE -T "${MECHANICAL_ROOT}")"
if [[ "${mount_source}" != "/dev/sda1" ]]; then
    echo "Refusing to train: ${MECHANICAL_ROOT} is on ${mount_source}, expected /dev/sda1" >&2
    exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "torchrun is not executable: ${TORCHRUN_BIN}" >&2
    exit 1
fi
if [[ -e "${RUN_ROOT}/${RUN_ID}" ]]; then
    echo "Refusing to overwrite existing run: ${RUN_ROOT}/${RUN_ID}" >&2
    exit 1
fi

export WANDB_DIR="${WANDB_ROOT}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
echo "Object training log: ${LOG_FILE}"
exec "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir libero_rlds \
    --dataset_name libero_object_no_noops \
    --run_root_dir "${RUN_ROOT}" \
    --shared_vision_checkpoint outputs/LIBERO-Spatial-Pro \
    --freeze_shared_vision True \
    --exclude_vision_from_lora True \
    --verify_shared_vision_sha256 True \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --num_steps_before_decay 200000 \
    --max_steps 200005 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 2 \
    --seed 7 \
    --grad_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_mode online \
    --wandb_project vla-adapter-shared-vision \
    --run_id_override "${RUN_ID}" \
    >> "${LOG_FILE}" 2>&1
