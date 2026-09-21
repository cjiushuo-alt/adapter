#!/usr/bin/env bash
# 从 HPCM2 训练 ckpt 中提取 Adapter 权重并保存为 .pt
# 用法: ./run_extract_adapter_weights.sh
# 或: bash run_extract_adapter_weights.sh
# 请根据实际路径修改下面两行，或通过环境变量覆盖

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

INPUT_CKPT="${INPUT_CKPT:-/media/cjs/shared/linux/Align_adapter/Align_adapter_HPCM_ckpts/checkpoints/train-step-step=03600-v1.ckpt}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/cjs/study/VLA/VLA-Adapter/Align_adapter-HPCM3_ckpt/train-step-step=3600-v1.pt}"

cd "$PROJECT_ROOT"
python Align_adapter-HPCM3/inference/utils/extract_adapter_weights.py \
    --input_ckpt "$INPUT_CKPT" \
    --output_path "$OUTPUT_PATH"
