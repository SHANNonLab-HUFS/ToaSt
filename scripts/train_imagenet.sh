#!/usr/bin/env bash
# Compress DeiT-S on ImageNet with ToaST, then fine-tune.
#
#   IMAGENET=/path/to/imagenet bash scripts/train_imagenet.sh
#
#   IMAGENET=/path/to/imagenet TARGET_FLOPS=2.9 bash scripts/train_imagenet.sh
#
# Override anything through the environment, e.g. MODEL=deit_base_patch16_224 GPUS=8.
set -euo pipefail

IMAGENET="${IMAGENET:?set IMAGENET to the dataset root (containing train/ and val/)}"
MODEL="${MODEL:-deit_small_patch16_224}"
GPUS="${GPUS:-4}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PORT="${PORT:-29513}"

# Pick the compression schedule one of two ways.
#
#   TARGET_FLOPS=2.9   -> load this model's schedule from configs/tcs.json (preferred)
#   HEAD_SPARSITY=90 plus FC1_RATIOS / FC2_RATIOS -> spell it out
#
# See `python -m experiments.flops_table` for the budgets each model has, and
# experiments/{layer_sensitivity,ratio_patterns}.py for why the schedules are back-loaded.
SCHEDULE_ARGS=()
if [[ -n "${TARGET_FLOPS:-}" ]]; then
    SCHEDULE_ARGS+=(--target-flops "${TARGET_FLOPS}")
    TAG="f${TARGET_FLOPS}"
else
    HEAD_SPARSITY="${HEAD_SPARSITY:-90}"
    FC1_RATIOS="${FC1_RATIOS:-0 0 0 0 0 0 0 0 0 0 0 0}"
    FC2_RATIOS="${FC2_RATIOS:-0 0 0 0 0 0 0 0 0 0 0.9 0.9}"
    SCHEDULE_ARGS+=(--weight-pruning --head-sparsity "${HEAD_SPARSITY}"
                    --importance gm --coupling coupled
                    --fc1-prune-ratio ${FC1_RATIOS}
                    --fc2-prune-ratio ${FC2_RATIOS})
    TAG="sp${HEAD_SPARSITY}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./log/imagenet_${MODEL}_${TAG}}"
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node="${GPUS}" --master_port="${PORT}" main.py \
    --model "${MODEL}" \
    --data-path "${IMAGENET}" \
    --data-set IMNET \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr 0.001 \
    --min-lr 0.00001 \
    --weight-decay 0.0001 \
    "${SCHEDULE_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --autoresume \
    "$@" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
