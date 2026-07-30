#!/usr/bin/env bash
# Transfer a ToaST-compressed model to CIFAR-100.
#
#   CIFAR100=/path/to/cifar100 PRETRAINED=/ckpt/toast_deit_s_90.pth \
#       bash scripts/finetune_cifar100.sh
#
# Omit PRETRAINED to start from timm's ImageNet weights (the uncompressed baseline).
set -euo pipefail

CIFAR100="${CIFAR100:?set CIFAR100 to the dataset root}"
MODEL="${MODEL:-deit_small_patch16_224}"
GPUS="${GPUS:-1}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-./log/cifar100_${MODEL}}"
PORT="${PORT:-29514}"

PRUNE_ARGS=()
if [[ -n "${TARGET_FLOPS:-}" ]]; then
    PRUNE_ARGS+=(--target-flops "${TARGET_FLOPS}")
elif [[ -n "${HEAD_SPARSITY:-}" ]]; then
    PRUNE_ARGS+=(--weight-pruning --head-sparsity "${HEAD_SPARSITY}")
fi
if [[ -n "${PRETRAINED:-}" ]]; then
    PRUNE_ARGS+=(--pretrained "${PRETRAINED}")
fi

mkdir -p "${OUTPUT_DIR}"

# Lighter augmentation and a warmup, unlike the ImageNet schedule; the 100-class head is
# re-initialised automatically when the checkpoint's head does not fit.
torchrun --nproc_per_node="${GPUS}" --master_port="${PORT}" main.py \
    --model "${MODEL}" \
    --data-path "${CIFAR100}" \
    --data-set CIFAR \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr 0.001 \
    --min-lr 0.000001 \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --scale-lr \
    --color-jitter 0.3 \
    --reprob 0.0 \
    --no-repeated-aug \
    "${PRUNE_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --autoresume \
    "$@" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
