#!/usr/bin/env bash
# Coupled vs single-sided importance: fine-tune one run per coupling and compare.
#
#   IMAGENET=/path/to/imagenet bash scripts/ablation_coupling.sh
#
# `coupled` scores [Q|K] and [V|O^T] together; the others score one projection of each pair
# and are the comparison the coupled formulation is argued against.
set -euo pipefail

IMAGENET="${IMAGENET:?set IMAGENET to the dataset root}"
MODEL="${MODEL:-deit_small_patch16_224}"
GPUS="${GPUS:-4}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HEAD_SPARSITY="${HEAD_SPARSITY:-90}"
TARGET_FLOPS="${TARGET_FLOPS:-}"
LOG_ROOT="${LOG_ROOT:-./log/ablation_coupling}"
PORT="${PORT:-29513}"
COUPLINGS="${COUPLINGS:-coupled q_only k_only proj_only average}"

for coupling in ${COUPLINGS}; do
    output_dir="${LOG_ROOT}/${coupling}"
    mkdir -p "${output_dir}"

    echo "=== coupling=${coupling}  start $(date -Is) ==="
    torchrun --nproc_per_node="${GPUS}" --master_port="${PORT}" main.py \
        --model "${MODEL}" \
        --data-path "${IMAGENET}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr 0.001 \
        --min-lr 0.00001 \
        ${TARGET_FLOPS:+--target-flops ${TARGET_FLOPS}} \
        --weight-pruning \
        --head-sparsity "${HEAD_SPARSITY}" \
        --importance gm \
        --coupling "${coupling}" \
        --output_dir "${output_dir}" \
        --autoresume \
        "$@" 2>&1 | tee -a "${output_dir}/train.log"
    echo "=== coupling=${coupling}  done $(date -Is) ==="
done

echo
echo "=== best Acc@1 per coupling ==="
for coupling in ${COUPLINGS}; do
    log="${LOG_ROOT}/${coupling}/log.txt"
    if [[ -f "${log}" ]]; then
        best=$(python3 -c "
import json, sys
best = max((json.loads(l).get('test_acc1', 0) for l in open(sys.argv[1])), default=0)
print(f'{best:.2f}')
" "${log}")
        printf '  %-10s %s%%\n' "${coupling}" "${best}"
    else
        printf '  %-10s (no log)\n' "${coupling}"
    fi
done
