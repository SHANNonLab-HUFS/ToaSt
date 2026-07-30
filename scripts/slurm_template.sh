#!/usr/bin/env bash
# SLURM template. Fill in the SBATCH values for your cluster, then:
#
#   sbatch --export=ALL,IMAGENET=/path/to/imagenet scripts/slurm_template.sh
#
#SBATCH --job-name=toast
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=toast_%j.out
#SBATCH --error=toast_%j.err
##SBATCH --account=<your-account>
##SBATCH --partition=<your-partition>
set -euo pipefail

# Adjust to your cluster's module system, or drop this block if you use conda/venv.
if command -v module >/dev/null 2>&1; then
    module purge
    module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10
fi

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== environment ==="
nvidia-smi
python --version
echo "GPUs on node: ${SLURM_GPUS_ON_NODE:-unknown}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GPUS="${SLURM_GPUS_ON_NODE:-4}" bash scripts/train_imagenet.sh "$@"
