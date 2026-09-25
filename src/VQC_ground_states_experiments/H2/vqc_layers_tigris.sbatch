#!/bin/bash -l

#SBATCH --job-name=vqc-layers
#SBATCH --account=rc-default
#SBATCH --partition=tigris

#SBATCH --array=1-3

#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

#SBATCH --output=logs/vqc_%A_%a.out
#SBATCH --error=logs/vqc_%A_%a.err


set -euo pipefail


# RIT-supported base environment
spack env activate default-quantum-aarch64-24112201

# Our small PennyLane environment
source ~/venvs/vqc-tigris/bin/activate


# Prevent numerical libraries from creating extra CPU threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1


echo "============================================"
echo "Job:          ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Layer count:  ${SLURM_ARRAY_TASK_ID}"
echo "============================================"


python -u slurm_layered_ground_state.py \
    --layers "${SLURM_ARRAY_TASK_ID}" \
    --steps 10 \
    --stepsize 0.1 \
    --seed 0 \
    --output "results/layer_${SLURM_ARRAY_TASK_ID}.csv"