#!/bin/bash -l
#SBATCH -J exaqc_cifar10_eval
#SBATCH -t 1-00:00:00
#SBATCH -A cps -p tier3
#SBATCH -o ./outs/cifar10/compare/output_eval.o
#SBATCH -e ./logs/cifar10/compare/error_eval.e
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="cifar10"
GENOME_PATH="./artifacts/cifar10_cnn_ry_g1000_q8_b64/runs/1/genome_242.json"
BATCH_SIZE=1

srun python3.11 -m src.examples.evaluate \
    --dataset $DATASET \
    --genome ${GENOME_PATH} \
    --device cuda \
    --batch_size $BATCH_SIZE
