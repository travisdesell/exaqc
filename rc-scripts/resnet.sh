#!/bin/bash -l
#SBATCH -J resnet50_cifar10
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/resnet50/output.o
#SBATCH -e ./logs/resnet50/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="cifar10"
MODEL="resnet50"
BATCH_SIZE=32


python3.11 -m src.examples.classical_image_classification \
    --dataset $DATASET \
    --model $MODEL \
    --model_config configs/classical/${DATASET}_${MODEL}.json \
    --batch_size $BATCH_SIZE \
    --validation_batch_size $BATCH_SIZE \
    --test_batch_size 1 \
    --epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 0.0005 \
    --label_smoothing 0.1 \
    --improvement_cutoff 20 \
    --num_workers 4 \
    --pin_memory \
    --device cuda \
    --seed 42
