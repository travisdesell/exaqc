#!/bin/bash -l
#SBATCH -J train_resnet
#SBATCH -t 0-08:00:00
#SBATCH -o ./outs/resnet/output.o
#SBATCH -e ./logs/resnet/error.e
#SBATCH -A cps -p tier3 -n 1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

python -m src.cnn.extract_resnet_embeddings \
    --dataset fashion_mnist \
    --mode train \
    --backbone resnet50 \
    --embedding-dim 15 \
    --pretrained \
    --epochs 100