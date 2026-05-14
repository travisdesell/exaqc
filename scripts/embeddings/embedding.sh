#!/bin/bash -l
#SBATCH -J extract_resnet
#SBATCH -t 0-08:00:00
#SBATCH -o ./outs/resnet/extract_output.o
#SBATCH -e ./logs/resnet/extract_error.e
#SBATCH -A cps -p tier3 -n 1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

python -m src.cnn.extract_resnet_embeddings \
    --dataset fashion_mnist \
    --mode extract \
    --checkpoint ./outputs/fashion_mnist_resnet50_emb15_b128_ep100.pt