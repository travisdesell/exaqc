#!/bin/bash -l
#SBATCH -J cnn_cifar10
#SBATCH -t 1-00:00:00
#SBATCH -o ./outs/cnn/output.o
#SBATCH -e ./logs/cnn/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="cifar10"
MODEL="cnn"
MODEL_CONFIG="configs/classical/cifar10_cnn_3.json"
BATCH_SIZE=64

MODEL_FILENAME=$(basename "$MODEL_CONFIG" .json)

python3.11 -m src.examples.classical_image_classification \
    --dataset cifar10 \
    --data_dir data \
    --out_dir artifacts/classical/${DATASET}_${MODEL}_f${MODEL_FILENAME}_b${BATCH_SIZE} \
    --model $MODEL \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --num_workers 4