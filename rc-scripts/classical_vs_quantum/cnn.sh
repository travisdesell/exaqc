#!/bin/bash -l
#SBATCH -J cnn_fmnist
#SBATCH -t 0-05:00:00
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="fashion_mnist"
MODEL="cnn"
# MODEL_CONFIGS=("configs/classical/cifar10_cnn_4.json" "configs/classical/cifar10_cnn_5.json" "configs/classical/cifar10_cnn_6.json")
MODEL_CONFIGS=("configs/fashion_mnist_cnn_2.json")
BATCH_SIZE=32

# MODEL_FILENAME=$(basename "$MODEL_CONFIG" .json)

MIN_COUNT=$1
MAX_COUNT=$2

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    MODEL_FILENAME=$(basename "$MODEL_CONFIG" .json)
    for i in $(seq $MIN_COUNT $MAX_COUNT); do

        TARGET_DIR="./outs/classical_v_quantum/$DATASET/runs/${i}"

        # Check if the directory does NOT exist
        if [ ! -d "$TARGET_DIR" ]; then
            echo "Directory does not exist. Creating it now..."
            mkdir -p "$TARGET_DIR"
        else
            echo "Directory already exists. Skipping."
        fi

        TARGET_DIR="./logs/classical_v_quantum/$DATASET/runs/${i}"

        # Check if the directory does NOT exist
        if [ ! -d "$TARGET_DIR" ]; then
            echo "Directory does not exist. Creating it now..."
            mkdir -p "$TARGET_DIR"
        else
            echo "Directory already exists. Skipping."
        fi
        python3.11 -m src.examples.classical_image_classification \
            --dataset $DATASET \
            --data_dir data \
            --out_dir artifacts/classical/${DATASET}_${MODEL}_f${MODEL_FILENAME}_b${BATCH_SIZE}/runs/${i} \
            --model $MODEL \
            --model_config $MODEL_CONFIG \
            --batch_size $BATCH_SIZE \
            --device cuda \
            --num_workers 4 \
            > ./outs/classical_v_quantum/$DATASET/runs/${i}/output_${MODEL}_${MODEL_FILENAME}.o \
            2> ./logs/classical_v_quantum/$DATASET/runs/${i}/error_${MODEL}_${MODEL_FILENAME}.e
    done
done

MODELS=("vgg11" "vgg13" "resnet18" "resnet50")
for MODEL in "${MODELS[@]}"; do
    for i in $(seq $MIN_COUNT $MAX_COUNT); do
        TARGET_DIR="./outs/classical_v_quantum/$DATASET/runs/${i}"

        # Check if the directory does NOT exist
        if [ ! -d "$TARGET_DIR" ]; then
            echo "Directory does not exist. Creating it now..."
            mkdir -p "$TARGET_DIR"
        else
            echo "Directory already exists. Skipping."
        fi

        TARGET_DIR="./logs/classical_v_quantum/$DATASET/runs/${i}"

        # Check if the directory does NOT exist
        if [ ! -d "$TARGET_DIR" ]; then
            echo "Directory does not exist. Creating it now..."
            mkdir -p "$TARGET_DIR"
        else
            echo "Directory already exists. Skipping."
        fi

        python3.11 -m src.examples.classical_image_classification \
            --dataset $DATASET \
            --data_dir data \
            --out_dir artifacts/classical/${DATASET}_${MODEL}_b${BATCH_SIZE}/runs/${i} \
            --model $MODEL \
            --batch_size $BATCH_SIZE \
            --device cuda \
            --num_workers 4 \
            > ./outs/classical_v_quantum/$DATASET/runs/${i}/output_${MODEL}.o \
            2> ./logs/classical_v_quantum/$DATASET/runs/${i}/error_${MODEL}.e
    done
done