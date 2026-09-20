#!/bin/bash -l
#SBATCH -J exaqc_cifar10_ry
#SBATCH -t 4-00:00:00
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="cifar10"
INPUT_QUBITS=8
OUTPUT_QUBITS=5
ENCODING="cnn"
DECODING="linear"
QUANTUM_ENC="ry"
QUANTUM_OUT="probs"
MODEL_CONFIG="configs/cifar10_cnn_6.json"
BATCH_SIZE=64
N_GENOMES=500

# if [[ "$DATASET" == "mnist" || "$DATASET" == "fashion_mnist" ]]; then
#     HIDDEN_DIMS=64
#     TRAIN_SAMPLES=3000
#     TEST_SAMPLES=500
# else
#     HIDDEN_DIMS=128
#     TRAIN_SAMPLES=2500
#     TEST_SAMPLES=500
# fi

# --training_samples $TRAIN_SAMPLES \
# --validation_samples $TEST_SAMPLES \
# --encoder_config configs/mnist_cnn_2.json \

MODEL_FILENAME=$(basename "$MODEL_CONFIG" .json)

MIN_COUNT=$1
MAX_COUNT=$2

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    TARGET_DIR="./outs/$DATASET/runs/$i"

    # Check if the directory does NOT exist
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Directory does not exist. Creating it now..."
        mkdir -p "$TARGET_DIR"
    else
        echo "Directory already exists. Skipping."
    fi

    TARGET_DIR="./logs/$DATASET/runs/$i"

    # Check if the directory does NOT exist
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Directory does not exist. Creating it now..."
        mkdir -p "$TARGET_DIR"
    else
        echo "Directory already exists. Skipping."
    fi
    srun python3.11 -m src.examples.classification \
        --dataset $DATASET \
        --target pennylane \
        --encoding $ENCODING \
        --decoding $DECODING \
        --encoder_config $MODEL_CONFIG \
        --input_qubits $INPUT_QUBITS \
        --output_qubits $OUTPUT_QUBITS \
        --quantum_input_mode $QUANTUM_ENC \
        --quantum_output_mode $QUANTUM_OUT \
        --device cuda \
        --batch_size $BATCH_SIZE \
        --validation_batch_size $BATCH_SIZE \
        --epochs 20 \
        --learning_rate 0.001 \
        --number_genomes $N_GENOMES \
        --mutation_strategy uniform 1 5 \
        --parent_strategy uniform 2 5 \
        --seed $((i + 40)) \
        --out_dir artifacts/classical/${DATASET}_e${ENCODING}_d${DECODING}_f${MODEL_FILENAME}_${QUANTUM_OUT}_g${N_GENOMES}_q${QUBITS}_b${BATCH_SIZE}/runs/${i} \
        steady_state \
        --max_population_size 30 \
        > ./outs/classical_v_quantum/$DATASET/runs/${i}/output_${QUANTUM_ENC}_q${QUBITS}.o \
        2> ./logs/classical_v_quantum/$DATASET/runs/${i}/error_${QUANTUM_ENC}_q${QUBITS}.o
done