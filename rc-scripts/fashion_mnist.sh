#!/bin/bash -l
#SBATCH -J exaqc_fmnist_u3
#SBATCH -t 3-00:00:00
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="fashion_mnist"
QUBITS=6
ENCODING="cnn"
MODEL_CONFIG="configs/mnist_fc.json"
QUANTUM_ENC="u3"
QUANTUM_OUT="probs"
BATCH_SIZE=32
N_GENOMES=800

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
        --decoding linear \
        --encoder_config ${MODEL_CONFIG} \
        --input_qubits $QUBITS \
        --output_qubits $QUBITS \
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
        --out_dir artifacts/${DATASET}_${ENCODING}_f${MODEL_FILENAME}_${QUANTUM_ENC}_${QUANTUM_OUT}_g${N_GENOMES}_q${QUBITS}_b${BATCH_SIZE}/runs/${i} \
        steady_state \
        --max_population_size 30 \
        > ./outs/$DATASET/runs/${i}/output_${QUANTUM_ENC}_1.o \
        2> ./logs/$DATASET/runs/${i}/error_${QUANTUM_ENC}_1.o
done