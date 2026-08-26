#!/bin/bash -l
#SBATCH -J exaqc_mnist_angle
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/mnist/runs/5/output_angle.o
#SBATCH -e ./logs/mnist/runs/5/error_angle.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="mnist"
QUBITS=5
ENCODING="linear"
QUANTUM_ENC="ry"
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
# --encoder_config configs/mnist_cnn_2.json \

MIN_COUNT=$1
MAX_COUNT=$2

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    srun python3.11 -m src.examples.classification \
        --dataset $DATASET \
        --target pennylane \
        --encoding $ENCODING \
        --decoding linear \
        --encoder_config configs/mnist_fc.json \
        --input_qubits $QUBITS \
        --output_qubits $QUBITS \
        --quantum_input_mode $QUANTUM_ENC \
        --quantum_output_mode probs \
        --device cuda \
        --batch_size $BATCH_SIZE \
        --validation_batch_size $BATCH_SIZE \
        --epochs 20 \
        --learning_rate 0.001 \
        --number_genomes $N_GENOMES \
        --mutation_strategy uniform 1 5 \
        --parent_strategy uniform 2 5 \
        --seed $((i + 40)) \
        --out_dir artifacts/${DATASET}_${ENCODING}_${QUANTUM_ENC}_g${N_GENOMES}_q${QUBITS}_b${BATCH_SIZE}/runs/${i} \
        steady_state \
        --max_population_size 30
done