#!/bin/bash -l
#SBATCH -J exaqc_cifar10_gqdo
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/cifar10/output_gqdo.o
#SBATCH -e ./logs/cifar10/error_gqdo.e
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
QUBITS=6
ENCODING="cnn"
BATCH_SIZE=32
QUANTUM_DROPOUT_TYPE="gate"
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

srun python3.11 -m src.examples.classification \
    --dataset $DATASET \
    --target pennylane \
    --encoding $ENCODING \
    --decoding linear \
    --encoder_config configs/mnist_cnn_3.json \
    --input_qubits $QUBITS \
    --output_qubits $QUBITS \
    --quantum_input_mode ry \
    --quantum_output_mode probs \
    --quantum_dropout_type $QUANTUM_DROPOUT_TYPE \
    --quantum_dropout_rate 0.1 \
    --device cuda \
    --batch_size $BATCH_SIZE \
    --validation_batch_size $BATCH_SIZE \
    --epochs 20 \
    --learning_rate 0.001 \
    --weight_decay 0.0005 \
    --number_genomes $N_GENOMES \
    --mutation_strategy uniform 1 5 \
    --parent_strategy uniform 2 5 \
    --seed 42 \
    --out_dir artifacts/${DATASET}_g${N_GENOMES}_${ENCODING}_b${BATCH_SIZE}_q${QUBITS}_3_dropout_${QUANTUM_DROPOUT_TYPE} \
    steady_state \
    --max_population_size 30