#!/usr/bin/env bash
# MNIST runs with the training hyperparameters co-evolved by simplex
# hyperparameter optimization (SHO), using the search settings of run_mnist.sh.
# Tunes the learning rate and Adam's settings (ranges from Kini et al., GECCO '23,
# with the learning rate's burn-in around run_mnist.sh's 0.001), the weight decay,
# the search's crossover rates and the number of mutations per child.
#
# Usage: bash scripts/run_mnist_sho.sh [DATASET] [OUT_DIR]
# DATASET is mnist (default), fashion_mnist or cifar10.

DATASET=${1:-mnist}
OUT_DIR=${2:-artifacts/${DATASET}_sho}

mpiexec -n 4 python -m src.examples.classification \
    --dataset $DATASET \
    --target pennylane \
    --encoding cnn \
    --encoder_config configs/mnist_cnn_1.json \
    --decoding linear \
    --input_qubits 4 \
    --output_qubits 4 \
    --quantum_input_mode ry \
    --quantum_output_mode probs \
    --batch_size 32 \
    --validation_batch_size 32 \
    --training_samples 5000 \
    --validation_samples 1000 \
    --epochs 20 \
    --learning_rate 0.001 \
    --number_genomes 500 \
    --mutation_strategy uniform 1 3 \
    --parent_strategy uniform 2 3 \
    --hyperparameter_strategy simplex \
    --sho_tune learning_rate=log:1e-4:1e-2:1e-5:0.1 \
        weight_decay=linear:0:1e-4:0:1e-3 \
        adam_beta1=linear:0.9:0.99 adam_beta2=linear:0.9:0.999 adam_epsilon=log:1e-9:1e-8 \
        binary_crossover_rate=linear:0:0.1:0:1 n_ary_crossover_rate=linear:0.1:0.3:0:1 \
        exponential_crossover_rate=linear:0:0.2:0:1 mutation_count=int:1:3:1:10 \
    --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5 \
    --seed 42 \
    --out_dir $OUT_DIR \
    steady_state \
    --max_population_size 30
