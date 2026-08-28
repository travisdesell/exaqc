#!/usr/bin/env bash
mpiexec -n 4 python -m src.examples.classification \
    --dataset mnist \
    --target pennylane \
    --encoding cnn \
    --encoder_config configs/mnist_cnn.json \
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
    --seed 42 \
    --out_dir artifacts/mnist \
    steady_state \
    --max_population_size 30