#!/usr/bin/env bash

MIN_COUNT=$1
MAX_COUNT=$2
OUT_DIR=$3

arguments=(
    --oversubscribe 
    -n 12 
    python3 -m src.examples.classification
    --logging_level INFO
    --dataset breast_cancer
    --number_genomes 1000 
    --input_qubits 8
    --batch_size 3
    --output_qubits 8
    --parent_strategy uniform 2 3
    --mutation_strategy uniform 2 3 
)

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec "${arguments[@]}" --out_dir $OUT_DIR/breast_i30_$3_${i} steady_state --max_population_size 30

done
