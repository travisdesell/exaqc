#!/bin/bash -l
#SBATCH -J breast_cancer
#SBATCH -t 1-00:00:00
#SBATCH -o ./outs/breast_cancer/output.o
#SBATCH -e ./logs/breast_cancer/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="breast_cancer"
QUBITS=5
ENCODING="identity"
QUANTUM_ENC="amplitude"
BATCH_SIZE=8
N_GENOMES=2000

srun python3 -m src.examples.classification \
    --dataset $DATASET \
    --target pennylane \
    --encoding $ENCODING \
    --decoding linear \
    --input_qubits $QUBITS \
    --output_qubits $QUBITS \
    --quantum_input_mode $QUANTUM_ENC \
    --quantum_output_mode probs \
    --batch_size $BATCH_SIZE \
    --validation_batch_size $BATCH_SIZE \
    --epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 0.0005 \
    --number_genomes $N_GENOMES \
    --mutation_strategy uniform 1 5 \
    --parent_strategy uniform 2 5 \
    --seed 42 \
    --out_dir artifacts/${DATASET}_${ENCODING}_${QUANTUM_ENC}_g${N_GENOMES}_q${QUBITS}_b${BATCH_SIZE} \
    steady_state \
    --max_population_size 30