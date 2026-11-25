#!/bin/bash -l
#SBATCH -J iris_qiskit
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/iris/runs/output_n.o
#SBATCH -e ./logs/iris/runs/error_n.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

spack env activate default-ml-x86_64-25052701

source ./.venv/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/dk7405/Quantum/exaqc/.venv/lib/python3.11/site-packages
export PATH=/home/dk7405/Quantum/exaqc/.venv/bin:$PATH

MIN_COUNT=1
MAX_COUNT=1
LOSS="ce"
TARGET="qiskit"

BACKENDS=("ibm_kingston" "ibm_marrakesh" "ibm_fez")

# for b in "${BACKENDS[@]}"; do
#     for i in $(seq $MIN_COUNT $MAX_COUNT); do
#         srun /home/dk7405/Quantum/exaqc/.venv/bin/python -m src.examples.pl_classification_noisy \
#             --dataset iris \
#             --loss ${LOSS} \
#             --epochs 30 \
#             --learning_rate 5e-4 \
#             --number_genomes 1000 \
#             --input_qubits 4 \
#             --batch_size 32 \
#             --encoding angle \
#             --noise_type ibm_backend \
#             --ibm_backend ${b} \
#             --ibm_noise_verbose \
#             --out_dir artifacts/iris_${b}/runs/${i} \
#             steady_state \
#             --max_population_size 30
#     done
# done

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    srun /home/dk7405/Quantum/exaqc/.venv/bin/python -m src.examples.pl_qiskit_classification \
        --dataset iris \
        --loss ${LOSS} \
        --epochs 30 \
        --learning_rate 5e-4 \
        --number_genomes 1000 \
        --input_qubits 4 \
        --batch_size 32 \
        --encoding angle \
        --noise_type ibm_backend \
        --ibm_backend ibm_fez \
        --ibm_noise_verbose \
        --out_dir artifacts/iris_noise_sqiskit/runs/${i} \
        --target ${TARGET} \
        steady_state \
        --max_population_size 30
done