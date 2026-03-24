#!/bin/bash -l
#SBATCH -J frozenlake
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/frozenlake/runs/output.o
#SBATCH -e ./logs/frozenlake/runs/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

for i in $(seq 1 1); do
    srun python -m src.examples.pl_reinforce \
    --algo reinforce \
    --logging_level INFO \
    --env frozenlake \
    --map_name 4x4 \
    --is_slippery \
    --input_qubits 4 \
    --output_qubits 4 \
    --episodes 1000 \
    --out_dir artifacts/frozenlake/runs/${i}
done
