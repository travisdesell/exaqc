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

for i in $(seq 1 10); do
    srun python -m src.examples.pl_reinforce \
    --algo reinforce \
    --logging_level INFO \
    --env frozenlake \
    --number_genomes 1000 \
    --map_name 4x4 \
    --is_slippery \
    --input_qubits 4 \
    --output_qubits 4 \
    --episodes 1000 \
    --eval_episodes 100 \
    --mutation_strategy uniform 1 3 \
    --out_dir artifacts/frozenlake/islands/runs/${i} \
    islands \
    --n_islands 5 \
    --max_island_size 6 \
    --islands_to_extinct 1 \
    --genomes_before_extinction 50 \
    --genomes_for_next_extinction 200
done
