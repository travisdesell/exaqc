#!/bin/bash -l
#SBATCH -J mountaincar
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/mountaincar/runs/output.o
#SBATCH -e ./logs/mountaincar/runs/error.e
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
    --env mountaincar_continuous \
    --algo reinforce \
    --number_genomes 1000 \
    --episodes 100 \
    --learning_rate 3e-4 \
    --rollout_steps 1024 \
    --max_steps 1000 \
    --input_qubits 2 \
    --output_qubits 2 \
    --mutation_strategy uniform 1 3 \
    --out_dir artifacts/mountaincar_continuous/run/${i} \
    islands \
    --n_islands 5 \
    --max_island_size 6 \
    --islands_to_extinct 1 \
    --genomes_before_extinction 50 \
    --genomes_for_next_extinction 200
done
