#!/bin/bash -l
#SBATCH -J walker2d
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/walker2d/islands/output1.o
#SBATCH -e ./logs/walker2d/islands/error1.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

for i in $(seq 1 10); do
  srun python -m src.examples.pl_reinforce \
    --env walker2d \
    --algo reinforce \
    --number_genomes 2000 \
    --episodes 100 \
    --learning_rate 3e-4 \
    --rollout_steps 2048 \
    --ppo_epochs 10 \
    --ppo_minibatch 256 \
    --max_steps 1000 \
    --input_qubits 6 \
    --output_qubits 6 \
    --episodes 100 \
    --out_dir artifacts/walker2d/islands/runs/${i} \
    islands --n_islands 10 --max_island_size 3
done
