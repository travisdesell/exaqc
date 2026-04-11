#!/bin/bash -l
#SBATCH -J cheetah
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/cheetah/pop/1/output.o
#SBATCH -e ./logs/cheetah/pop/1/error.e
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
    --env halfcheetah \
    --algo reinforce \
    --episodes 10000 \
    --eval_episodes 50 \
    --learning_rate 3e-4 \
    --rollout_steps 2048 \
    --ppo_epochs 10 \
    --ppo_minibatch 256 \
    --max_steps 1000 \
    --input_qubits 6 \
    --output_qubits 4 \
    --out_dir artifacts/halfcheetah/pop/runs/${i} \
    steady_state \
    --max_population_size 30
done
