#!/bin/bash -l
#SBATCH -J cheetah
#SBATCH -t 0-12:00:00
#SBATCH -o ./outs/cheetah/runs/1/output.o
#SBATCH -e ./logs/cheetah/runs/1/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

srun python -m src.examples.pl_reinforce \
  --env halfcheetah \
  --algo ppo \
  --episodes 100 \
  --learning_rate 3e-4 \
  --rollout_steps 2048 \
  --ppo_epochs 10 \
  --ppo_minibatch 256 \
  --max_steps 1000 \
  --input_qubits 6 \
  --output_qubits 4 \
  --out_dir artifacts/halfcheetah/run/1
