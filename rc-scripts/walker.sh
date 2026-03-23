#!/bin/bash -l
#SBATCH -J walker2d
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/walker2d/runs/1/output.o
#SBATCH -e ./logs/walker2d/runs/1/error.e
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
    --env walker2d \
    --algo ppo \
    --episodes 100 \
    --learning_rate 3e-4 \
    --rollout_steps 2048 \
    --ppo_epochs 10 \
    --ppo_minibatch 256 \
    --max_steps 1000 \
    --input_qubits 6 \
    --output_qubits 4 \
    --out_dir artifacts/walker2d/pop/runs/${i} \
    steady_state \
    --max_population_size 30
done
