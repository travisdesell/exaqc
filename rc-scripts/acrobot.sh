#!/bin/bash -l
#SBATCH -J cartpole
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/acrobot/runs/output.o
#SBATCH -e ./logs/acrobot/runs/error.e
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
    --env acrobot \
    --algo reinforce \
    --number_genomes 1000 \
    --episodes 100 \
    --learning_rate 1e-3 \
    --rollout_steps 1024 \
    --ppo_epochs 4 \
    --ppo_minibatch 128 \
    --max_steps 500 \
    --input_qubits 6 \
    --output_qubits 3 \
    --out_dir artifacts/acrobot/pop/runs/${i} \
    steady_state \
    --max_population_size 50
done