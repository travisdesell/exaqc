#!/bin/bash -l
#SBATCH -J frozenlake
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/frozenlake_a/pop/runs/output.o
#SBATCH -e ./logs/frozenlake_a/pop/runs/error.e
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
    --algo q_learning \
    --env frozenlake \
    --number_genomes 300 \
    --input_qubits 4 \
    --output_qubits 4 \
    --episodes 200 \
    --eval_episodes 20 \
    --max_steps 100 \
    --gamma 0.99 \
    --learning_rate 0.01 \
    --entropy_coef 0.02 \
    --seed 0 \
    --log_every 10 \
    --logging_level INFO \
    --out_dir artifacts/frozenlake/pop/runs_angle/${i} \
    steady_state --max_population_size 30
done
