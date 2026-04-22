#!/bin/bash -l
#SBATCH -J frozenlake_i
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/frozenlake/pop/runs/output.o
#SBATCH -e ./logs/frozenlake/pop/runs/error.e
#SBATCH -A cps -p tier3
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

for i in $(seq 1 10); do
    srun python -m src.examples.pl_reinforce \
    --algo reinforce \
    --env frozenlake \
    --number_genomes 2000 \
    --input_qubits 4 \
    --output_qubits 4 \
    --episodes 1000 \
    --eval_episodes 100 \
    --max_steps 1000 \
    --gamma 0.99 \
    --learning_rate 0.001 \
    --entropy_coef 0.02 \
    --seed 0 \
    --log_every 50 \
    --logging_level INFO \
    --out_dir artifacts/frozenlake_rf/pop/runs1/${i} \
    steady_state --max_population_size 50
done

# --is_slippery \