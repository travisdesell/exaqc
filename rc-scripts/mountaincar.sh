#!/bin/bash -l
#SBATCH -J mountaincar_rf
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/mountaincar/rf_pop/output2.o
#SBATCH -e ./logs/mountaincar/rf_pop/error2.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

for i in $(seq 1 4); do
  srun python -m src.examples.pl_reinforce \
    --env mountaincar_continuous \
    --algo reinforce \
    --number_genomes 1000 \
    --episodes 100 \
    --learning_rate 3e-4 \
    --rollout_steps 1024 \
    --ppo_epochs 10 \
    --ppo_minibatch 128 \
    --max_steps 1000 \
    --input_qubits 2 \
    --output_qubits 2 \
    --out_dir artifacts/mountaincar_continuous/rf_pop/runs/${i} \
    steady_state \
    --max_population_size 30
done
