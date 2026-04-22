#!/bin/bash -l
#SBATCH -J pendulum
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/pendulum/pop/runs/output.o
#SBATCH -e ./logs/pendulum/pop/runs/error.e
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
    --env pendulum \
    --algo reinforce \
    --number_genomes 1000 \
    --episodes 200 \
    --learning_rate 3e-4 \
    --input_qubits 3 \
    --output_qubits 1 \
    --out_dir artifacts/pendulum/reinforce/runs/${i} \
    steady_state \
    --max_population_size 30
done