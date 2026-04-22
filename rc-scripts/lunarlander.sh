#!/bin/bash -l
#SBATCH -J lunarlander
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/lunarlander/pop/runs/output.o
#SBATCH -e ./logs/lunarlander/pop/runs/error.e
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
    --env lunarlander \
    --algo reinforce \
    --number_genomes 1000 \
    --episodes 200 \
    --eval_episodes 100 \
    --learning_rate 1e-3 \
    --input_qubits 8 \
    --output_qubits 4 \
    --out_dir artifacts/lunarlander/reinforce/runs/${i} \
    steady_state \
    --max_population_size 30
done