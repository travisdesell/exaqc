#!/bin/bash -l
#SBATCH -J wine
#SBATCH -t 0-03:00:00
#SBATCH -o ./outs/wine/runs/output.o
#SBATCH -e ./logs/wine/runs/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

srun python3 -m src.examples.pl_classification --logging_level INFO --dataset wine --batch_size 64 --number_genomes 1000 --loss focal --out_dir ./artifacts/iris_f/runs/1 steady_state --max_population_size 30