#!/bin/bash -l
#SBATCH -J iris
#SBATCH -t 0-03:00:00
#SBATCH -o ./outs/iris/runs/1/output.o
#SBATCH -e ./logs/iris/runs/1/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

srun python3 -m src.examples.pl_classification --logging_level INFO --dataset iris --number_genomes 500 --input_qubits 4 --out_dir ./artifacts/iris/runs/1