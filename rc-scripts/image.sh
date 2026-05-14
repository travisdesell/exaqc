#!/bin/bash -l
#SBATCH -J exaqc_mnist
#SBATCH -t 1-00:00:00
#SBATCH -o ./outs/mnist/runs/output.o
#SBATCH -e ./logs/mnist/runs/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

srun python -m src.examples.pl_image \
  --dataset mnist \
  --loss ce \
  --epochs 50 \
  --learning_rate 1e-3 \
  --number_genomes 500 \
  --input_qubits 8 \
  --batch_size 32 \
  --hidden_dims \
  --activation tanh \
  --max_train_samples 2000 \
  --max_test_samples 500 \
  --out_dir artifacts/mnist_linear_encoder \
  --encoding u3 \
  steady_state \
  --max_population_size 30
