#!/bin/bash -l
#SBATCH -J exaqc_cifar10_u3
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/cifar10/output_u3.o
#SBATCH -e ./logs/cifar10/error_u3.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

QUBITS=10
ENCODING="u3"
ACTIVATION="tanh"

srun python3.11 -m src.examples.pl_image \
  --dataset cifar10 \
  --target pennylane \
  --loss ce \
  --epochs 50 \
  --learning_rate 1e-4 \
  --number_genomes 1000 \
  --input_qubits $QUBITS \
  --batch_size 32 \
  --device gpu \
  --encoder_type cnn \
  --conv_channels 8 16 \
  --hidden_dims 128 \
  --activation $ACTIVATION \
  --mutation_strategy uniform 1 3 \
  --out_dir artifacts/cifar10_cnn_encoder/enc_u3 \
  --max_train_samples 5000 \
  --max_test_samples 1000 \
  --encoding $ENCODING \
  steady_state \
  --max_population_size 30