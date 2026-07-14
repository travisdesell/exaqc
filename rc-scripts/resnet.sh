#!/bin/bash -l
#SBATCH -J exaqc_resnet_cifar10_u3
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/resnet_cifar10/output_u3.o
#SBATCH -e ./logs/resnet_cifar10/error_u3.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="cifar10"
QUBITS=8
ENCODING="u3"
ACTIVATION="tanh"
BATCH_SIZE=64

if [[ "$DATASET" == "mnist" || "$DATASET" == "fashion_mnist" ]]; then
    HIDDEN_DIMS=64
    TRAIN_SAMPLES=3000
    TEST_SAMPLES=500
else
    HIDDEN_DIMS=128
    TRAIN_SAMPLES=2500
    TEST_SAMPLES=500
fi

#   --freeze_resnet \
srun python3.11 -m src.examples.pl_image \
  --dataset $DATASET \
  --loss ce \
  --epochs 100 \
  --learning_rate 1e-3 \
  --number_genomes 2000 \
  --input_qubits $QUBITS \
  --batch_size $BATCH_SIZE \
  --device gpu \
  --encoder_type resnet \
  --resnet_model resnet50 \
  --resnet_pretrained \
  --hidden_dims $HIDDEN_DIMS \
  --activation $ACTIVATION \
  --mutation_strategy uniform 1 3 \
  --parent_strategy uniform 1 3 \
  --out_dir artifacts/${DATASET}_resnet_encoder/e2e/enc_u3_1 \
  --max_train_samples $TRAIN_SAMPLES \
  --max_test_samples $TEST_SAMPLES \
  --encoding $ENCODING \
  steady_state \
  --max_population_size 30