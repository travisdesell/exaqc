#!/bin/bash -l
#SBATCH -J exaqc_resnet_fmnist_u3
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/resnet_fmnist/output_u3.o
#SBATCH -e ./logs/resnet_fmnist/error_u3.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="fashion_mnist"
QUBITS=8
ENCODING="u3"
ACTIVATION="tanh"
BATCH_SIZE=64

if [[ "$DATASET" == "mnist" || "$DATASET" == "fashion_mnist" ]]; then
    HIDDEN_DIMS=64
    TRAIN_SAMPLES=6000
    TEST_SAMPLES=1000
else
    HIDDEN_DIMS=128
    TRAIN_SAMPLES=2500
    TEST_SAMPLES=500
fi

#   --freeze_resnet \
srun python3.11 -m src.examples.pl_image \
  --dataset $DATASET \
  --loss ce \
  --epochs 50 \
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
  --out_dir artifacts/${DATASET}_resnet_encoder/e2e/enc_u3 \
  --max_train_samples $TRAIN_SAMPLES \
  --max_test_samples $TEST_SAMPLES \
  --encoding $ENCODING \
  steady_state \
  --max_population_size 30