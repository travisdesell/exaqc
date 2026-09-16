#!/bin/bash -l
#SBATCH -J exaqc_cifar10_eval
#SBATCH -t 1-00:00:00
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="cifar10"
RESULT_FOLDER="cifar10_ecnn_dlinear_doentangling_fcifar10_cnn_3_qery_qoprobs_g500_q8_b64"
RUN=1
GENOME_PATH="./artifacts/${RESULT_FOLDER}/runs/${RUN}/genome_242.json"
BATCH_SIZE=1

MODEL_FILENAME=$(basename "$GENOME_PATH" .json)

srun python3.11 -m src.examples.evaluate \
    --dataset $DATASET \
    --genome ${GENOME_PATH} \
    --device cuda \
    --batch_size $BATCH_SIZE \
    > ./outs/${DATASET}/${RESULT_FOLDER}_r${RUN}_${MODEL_FILENAME}.o \
    2> ./logs/${DATASET}/${RESULT_FOLDER}_r${RUN}_${MODEL_FILENAME}.e
