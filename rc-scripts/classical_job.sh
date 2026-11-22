#!/bin/bash -l
#
# Trains one classical image baseline (as in rc-scripts/resnet.sh) as a Slurm
# job on RIT Tigris.
#
#   sbatch classical_job.sh <dataset> <model> <seed>
#
#   <dataset>  mnist | fashion_mnist | cifar10
#   <model>    resnet18 | resnet50 | vgg16 | densenet121 | efficientnet_b0 |
#              mobilenet_v3_large | convnext_tiny | cnn | mlp | linear ...
#   <seed>     --seed; also the run number in the output directory
#
# Training settings match resnet.sh: 100 epochs, Adam lr 1e-3, weight decay
# 5e-4, label smoothing 0.1, early stop after 20 epochs without improvement,
# batch 32, test batch 1. Model config defaults to
# configs/classical/<dataset>_<model>.json (only cifar10_* and mnist_cnn /
# mnist_resnet18 exist in the repo; set MODEL_CONFIG for anything else).
#
# Settings (environment variables):
#   MODEL_CONFIG   [configs/classical/<dataset>_<model>.json]
#   BATCH_SIZE [32]  TEST_BATCH_SIZE [1]  EPOCHS [100]  LEARNING_RATE [0.001]
#   WEIGHT_DECAY [0.0005]  LABEL_SMOOTHING [0.1]  IMPROVEMENT_CUTOFF [20]
#   NUM_WORKERS    data loader workers           [4]
#   DEVICE         cuda | cpu                    [cuda]
#   REPO, VENV, PYTHON, DATA_DIR, OUT_ROOT       cluster paths (defaults below)
#   DRY_RUN        1 = print the command instead of running it
#
#SBATCH -J exaqc_classical
#SBATCH -t 0-03:00:00
#SBATCH -A examm-rnn -p tigris
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=64GB

set -eu

if [ $# -ne 3 ]; then
    echo "usage: $0 <dataset> <model> <seed>" >&2
    exit 2
fi
DATASET=$1
MODEL=$2
SEED=$3

MODEL_CONFIG=${MODEL_CONFIG:-configs/classical/${DATASET}_${MODEL}.json}
BATCH_SIZE=${BATCH_SIZE:-32}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LEARNING_RATE:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
IMPROVEMENT_CUTOFF=${IMPROVEMENT_CUTOFF:-20}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}

REPO=${REPO:-/home/dk7405/Quantum/exaqc}
VENV=${VENV:-.exaqc}
PYTHON=${PYTHON:-python3.12}
DATA_DIR=${DATA_DIR:-/home/dk7405/datasets}
OUT_ROOT=${OUT_ROOT:-/home/dk7405/genome_archives/classical_runs}

# classical_image_classification writes to <out_dir>/<dataset>/<model>.
OUT_DIR="$OUT_ROOT/${DATASET}_${MODEL}_b${BATCH_SIZE}/runs/${SEED}"

COMMAND=(
    "$PYTHON" -m src.examples.classical_image_classification
    --dataset "$DATASET"
    --model "$MODEL"
    --model_config "$MODEL_CONFIG"
    --data_dir "$DATA_DIR"
    --out_dir "$OUT_DIR"
    --batch_size "$BATCH_SIZE"
    --validation_batch_size "$BATCH_SIZE"
    --test_batch_size "$TEST_BATCH_SIZE"
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --label_smoothing "$LABEL_SMOOTHING"
    --improvement_cutoff "$IMPROVEMENT_CUTOFF"
    --num_workers "$NUM_WORKERS"
    --pin_memory
    --device "$DEVICE"
    --seed "$SEED"
)

if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

set +u
cd "$REPO"
source "$VENV/bin/activate"
set -u

if [ ! -f "$MODEL_CONFIG" ]; then
    echo "error: model config $MODEL_CONFIG not found; set MODEL_CONFIG" >&2
    exit 2
fi

"${COMMAND[@]}"
