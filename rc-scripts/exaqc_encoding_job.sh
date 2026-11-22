#!/bin/bash -l
#
# Runs one EXAQC 10-class image search with a given quantum input encoding, for
# the reviewer's dimension-matched amplitude-vs-angle comparison (RIT Tigris).
#
#   sbatch exaqc_encoding_job.sh <dataset> <input_mode> <run>
#
#   <dataset>     mnist | fashion_mnist | cifar10
#   <input_mode>  ry | rx | u3 | amplitude
#   <run>         run number; also the --seed
#
# Why this is dimension matched: classification.py sizes the encoder output as
# d_q = input_qubits for every mode except u3 (3 * input_qubits). Amplitude
# embedding normalises those d_q values and pads them with zeros to 2^n
# amplitudes. So at the same QUBITS, ry and amplitude get the same encoder, the
# same d_q and the same register; only the embedding differs.
#
# Defaults follow the Table 1 runs (read from their saved genomes): CNN encoder
# with no conv blocks and one 128-unit layer for MNIST/Fashion-MNIST (6 qubits,
# batch 32, 800 genomes), 3-conv-block CNN for CIFAR-10 (8 qubits, batch 64,
# 1,000 genomes), 20 epochs, lr 1e-3, probability readout, steady state.
# Settings (environment variables):
#   QUBITS           input (and output) qubits       [6; 8 for cifar10]
#   ENCODER_CONFIG   encoder config                  [configs/mnist_fc_table1.json; configs/cifar10_cnn.json]
#   BATCH_SIZE       [32; 64 for cifar10]   EPOCHS [20]   POP_SIZE [30]
#   NUMBER_GENOMES   [800; 1000 for cifar10]
#   MUTATIONS        min and max mutations per child [1 5]
#   PARENTS          min and max n-ary parents       [2 5]
#   REPO, VENV, PYTHON, DATA_DIR, OUT_ROOT, OPENMPI  cluster paths (defaults below)
#   DRY_RUN          1 = print the command instead of running it
#
#SBATCH -J exaqc_encoding
#SBATCH -t 5-00:00:00
#SBATCH -A examm-rnn -p tigris
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=144
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB

set -eu

if [ $# -ne 3 ]; then
    echo "usage: $0 <dataset> <input_mode> <run>" >&2
    exit 2
fi
DATASET=$1
INPUT_MODE=$2
RUN=$3

case "$INPUT_MODE" in
    ry|rx|rz|u3|amplitude) ;;
    *) echo "error: unknown input mode: $INPUT_MODE" >&2; exit 2 ;;
esac

case "$DATASET" in
    mnist|fashion_mnist)
        DEFAULT_QUBITS=6; DEFAULT_BATCH=32; DEFAULT_GENOMES=800
        DEFAULT_ENCODER=configs/mnist_fc.json ;;
    cifar10)
        DEFAULT_QUBITS=8; DEFAULT_BATCH=64; DEFAULT_GENOMES=1000
        DEFAULT_ENCODER=configs/cifar10_cnn_3.json ;;
    *) echo "error: unknown image dataset: $DATASET" >&2; exit 2 ;;
esac

QUBITS=${QUBITS:-$DEFAULT_QUBITS}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH}
NUMBER_GENOMES=${NUMBER_GENOMES:-$DEFAULT_GENOMES}
ENCODING=${ENCODING:-cnn}
ENCODER_CONFIG=${ENCODER_CONFIG:-$DEFAULT_ENCODER}
EPOCHS=${EPOCHS:-20}
POP_SIZE=${POP_SIZE:-30}
read -r -a MUTATIONS <<< "${MUTATIONS:-1 5}"
read -r -a PARENTS <<< "${PARENTS:-2 5}"

REPO=${REPO:-/home/dk7405/Quantum/exaqc}
VENV=${VENV:-.exaqc}
PYTHON=${PYTHON:-python3.12}
DATA_DIR=${DATA_DIR:-/home/dk7405/datasets}
OUT_ROOT=${OUT_ROOT:-/home/dk7405/genome_archives/dq_runs}
OPENMPI=${OPENMPI:-/ttqroyz}

OUT_DIR="$OUT_ROOT/${DATASET}_${INPUT_MODE}_q${QUBITS}_pop${POP_SIZE}_${RUN}"

COMMAND=(
    "$PYTHON" -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$DATASET"
    --data_dir "$DATA_DIR"
    --target pennylane
    --encoding "$ENCODING"
    --encoder_config "$ENCODER_CONFIG"
    --decoding linear
    --input_qubits "$QUBITS"
    --output_qubits "$QUBITS"
    --quantum_input_mode "$INPUT_MODE"
    --quantum_output_mode probs
    --batch_size "$BATCH_SIZE"
    --validation_batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --learning_rate 0.001
    --number_genomes "$NUMBER_GENOMES"
    --mutation_strategy uniform "${MUTATIONS[@]}"
    --parent_strategy uniform "${PARENTS[@]}"
    --seed "$RUN"
    --out_dir "$OUT_DIR"
    --shared_file_system
    steady_state --max_population_size "$POP_SIZE"
)

if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

set +u
export PMIX_MCA_psec="^munge"
spack load openmpi "$OPENMPI"
cd "$REPO"
source "$VENV/bin/activate"
set -u

srun "${COMMAND[@]}"
