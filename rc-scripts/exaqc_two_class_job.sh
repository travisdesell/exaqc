#!/bin/bash -l
#
# Runs one EXAQC binary (2-class) image classification search as a Slurm job
# (RIT Tigris), for the reviewer's 2-class comparison.
#
#   sbatch exaqc_two_class_job.sh <dataset> <run>
#
#   <dataset>  mnist | fashion_mnist | cifar10
#   <run>      run number; also the --seed
#
# Needs the --classes option from classes-flag.patch (kept next to this script)
# applied to the repository on the cluster.
#
# Default class pairs are the ones AQEA-QAS (Li et al., 2025, Sec. 4.1) used,
# which is also where Table 1's EQNAS and AQEA-QAS numbers come from:
#   MNIST digits 3 and 6, Fashion-MNIST T-shirt (0) and dress (3),
#   CIFAR-10 cat (3) and dog (5).
#
# Search defaults follow the paper's Table 1 setup: steady-state population of
# 30, 800 genomes (1,000 for CIFAR-10), 100 epochs of Adam at lr 1e-3, RY input,
# probability readout, a one-layer CNN encoder for MNIST/Fashion-MNIST.
# Settings (environment variables):
#   CLASSES          original labels to keep           [per dataset, see above]
#   POP_SIZE         steady-state population size      [30]
#   NUMBER_GENOMES   genomes per run                   [800; 1000 for cifar10]
#   EPOCHS           training epochs per genome        [100]
#   QUBITS           input (and output) qubits         [5; 8 for cifar10]
#   INPUT_MODE       ry | rx | u3 | amplitude          [ry]
#   ENCODER_CONFIG   CNN encoder config                [configs/mnist_cnn_1.json; must be set for cifar10]
#   MUTATIONS        min and max mutations per child   [1 5]
#   PARENTS          min and max n-ary parents         [2 5]
#   REPO, VENV, PYTHON, DATA_DIR, OUT_ROOT, OPENMPI    cluster paths (see defaults below)
#   DRY_RUN          1 = print the command instead of running it
#
#SBATCH -J exaqc_two_class
#SBATCH -t 2-10:00:00
#SBATCH -A examm-rnn -p tigris
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=144
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB

set -eu

if [ $# -ne 2 ]; then
    echo "usage: $0 <dataset> <run>" >&2
    exit 2
fi
DATASET=$1
RUN=$2

case "$DATASET" in
    mnist)
        DEFAULT_CLASSES="3 6"; DEFAULT_GENOMES=800; DEFAULT_QUBITS=5
        DEFAULT_ENCODER=configs/mnist_cnn_1.json ;;
    fashion_mnist)
        DEFAULT_CLASSES="0 3"; DEFAULT_GENOMES=800; DEFAULT_QUBITS=5
        DEFAULT_ENCODER=configs/mnist_cnn_1.json ;;
    cifar10)
        DEFAULT_CLASSES="3 5"; DEFAULT_GENOMES=1000; DEFAULT_QUBITS=8
        DEFAULT_ENCODER= ;;
    *) echo "error: unknown image dataset: $DATASET" >&2; exit 2 ;;
esac

read -r -a CLASSES <<< "${CLASSES:-$DEFAULT_CLASSES}"
POP_SIZE=${POP_SIZE:-30}
NUMBER_GENOMES=${NUMBER_GENOMES:-$DEFAULT_GENOMES}
EPOCHS=${EPOCHS:-100}
QUBITS=${QUBITS:-$DEFAULT_QUBITS}
INPUT_MODE=${INPUT_MODE:-ry}
ENCODER_CONFIG=${ENCODER_CONFIG:-$DEFAULT_ENCODER}
read -r -a MUTATIONS <<< "${MUTATIONS:-1 5}"
read -r -a PARENTS <<< "${PARENTS:-2 5}"

if [ "${#CLASSES[@]}" -lt 2 ]; then
    echo "error: CLASSES needs at least two labels, got '${CLASSES[*]}'" >&2
    exit 2
fi
if [ -z "$ENCODER_CONFIG" ]; then
    echo "error: set ENCODER_CONFIG to the 3-layer CNN config used for CIFAR-10" >&2
    exit 2
fi

REPO=${REPO:-/home/dk7405/Quantum/exaqc}
VENV=${VENV:-.exaqc}
PYTHON=${PYTHON:-python3.12}
DATA_DIR=${DATA_DIR:-/home/dk7405/datasets}
OUT_ROOT=${OUT_ROOT:-/home/dk7405/genome_archives/two_class_runs}
OPENMPI=${OPENMPI:-/ttqroyz}

CLASS_TAG=$(IFS=v; echo "${CLASSES[*]}")
OUT_DIR="$OUT_ROOT/${DATASET}_c${CLASS_TAG}_${INPUT_MODE}_q${QUBITS}_pop${POP_SIZE}_${RUN}"

COMMAND=(
    "$PYTHON" -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$DATASET"
    --classes "${CLASSES[@]}"
    --data_dir "$DATA_DIR"
    --target pennylane
    --encoding cnn
    --encoder_config "$ENCODER_CONFIG"
    --decoding linear
    --input_qubits "$QUBITS"
    --output_qubits "$QUBITS"
    --quantum_input_mode "$INPUT_MODE"
    --quantum_output_mode probs
    --batch_size 64
    --validation_batch_size 64
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
spack load openmpi /ttqroyz
cd "$REPO"
source ".exaqc/bin/activate"
set -u

srun "${COMMAND[@]}"
