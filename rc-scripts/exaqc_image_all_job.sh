#!/bin/bash -l
#
# Runs a single EXAQC image classification search as a Slurm job (RIT Tigris).
#
#   sbatch exaqc_image_all_job.sh <dataset> <population> <sho|fixed> <run>
#
#   <dataset>     mnist | fashion_mnist | cifar10
#   <population>  fully_connected | ring | star | 2d_mesh | tree | random | steady_state
#   <sho|fixed>   co-evolve hyperparameters with SHO, or use the configured ones
#   <run>         run number; also the --seed
#
# Normally submitted by submit_image_all.sh, which loops over runs, datasets
# and populations and sets the job name and log files. One job is one run.
# Search and cluster settings come from environment variables (sbatch passes
# the submitting shell's environment through), with the defaults below.
#
# The defaults are the settings of the sho04 SHO islands runs (commit
# 98bccff): linear encoder with configs/mnist_fc.json, 4 qubits, probs
# readout, 20 epochs, batch 64, 1,000 genomes, 6 islands of 10.
#
# With sho, every run evolves the training settings (learning rate, weight
# decay, Adam), and islands runs also evolve the operator settings (crossover
# rates and mutation count). EXAQC only allows operator settings to be tuned
# on islands, so steady-state runs evolve the training settings only.
# SHO_OPERATORS=0 keeps islands to the training settings too. With fixed, the
# crossover rates are EXAQC's defaults (binary 0, n-ary 0.2, exponential 0.1),
# the centres of the SHO burn-in ranges.
#
# Search settings:
#   N_ISLANDS         islands per run              [6]
#   MAX_ISLAND_SIZE   genomes per island           [10]
#   MESH              2d_mesh dimensions, must multiply to N_ISLANDS [2 3]
#   TREE_CHILDREN     children per node for tree   [2]
#   RANDOM_EDGES      min and max edges for random [1 3]
#   POP_SIZE          steady-state population size [N_ISLANDS * MAX_ISLAND_SIZE]
#   NUMBER_GENOMES    genomes per run              [1000]
#   MUTATIONS         min and max mutations per child (uniform) [1 5]
#   PARENTS           min and max n-ary crossover parents (uniform) [2 5]
#   ENCODING          linear or cnn                [linear]
#   ENCODER_CONFIG    encoder configuration        [configs/mnist_fc.json]
#   EPOCHS            training epochs per genome   [20]
#   SHO_OPERATORS     1 = also tune operator settings on islands [1]
#
# Cluster settings:
#   REPO              repository on the cluster    [/home/dk7405/Quantum/exaqc]
#   VENV              venv to activate, relative to REPO or absolute [.exaqc]
#   PYTHON            python inside the venv       [python3.12]
#   DATA_DIR          image datasets (download them first; nodes may be offline) [/home/dk7405/datasets]
#   OUT_ROOT          where the archives go        [/home/dk7405/genome_archives/image_runs]
#   OPENMPI           spack spec for OpenMPI       [/ttqroyz]
#   DRY_RUN           1 = print the command instead of running it
#
# The resource requests below are defaults; submit_image_all.sh overrides them
# on the sbatch command line.
#
#SBATCH -J exaqc_image_all
#SBATCH -t 3-00:00:00
#SBATCH -A examm-rnn -p tigris
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=144
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB

set -eu

if [ $# -ne 4 ]; then
    echo "usage: $0 <dataset> <population> <sho|fixed> <run>" >&2
    exit 2
fi
DATASET=$1
POPULATION=$2
MODE=$3
RUN=$4
case "$MODE" in
    sho|fixed) ;;
    *) echo "error: mode must be sho or fixed, not '$MODE'" >&2; exit 2 ;;
esac
case "$DATASET" in
    mnist|fashion_mnist|cifar10) ;;
    *) echo "error: unknown image dataset: $DATASET" >&2; exit 2 ;;
esac

N_ISLANDS=${N_ISLANDS:-6}
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-10}
read -r -a MESH <<< "${MESH:-2 3}"
TREE_CHILDREN=${TREE_CHILDREN:-2}
read -r -a RANDOM_EDGES <<< "${RANDOM_EDGES:-1 3}"
# equal total capacity by default, so steady-state is not just a smaller population
POP_SIZE=${POP_SIZE:-$((N_ISLANDS * MAX_ISLAND_SIZE))}
NUMBER_GENOMES=${NUMBER_GENOMES:-2000}
read -r -a MUTATIONS <<< "${MUTATIONS:-1 5}"
read -r -a PARENTS <<< "${PARENTS:-2 5}"
ENCODING=${ENCODING:-linear}
ENCODER_CONFIG=${ENCODER_CONFIG:-configs/mnist_fc.json}
EPOCHS=${EPOCHS:-20}
SHO_OPERATORS=${SHO_OPERATORS:-1}

REPO=${REPO:-/home/dk7405/Quantum/exaqc}
VENV=${VENV:-.exaqc}
PYTHON=${PYTHON:-python3.12}
DATA_DIR=${DATA_DIR:-/home/dk7405/datasets}
OUT_ROOT=${OUT_ROOT:-/home/dk7405/genome_archives/image_runs}
OPENMPI=${OPENMPI:-/ttqroyz}

ISLANDS=(islands --n_islands "$N_ISLANDS" --max_island_size "$MAX_ISLAND_SIZE" --islands_to_extinct 1)
TAG="i${N_ISLANDS}_p${MAX_ISLAND_SIZE}"
case "$POPULATION" in
    fully_connected|ring|star)
        TAG="${TAG}_$POPULATION"; POPULATION_ARGS=("${ISLANDS[@]}" --topology "$POPULATION") ;;
    2d_mesh)
        if [ $((MESH[0] * MESH[1])) -ne "$N_ISLANDS" ]; then
            echo "error: MESH=${MESH[*]} does not multiply to N_ISLANDS=$N_ISLANDS" >&2
            exit 2
        fi
        TAG="${TAG}_2d_mesh_${MESH[0]}x${MESH[1]}"; POPULATION_ARGS=("${ISLANDS[@]}" --topology 2d_mesh "${MESH[@]}") ;;
    tree)
        TAG="${TAG}_tree_${TREE_CHILDREN}"; POPULATION_ARGS=("${ISLANDS[@]}" --topology tree "$TREE_CHILDREN") ;;
    random)
        TAG="${TAG}_random_${RANDOM_EDGES[0]}_${RANDOM_EDGES[1]}"; POPULATION_ARGS=("${ISLANDS[@]}" --topology random "${RANDOM_EDGES[@]}") ;;
    steady_state)
        TAG="pop${POP_SIZE}_steady_state"; POPULATION_ARGS=(steady_state --max_population_size "$POP_SIZE") ;;
    *) echo "error: unknown population: $POPULATION" >&2; exit 2 ;;
esac

# The ranges of the sho04 runs: burn-in range, then the full allowed range.
SHO_ARGS=()
if [ "$MODE" = sho ]; then
    TUNE=(
        learning_rate=log:1e-4:1e-2:1e-5:0.1
        weight_decay=linear:0:1e-4:0:1e-3
        adam_beta1=linear:0.9:0.99
        adam_beta2=linear:0.9:0.999
        adam_epsilon=log:1e-9:1e-8
    )
    if [ "$POPULATION" != steady_state ] && [ "$SHO_OPERATORS" = 1 ]; then
        TUNE+=(
            binary_crossover_rate=linear:0:0.1:0:1
            n_ary_crossover_rate=linear:0.1:0.3:0:1
            exponential_crossover_rate=linear:0:0.2:0:1
            mutation_count=int:1:3:1:10
        )
    fi
    SHO_ARGS=(--hyperparameter_strategy simplex --sho_tune "${TUNE[@]}"
              --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5)
fi

# The archive directory carries the same label as the job name that
# submit_image_all.sh gives this job.
OUT_DIR="$OUT_ROOT/${DATASET}_${MODE}_${TAG}_${RUN}"

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job instead of leaving workers waiting until the time limit. The
# population subcommand comes last: --topology takes any number of values.
COMMAND=(
    "$PYTHON" -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$DATASET"
    --data_dir "$DATA_DIR"
    --target pennylane
    --encoding "$ENCODING"
    --encoder_config "$ENCODER_CONFIG"
    --decoding linear
    --input_qubits 4
    --output_qubits 4
    --quantum_input_mode ry
    --quantum_output_mode probs
    --batch_size 64
    --validation_batch_size 64
    --epochs "$EPOCHS"
    --learning_rate 0.001
    --number_genomes "$NUMBER_GENOMES"
    --mutation_strategy uniform "${MUTATIONS[@]}"
    --parent_strategy uniform "${PARENTS[@]}"
    ${SHO_ARGS[@]+"${SHO_ARGS[@]}"}
    --seed "$RUN"
    --out_dir "$OUT_DIR"
    --shared_file_system
    "${POPULATION_ARGS[@]}"
)

if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

# spack and venv activation scripts read unset variables
set +u
export PMIX_MCA_psec="^munge"
spack load openmpi "$OPENMPI"
cd "$REPO"
source "$VENV/bin/activate"
set -u

srun "${COMMAND[@]}"
