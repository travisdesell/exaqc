#!/bin/bash -l
#
# Runs a single EXAQC UCI classification search as a Slurm job (RIT Tigris).
#
#   sbatch exaqc_uci_all_job.sh <dataset> <population> <mode> <run>
#
#   <dataset>     iris | seeds | wine | breast_cancer
#   <population>  fully_connected | ring | star | 2d_mesh | tree | random | steady_state
#   <mode>        fixed          use the configured hyperparameters
#                 sho            SHO on the training settings, plus the operator
#                                settings on islands (see below)
#                 sho_training   SHO on the training settings only
#                 sho_operators  SHO on the operator settings only (islands only):
#                                crossover rates, mutation count, mutation weights
#                 sho_mutations  SHO on the mutation settings only (islands only):
#                                mutation count and mutation weights
#   <run>         run number; also the --seed
#
# Normally submitted by submit_uci_all.sh, which loops over runs, datasets and
# populations and sets the job name and log files. One job is one run.
# Search and cluster settings come from environment variables (sbatch passes
# the submitting shell's environment through), with the defaults below.
#
# With sho, islands runs evolve the training settings (learning rate, Adam)
# and the operator settings (crossover rates, mutation count, mutation
# weights). EXAQC only allows operator settings to be tuned on islands, so
# steady-state runs evolve the training settings only. SHO_OPERATORS=0 keeps
# islands to the training settings too.
#
# Search settings:
#   N_ISLANDS         islands per run              [6]
#   MAX_ISLAND_SIZE   genomes per island           [10]
#   MESH              2d_mesh dimensions, must multiply to N_ISLANDS [2 3]
#   TREE_CHILDREN     children per node for tree   [2]
#   RANDOM_EDGES      min and max edges for random [1 3]
#   POP_SIZE          steady-state population size [N_ISLANDS * MAX_ISLAND_SIZE]
#   NUMBER_GENOMES    genomes per run              [1000]
#   MUTATIONS         min and max mutations per child (uniform) [1 3]
#   PARENTS           min and max n-ary crossover parents (uniform) [2 5]
#   SHO_OPERATORS     1 = also tune operator settings on islands [1]
#
# Cluster settings:
#   REPO              repository on the cluster    [/home/dk7405/Quantum/exaqc]
#   VENV              venv to activate, relative to REPO or absolute [.exaqc]
#   PYTHON            python inside the venv       [python3.12]
#   OUT_ROOT          where the archives go        [/home/dk7405/genome_archives/uci_runs]
#   OPENMPI           spack spec for OpenMPI       [/ttqroyz]
#   DRY_RUN           1 = print the command instead of running it
#
# The resource requests below are defaults; submit_uci_all.sh overrides them
# on the sbatch command line.
#
#SBATCH -J exaqc_uci_all
#SBATCH -t 0-12:00:00
#SBATCH -A examm-rnn -p tigris
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

set -eu

if [ $# -ne 4 ]; then
    echo "usage: $0 <dataset> <population> <fixed|sho|sho_training|sho_operators|sho_mutations> <run>" >&2
    exit 2
fi
DATASET=$1
POPULATION=$2
MODE=$3
RUN=$4
case "$MODE" in
    fixed|sho|sho_training|sho_operators|sho_mutations) ;;
    *) echo "error: mode must be fixed, sho, sho_training, sho_operators or sho_mutations, not '$MODE'" >&2; exit 2 ;;
esac

N_ISLANDS=${N_ISLANDS:-6}
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-10}
read -r -a MESH <<< "${MESH:-2 3}"
TREE_CHILDREN=${TREE_CHILDREN:-2}
read -r -a RANDOM_EDGES <<< "${RANDOM_EDGES:-1 3}"
# equal total capacity by default, so steady-state is not just a smaller population
POP_SIZE=${POP_SIZE:-$((N_ISLANDS * MAX_ISLAND_SIZE))}
NUMBER_GENOMES=${NUMBER_GENOMES:-1000}
read -r -a MUTATIONS <<< "${MUTATIONS:-1 3}"
read -r -a PARENTS <<< "${PARENTS:-2 5}"
SHO_OPERATORS=${SHO_OPERATORS:-1}

REPO=${REPO:-/home/dk7405/Quantum/exaqc}
VENV=${VENV:-.exaqc}
PYTHON=${PYTHON:-python3.12}
OUT_ROOT=${OUT_ROOT:-/home/dk7405/genome_archives/uci_runs}
OPENMPI=${OPENMPI:-/ttqroyz}

# identity encoding needs one input qubit per feature; breast_cancer is
# linearly encoded instead, with a clipped decoder for its two classes
case "$DATASET" in
    iris) DATASET_ARGS=(--input_qubits 4 --output_qubits 2 --encoding identity --decoding linear) ;;
    seeds) DATASET_ARGS=(--input_qubits 7 --output_qubits 2 --encoding identity --decoding linear) ;;
    wine) DATASET_ARGS=(--input_qubits 13 --output_qubits 2 --encoding identity --decoding linear) ;;
    breast_cancer) DATASET_ARGS=(--input_qubits 8 --output_qubits 1 --encoding linear --decoding clipped) ;;
    *) echo "error: unknown UCI dataset: $DATASET" >&2; exit 2 ;;
esac

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

# The ranges used for the paper's runs (Kini et al., GECCO '23 for the
# training settings): burn-in range, then the full allowed range.
TRAINING_TUNE=(
    learning_rate=log:1e-3:5e-2:1e-5:0.3
    adam_beta1=linear:0.9:0.99
    adam_beta2=linear:0.9:0.99
    adam_epsilon=log:1e-9:1e-8
)
CROSSOVER_TUNE=(
    binary_crossover_rate=linear:0.05:0.15:0:1
    n_ary_crossover_rate=linear:0.05:0.15:0:1
    exponential_crossover_rate=linear:0.05:0.15:0:1
)
MUTATION_TUNE=(
    mutation_count=int:1:3:1:10
    add_gate_weight=linear:8:14:0:30
    reorder_gate_weight=linear:1:3:0:10
    qubit_swap_weight=linear:1:3:0:10
    enable_gate_weight=linear:0.5:1.5:0:10
    disable_gate_weight=linear:1:3:0:10
    clone_weight=linear:1:3:0:10
    mutate_some_weights_weight=linear:1:3:0:10
    mutate_all_weights_weight=linear:1:3:0:10
)
case "$MODE" in
    sho_operators|sho_mutations)
        if [ "$POPULATION" = steady_state ]; then
            echo "error: $MODE tunes operator settings, which EXAQC only allows on islands" >&2
            exit 2
        fi ;;
esac
SHO_ARGS=()
TUNE=()
case "$MODE" in
    sho)
        TUNE=("${TRAINING_TUNE[@]}")
        if [ "$POPULATION" != steady_state ] && [ "$SHO_OPERATORS" = 1 ]; then
            TUNE+=("${CROSSOVER_TUNE[@]}" "${MUTATION_TUNE[@]}")
        fi ;;
    sho_training) TUNE=("${TRAINING_TUNE[@]}") ;;
    sho_operators) TUNE=("${CROSSOVER_TUNE[@]}" "${MUTATION_TUNE[@]}") ;;
    sho_mutations) TUNE=("${MUTATION_TUNE[@]}") ;;
esac
if [ ${#TUNE[@]} -gt 0 ]; then
    SHO_ARGS=(--hyperparameter_strategy simplex --sho_tune "${TUNE[@]}"
              --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5)
fi

# The archive directory carries the same label as the job name that
# submit_uci_all.sh gives this job.
OUT_DIR="$OUT_ROOT/${DATASET}_${MODE}_${TAG}_${RUN}"

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job instead of leaving workers waiting until the time limit. The
# population subcommand comes last: --topology takes any number of values.
COMMAND=(
    "$PYTHON" -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$DATASET"
    --number_genomes "$NUMBER_GENOMES"
    "${DATASET_ARGS[@]}"
    --batch_size 3
    --mutation_strategy uniform "${MUTATIONS[@]}"
    --parent_strategy uniform "${PARENTS[@]}"
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1
    -qim ry -qom probs
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