#!/bin/bash
#
# Runs a single EXAQC classification search on this machine.
#
#   bash scripts/run_classification.sh
#
# The classification counterpart to scripts/run_rl.sh, for the non-vision
# datasets (breast_cancer, iris, seeds, wine), launched with the local mpiexec
# and configured entirely through environment variables (all optional; defaults
# below), for example:
#
#   DATASET=wine TOPOLOGY="2d_mesh 4 5" N_ISLANDS=20 bash scripts/run_classification.sh
#   DATASET=breast_cancer N_PROCESSES=8 EPOCHS=10 bash scripts/run_classification.sh
#   USE_ONLY="rx ry rz cx cy cz" OUT_DIR=./exaqc_runs/rxyz_cxyz bash scripts/run_classification.sh
#   BASE_DIR=~/Data/exaqc_runs bash scripts/run_classification.sh
#   POPULATION=steady_state MAX_POPULATION_SIZE=30 bash scripts/run_classification.sh
#   TAG=run2 bash scripts/run_classification.sh
#
# The 2026 PPSN runs are reproduced by scripts/run_ppsn2026_classification.sh
# instead.
#
# Note that --restart defaults to auto, so running again with the same OUT_DIR
# continues that run rather than starting a new one. Set DRY_RUN to print the
# command instead of running it.

set -eu

#: MPI processes to launch: one master plus N_PROCESSES - 1 workers that train
#: genomes concurrently. A run needs at least 2.
N_PROCESSES=${N_PROCESSES:-12}

#: Dataset to classify (--dataset); one of the non-vision datasets below.
DATASET=${DATASET:-iris}

#: The datasets this script runs. The vision datasets (mnist, fashion_mnist,
#: cifar10) need a CNN encoder and are left to their own scripts.
DATASETS="breast_cancer iris seeds wine"

#: Qubits the encoder feeds the circuit, and the circuit hands the decoder.
INPUT_QUBITS=${INPUT_QUBITS:-5}
OUTPUT_QUBITS=${OUTPUT_QUBITS:-5}

#: How inputs enter, and outputs leave, the circuit (-qim / -qom).
QIM=${QIM:-u3}
QOM=${QOM:-probs}

#: The classical encoder and decoder around the circuit (--encoding / --decoding).
ENCODING=${ENCODING:-linear}
DECODING=${DECODING:-linear}

#: The population strategy: islands or steady_state.
POPULATION=${POPULATION:-islands}

#: Islands, and genomes per island (POPULATION=islands).
N_ISLANDS=${N_ISLANDS:-25}
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-5}

#: How the islands are connected, followed by the topology's own arguments, as
#: one quoted value, e.g. TOPOLOGY="2d_mesh 4 5" or TOPOLOGY="random 1 3".
#: src/evolution/topology.py validates the arguments.
TOPOLOGY=${TOPOLOGY:-2d_mesh 5 5}

#: Genomes the population keeps (POPULATION=steady_state).
MAX_POPULATION_SIZE=${MAX_POPULATION_SIZE:-30}

#: Training epochs per genome.
EPOCHS=${EPOCHS:-30}

#: Epochs without an improved validation loss before training stops early.
IMPROVEMENT_CUTOFF=${IMPROVEMENT_CUTOFF:-3}

#: Training batch size.
BATCH_SIZE=${BATCH_SIZE:-3}

#: Genomes the run evaluates.
NUMBER_GENOMES=${NUMBER_GENOMES:-1000}

#: The gates the search may use, as one quoted, space-separated value.
USE_ONLY=${USE_ONLY:-rx ry rz p u crx cry crz cp rxx ryy rzz cx cy cz}

#: Optional label appended to the output directory as _${TAG}, so runs that
#: differ only in settings the directory name does not show stay apart. Letters,
#: digits, - and _ only; unset adds nothing.
TAG=${TAG:-}

#: Directory the default output directory is created in.
BASE_DIR=${BASE_DIR:-./exaqc_runs}

#: Python and MPI launcher to use. MPIEXEC_ARGS defaults to Open MPI's
#: --oversubscribe so N_PROCESSES may exceed the core count; clear it for an MPI
#: that does not accept that flag.
PYTHON=${PYTHON:-python3}
MPIEXEC=${MPIEXEC:-mpiexec}
MPIEXEC_ARGS=${MPIEXEC_ARGS---oversubscribe}

# Prints an error and exits.
die() {
    echo "error: $*" >&2
    exit 2
}

# Fails unless $2 is a whole number, naming it $1.
require_whole_number() {
    case "$2" in
        '' | *[!0-9]*) die "$1 must be a non-negative integer, but found: $2" ;;
    esac
}

# Fails unless $2 is a whole number greater than 0, naming it $1.
require_positive() {
    require_whole_number "$1" "$2"
    [ "$2" -gt 0 ] || die "$1 must be a positive integer, but found: $2"
}

# Fails unless $2 is one of the space-separated words in $3, naming it $1.
require_choice() {
    case " $3 " in
        *" $2 "*) ;;
        *) die "$1 must be one of: $3, but found: $2" ;;
    esac
}

require_choice DATASET "$DATASET" "$DATASETS"
require_whole_number N_PROCESSES "$N_PROCESSES"
[ "$N_PROCESSES" -ge 2 ] ||
    die "N_PROCESSES must be at least 2 (one master plus a worker), but found: $N_PROCESSES"
require_positive INPUT_QUBITS "$INPUT_QUBITS"
require_positive OUTPUT_QUBITS "$OUTPUT_QUBITS"
require_choice QIM "$QIM" "u3 rx ry rz basis amplitude"
require_choice QOM "$QOM" "probs expval"
require_choice ENCODING "$ENCODING" "identity linear"
require_choice DECODING "$DECODING" "clipped linear"
require_choice POPULATION "$POPULATION" "islands steady_state"
require_positive EPOCHS "$EPOCHS"
require_whole_number IMPROVEMENT_CUTOFF "$IMPROVEMENT_CUTOFF"
require_positive BATCH_SIZE "$BATCH_SIZE"
require_positive NUMBER_GENOMES "$NUMBER_GENOMES"

case "$USE_ONLY" in
    *[!a-z0-9\ ]*) die "USE_ONLY must be space-separated gate names (lowercase letters and digits), but found: $USE_ONLY" ;;
    *[a-z0-9]*) ;;
    *) die "USE_ONLY must name at least one gate, but found: '$USE_ONLY'" ;;
esac
read -r -a USE_ONLY_GATES <<<"$USE_ONLY"

case "$TAG" in
    *[!A-Za-z0-9_-]*) die "TAG must contain only letters, digits, '-' and '_', but found: $TAG" ;;
esac

# The population sub-command and its arguments, and a default output directory
# built from the settings that separate one experiment from another, joined with
# underscores.
if [ "$POPULATION" = islands ]; then
    require_positive N_ISLANDS "$N_ISLANDS"
    require_positive MAX_ISLAND_SIZE "$MAX_ISLAND_SIZE"
    case "$TOPOLOGY" in
        *[!a-z0-9_\ ]*) die "TOPOLOGY must be a topology name and its whole-number arguments, but found: $TOPOLOGY" ;;
    esac
    read -r -a TOPOLOGY_ARGUMENTS <<<"$TOPOLOGY"
    [ ${#TOPOLOGY_ARGUMENTS[@]} -gt 0 ] || die "TOPOLOGY must name a topology"

    POPULATION_ARGUMENTS=(
        islands
        --n_islands "$N_ISLANDS"
        --max_island_size "$MAX_ISLAND_SIZE"
        --islands_to_extinct 1
        --topology "${TOPOLOGY_ARGUMENTS[@]}"
    )
    TOPOLOGY_TAG=$(
        IFS=_
        echo "${TOPOLOGY_ARGUMENTS[*]}"
    )
    # e.g. ./exaqc_runs/iris_2d_mesh_5_5_i25
    DEFAULT_OUT_DIR=${BASE_DIR%/}/${DATASET}_${TOPOLOGY_TAG}_i${N_ISLANDS}
else
    require_positive MAX_POPULATION_SIZE "$MAX_POPULATION_SIZE"
    POPULATION_ARGUMENTS=(steady_state --max_population_size "$MAX_POPULATION_SIZE")
    # e.g. ./exaqc_runs/iris_steady_state_p30
    DEFAULT_OUT_DIR=${BASE_DIR%/}/${DATASET}_steady_state_p${MAX_POPULATION_SIZE}
fi

# Where the run's genome archive is written. TAG, when given, is appended to it,
# or to an OUT_DIR given explicitly.
OUT_DIR=${OUT_DIR:-$DEFAULT_OUT_DIR}
OUT_DIR=${OUT_DIR%/}${TAG:+_$TAG}

read -r -a MPIEXEC_ARGUMENTS <<<"$MPIEXEC_ARGS"

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole run, rather than leaving the other ranks waiting on it. The population
# sub-command comes last: --topology takes any number of values, so anything
# after it would be read as one of them. The empty-array expansion is written as
# ${A[@]+"${A[@]}"} because macOS ships bash 3.2, whose `set -u` rejects a bare
# empty-array expansion.
COMMAND=(
    "$MPIEXEC" ${MPIEXEC_ARGUMENTS[@]+"${MPIEXEC_ARGUMENTS[@]}"} -n "$N_PROCESSES"
    "$PYTHON" -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$DATASET"
    --input_qubits "$INPUT_QUBITS"
    --output_qubits "$OUTPUT_QUBITS"
    --number_genomes "$NUMBER_GENOMES"
    --epochs "$EPOCHS"
    --improvement_cutoff "$IMPROVEMENT_CUTOFF"
    --batch_size "$BATCH_SIZE"
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 2 5
    --binary_crossover_rate 0.0
    --n_ary_crossover_rate 0.2
    --exponential_crossover_rate 0.1
    -qim "$QIM"
    -qom "$QOM"
    --encoding "$ENCODING"
    --decoding "$DECODING"
    # --use_only takes any number of values, so another flag must follow it
    --use_only "${USE_ONLY_GATES[@]}"
    --out_dir "$OUT_DIR"
    "${POPULATION_ARGUMENTS[@]}"
)

if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

"${COMMAND[@]}"
