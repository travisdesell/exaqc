#!/bin/bash
#
# Reproduces one run of the 2026 PPSN classification experiments on this
# machine.
#
#   bash scripts/run_ppsn2026_classification.sh <dataset>
#
# <dataset> is one of the datasets those experiments used: breast_cancer, iris,
# seeds or wine. Each runs with exactly the settings of the original experiment
# script; for other searches use scripts/run_classification.sh. Run i of an
# experiment is reproduced, into
# ./2026_ppsn_exaqc/classification/<name>_i30_per_class_<i> (where <name> is
# "breast" for breast_cancer and the dataset otherwise; BASE_DIR replaces
# ./2026_ppsn_exaqc), with:
#
#   TAG=1 bash scripts/run_ppsn2026_classification.sh iris
#
# and all ten runs of every dataset with:
#
#   for dataset in iris seeds breast_cancer wine; do
#       for i in $(seq 1 10); do TAG=$i bash scripts/run_ppsn2026_classification.sh $dataset; done
#   done
#
# Only the settings that do not change the search are taken from environment
# variables (all optional; defaults below). Note that --restart defaults to
# auto, so running again with the same OUT_DIR continues that run rather than
# starting a new one. Set DRY_RUN to print the command instead of running it.

set -eu

#: MPI processes to launch: one master plus N_PROCESSES - 1 workers that train
#: genomes concurrently. A run needs at least 2.
N_PROCESSES=${N_PROCESSES:-12}

#: Label appended to the output directory as _${TAG}; the experiments used the
#: run index. Letters, digits, - and _ only; unset adds nothing.
TAG=${TAG:-}

#: Directory the experiments' archives are written under, in its
#: classification/ subdirectory.
BASE_DIR=${BASE_DIR:-./2026_ppsn_exaqc}

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

[ $# -eq 1 ] || die "usage: bash $0 <breast_cancer|iris|seeds|wine>"
DATASET=$1

case "$N_PROCESSES" in
    '' | *[!0-9]*) die "N_PROCESSES must be a non-negative integer, but found: $N_PROCESSES" ;;
esac
[ "$N_PROCESSES" -ge 2 ] ||
    die "N_PROCESSES must be at least 2 (one master plus a worker), but found: $N_PROCESSES"

case "$TAG" in
    *[!A-Za-z0-9_-]*) die "TAG must contain only letters, digits, '-' and '_', but found: $TAG" ;;
esac

# Each dataset's qubit counts, and the name its archives were written under.
case "$DATASET" in
    breast_cancer) INPUT_QUBITS=8 OUTPUT_QUBITS=1 NAME=breast ;;
    iris) INPUT_QUBITS=4 OUTPUT_QUBITS=2 NAME=iris ;;
    seeds) INPUT_QUBITS=6 OUTPUT_QUBITS=2 NAME=seeds ;;
    wine) INPUT_QUBITS=4 OUTPUT_QUBITS=2 NAME=wine ;;
    *) die "dataset must be one of: breast_cancer iris seeds wine, but found: $DATASET" ;;
esac

# Where the run's genome archive is written, in the layout the experiments used.
# TAG, when given, is appended to it, or to an OUT_DIR given explicitly.
OUT_DIR=${OUT_DIR:-${BASE_DIR%/}/classification/${NAME}_i30_per_class}
OUT_DIR=${OUT_DIR%/}${TAG:+_$TAG}

read -r -a MPIEXEC_ARGUMENTS <<<"$MPIEXEC_ARGS"

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole run, rather than leaving the other ranks waiting on it; it does not
# change the search. The empty-array expansion is written as ${A[@]+"${A[@]}"}
# because macOS ships bash 3.2, whose `set -u` rejects a bare empty-array
# expansion.
COMMAND=(
    "$MPIEXEC" ${MPIEXEC_ARGUMENTS[@]+"${MPIEXEC_ARGUMENTS[@]}"} -n "$N_PROCESSES"
    "$PYTHON" -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$DATASET"
    --number_genomes 1000
    --input_qubits "$INPUT_QUBITS"
    --output_qubits "$OUTPUT_QUBITS"
    --batch_size 3
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 5 5
    --binary_crossover_rate 0.1
    --n_ary_crossover_rate 0.1
    --exponential_crossover_rate 0.1
    -qim amplitude
    -qom probs
    --encoding identity
    --decoding clipped
    --out_dir "$OUT_DIR"
    steady_state
    --max_population_size 30
)

if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

"${COMMAND[@]}"
