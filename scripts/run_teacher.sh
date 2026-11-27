#!/bin/bash
#
# Runs a single EXAQC quantum-teacher imitation search on this machine.
#
#   bash scripts/run_teacher.sh
#
# The teacher counterpart to scripts/run_classification.sh and
# scripts/run_rl.sh: launched with the local mpiexec and configured entirely
# through environment variables (all optional; defaults below), for example:
#
#   TEACHER=bell_out INPUT_QUBITS=1 OUTPUT_QUBITS=2 bash scripts/run_teacher.sh
#   POPULATION=steady_state_speciation SPECIES_THRESHOLD=0.6 bash scripts/run_teacher.sh
#   N_PROCESSES=8 EPOCHS=10 bash scripts/run_teacher.sh
#   TAG=run2 bash scripts/run_teacher.sh
#
# Note that --restart defaults to auto, so running again with the same OUT_DIR
# continues that run rather than starting a new one. Set DRY_RUN to print the
# command instead of running it.

set -eu

#: MPI processes to launch: one master plus N_PROCESSES - 1 workers that train
#: genomes concurrently. A run needs at least 2.
N_PROCESSES=${N_PROCESSES:-12}

#: Reference circuit to imitate (--teacher).
TEACHER=${TEACHER:-half_adder}

#: Teachers this script accepts (matches src.circuits.teacher_circuits.TEACHER_NAMES).
TEACHERS="identity x_out4 bell_out copy_in_to_out parity012_to_out4 input_controlled_bell 2layer_out_block grover half_adder"

#: Input wires, then readout wires after them (--input_qubits / --output_qubits).
INPUT_QUBITS=${INPUT_QUBITS:-2}
OUTPUT_QUBITS=${OUTPUT_QUBITS:-2}

#: How inputs enter, and outputs leave, the circuit (-qim / -qom). Teacher
#: genomes are purely quantum, so -qim is one of rx/ry/rz.
QIM=${QIM:-ry}
QOM=${QOM:-probs}

#: Loss optimized during training (--loss).
LOSS=${LOSS:-fidelity}

#: The population strategy: islands, steady_state, or steady_state_speciation.
POPULATION=${POPULATION:-islands}

#: Islands, and genomes per island (POPULATION=islands).
N_ISLANDS=${N_ISLANDS:-25}
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-5}

#: How the islands are connected, followed by the topology's own arguments, as
#: one quoted value, e.g. TOPOLOGY="2d_mesh 4 5" or TOPOLOGY="random 1 3".
#: src/evolution/topology.py validates the arguments.
TOPOLOGY=${TOPOLOGY:-2d_mesh 5 5}

#: Genomes the population keeps (POPULATION=steady_state or
#: steady_state_speciation).
MAX_POPULATION_SIZE=${MAX_POPULATION_SIZE:-30}

#: Compatibility-distance join threshold (POPULATION=steady_state_speciation).
SPECIES_THRESHOLD=${SPECIES_THRESHOLD:-0.6}

#: Training epochs per genome.
EPOCHS=${EPOCHS:-30}

#: Epochs without an improved validation loss before training stops early.
IMPROVEMENT_CUTOFF=${IMPROVEMENT_CUTOFF:-3}

#: Training batch size.
BATCH_SIZE=${BATCH_SIZE:-8}

#: Generated training and validation samples.
N_TRAINING_SAMPLES=${N_TRAINING_SAMPLES:-64}
N_VALIDATION_SAMPLES=${N_VALIDATION_SAMPLES:-64}

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

# Fails unless $2 is a finite number greater than 0, naming it $1.
# Matches SteadyStateSpeciation: species_threshold must be > 0.
require_positive_float() {
    case "$2" in
        '' | *[!0-9.]* | *.*.*) die "$1 must be a positive number, but found: $2" ;;
    esac
    awk -v t="$2" 'BEGIN { exit !(t > 0) }' ||
        die "$1 must be a positive number (> 0), but found: $2"
}

# Fails unless $2 is one of the space-separated words in $3, naming it $1.
require_choice() {
    case " $3 " in
        *" $2 "*) ;;
        *) die "$1 must be one of: $3, but found: $2" ;;
    esac
}

require_choice TEACHER "$TEACHER" "$TEACHERS"
require_whole_number N_PROCESSES "$N_PROCESSES"
[ "$N_PROCESSES" -ge 2 ] ||
    die "N_PROCESSES must be at least 2 (one master plus a worker), but found: $N_PROCESSES"
require_positive INPUT_QUBITS "$INPUT_QUBITS"
require_positive OUTPUT_QUBITS "$OUTPUT_QUBITS"
require_choice QIM "$QIM" "rx ry rz"
require_choice QOM "$QOM" "probs expval"
require_choice LOSS "$LOSS" "fidelity angle kl mse"
require_choice POPULATION "$POPULATION" "islands steady_state steady_state_speciation"
require_positive EPOCHS "$EPOCHS"
require_whole_number IMPROVEMENT_CUTOFF "$IMPROVEMENT_CUTOFF"
require_positive BATCH_SIZE "$BATCH_SIZE"
require_positive N_TRAINING_SAMPLES "$N_TRAINING_SAMPLES"
require_positive N_VALIDATION_SAMPLES "$N_VALIDATION_SAMPLES"
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
    # e.g. ./exaqc_runs/half_adder_2d_mesh_5_5_i25
    DEFAULT_OUT_DIR=${BASE_DIR%/}/${TEACHER}_${TOPOLOGY_TAG}_i${N_ISLANDS}
elif [ "$POPULATION" = steady_state_speciation ]; then
    require_positive MAX_POPULATION_SIZE "$MAX_POPULATION_SIZE"
    require_positive_float SPECIES_THRESHOLD "$SPECIES_THRESHOLD"
    POPULATION_ARGUMENTS=(
        steady_state_speciation
        --max_population_size "$MAX_POPULATION_SIZE"
        --species_threshold "$SPECIES_THRESHOLD"
    )
    # e.g. ./exaqc_runs/half_adder_speciation_p30_t0.6
    DEFAULT_OUT_DIR=${BASE_DIR%/}/${TEACHER}_speciation_p${MAX_POPULATION_SIZE}_t${SPECIES_THRESHOLD}
else
    require_positive MAX_POPULATION_SIZE "$MAX_POPULATION_SIZE"
    POPULATION_ARGUMENTS=(steady_state --max_population_size "$MAX_POPULATION_SIZE")
    # e.g. ./exaqc_runs/half_adder_steady_state_p30
    DEFAULT_OUT_DIR=${BASE_DIR%/}/${TEACHER}_steady_state_p${MAX_POPULATION_SIZE}
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
    "$PYTHON" -m mpi4py -m src.examples.teacher
    --logging_level INFO
    --teacher "$TEACHER"
    --input_qubits "$INPUT_QUBITS"
    --output_qubits "$OUTPUT_QUBITS"
    --number_genomes "$NUMBER_GENOMES"
    --loss "$LOSS"
    --epochs "$EPOCHS"
    --improvement_cutoff "$IMPROVEMENT_CUTOFF"
    --batch_size "$BATCH_SIZE"
    --n_training_samples "$N_TRAINING_SAMPLES"
    --n_validation_samples "$N_VALIDATION_SAMPLES"
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 5 5
    --binary_crossover_rate 0.1
    --n_ary_crossover_rate 0.1
    --exponential_crossover_rate 0.1
    -qim "$QIM"
    -qom "$QOM"
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
