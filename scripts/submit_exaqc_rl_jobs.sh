#!/bin/sh
#
# Submits one Slurm job per EXAQC reinforcement-learning run.
#
#   sh scripts/submit_exaqc_rl_jobs.sh <runs> <env> <input_qubits> <output_qubits> \
#                                      <topology> [topology arguments...]
#
# for example:
#
#   sh scripts/submit_exaqc_rl_jobs.sh 5 walker2d 6 6 2d_mesh 4 5
#
# Each run is scheduled as its own job (there is no loop inside the job script),
# so the runs of an experiment are queued independently and a failure in one
# does not take the others with it.
#
# Everything is validated before anything is submitted: a topology given the
# wrong number of arguments costs an error message rather than a queue full of
# jobs that each die seconds after starting. The topology rules checked here
# mirror src/evolution/topology.py, which remains the real validator -- these
# checks exist to fail fast, not to replace it.
#
# Written for POSIX sh (invoked as `sh ...`), so no arrays and no [[ ]].
#
# Environment variables:
#   N_ISLANDS        islands per run (default 20); the 2d_mesh and random
#                    topologies are validated against it
#   MAX_ISLAND_SIZE  genomes per island (default 5)
#   DRY_RUN          when set, print the sbatch commands instead of submitting

set -eu

#: Where the job script writes Slurm's stdout/stderr, one pair per job.
LOG_DIR=/home/tjdvse/logs/exaqc_test

#: Islands per run. The 2d_mesh and random topologies are defined in terms of
#: this, so it is owned here (rather than in the job script) and passed down:
#: one value validated and used.
N_ISLANDS=${N_ISLANDS:-20}

#: Genomes per island.
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-5}

#: The environments src.examples.reinforcement_learning accepts, from its
#: ENV_IDS mapping. Kept space-delimited for an exact-token match below, so
#: `mountaincar` does not match `mountaincar_continuous`.
ENVIRONMENTS="cartpole acrobot mountaincar mountaincar_continuous frozenlake pendulum hopper walker2d halfcheetah ant humanoid"

#: The topologies src/evolution/topology.py knows how to build.
TOPOLOGIES="fully_connected ring star 2d_mesh tree random"

usage() {
    cat >&2 <<'USAGE'
usage: sh scripts/submit_exaqc_rl_jobs.sh <runs> <env> <input_qubits> <output_qubits> <topology> [topology arguments...]

  runs           number of runs to submit for this experiment, one job each
  env            reinforcement-learning environment to evolve for
  input_qubits   qubits the encoder feeds the circuit
  output_qubits  qubits the circuit hands the decoder
  topology       how the islands are connected, with its own arguments:

                   fully_connected                 (no arguments)
                   ring                            (no arguments)
                   star                            (no arguments)
                   2d_mesh <x_dim> <y_dim>         x_dim * y_dim must equal the
                                                   island count
                   tree <n_children>
                   random <min_edges> <max_edges>  1 <= min_edges <= max_edges,
                                                   max_edges < the island count

examples:
  sh scripts/submit_exaqc_rl_jobs.sh 5 walker2d 6 6 2d_mesh 4 5
  sh scripts/submit_exaqc_rl_jobs.sh 5 walker2d 6 6 ring
  N_ISLANDS=30 sh scripts/submit_exaqc_rl_jobs.sh 3 hopper 6 3 tree 2
USAGE
    exit 2
}

die() {
    printf 'error: %s\n\n' "$*" >&2
    usage
}

# A whole number, as src/evolution/topology.py's own argument check requires.
is_whole_number() {
    case "$1" in
        '' | *[!0-9]*) return 1 ;;
    esac
    return 0
}

contains_word() {
    # $1 is the space-delimited list, $2 the word; the padding makes this an
    # exact-token match rather than a substring one.
    case " $1 " in
        *" $2 "*) return 0 ;;
    esac
    return 1
}

[ $# -ge 5 ] || usage

RUNS=$1
ENVIRONMENT=$2
INPUT_QUBITS=$3
OUTPUT_QUBITS=$4
TOPOLOGY=$5
shift 5
# whatever is left is the topology's own arguments

is_whole_number "$RUNS" && [ "$RUNS" -gt 0 ] ||
    die "runs must be a positive integer, but found: $RUNS"

contains_word "$ENVIRONMENTS" "$ENVIRONMENT" ||
    die "unknown environment: $ENVIRONMENT (choose one of: $ENVIRONMENTS)"

is_whole_number "$INPUT_QUBITS" && [ "$INPUT_QUBITS" -gt 0 ] ||
    die "input_qubits must be a positive integer, but found: $INPUT_QUBITS"

is_whole_number "$OUTPUT_QUBITS" && [ "$OUTPUT_QUBITS" -gt 0 ] ||
    die "output_qubits must be a positive integer, but found: $OUTPUT_QUBITS"

contains_word "$TOPOLOGIES" "$TOPOLOGY" ||
    die "unknown topology: $TOPOLOGY (choose one of: $TOPOLOGIES)"

# Each topology takes its own arguments, and a wrong count is the mistake this
# script exists to catch before a job is queued. The rules, and their wording,
# follow src/evolution/topology.py.
case "$TOPOLOGY" in
    fully_connected | ring | star)
        [ $# -eq 0 ] ||
            die "$TOPOLOGY topology takes no additional arguments, but found $#: $*"
        TOPOLOGY_TAG=$TOPOLOGY
        ;;

    2d_mesh)
        [ $# -eq 2 ] ||
            die "2d_mesh topology requires 2 additional argument(s): <x_dim> <y_dim>"
        is_whole_number "$1" ||
            die "2d_mesh topology requires x_dim to be a non-negative integer, but found x_dim: $1"
        is_whole_number "$2" ||
            die "2d_mesh topology requires y_dim to be a non-negative integer, but found y_dim: $2"
        [ $(($1 * $2)) -eq "$N_ISLANDS" ] ||
            die "2d_mesh requires x_dim ($1) * y_dim ($2) == n_islands ($N_ISLANDS)"
        TOPOLOGY_TAG="${TOPOLOGY}_${1}x${2}"
        ;;

    tree)
        [ $# -eq 1 ] ||
            die "tree topology requires 1 additional argument(s): <n_children>"
        is_whole_number "$1" ||
            die "tree topology requires n_children to be a non-negative integer, but found n_children: $1"
        TOPOLOGY_TAG="${TOPOLOGY}_${1}"
        ;;

    random)
        [ $# -eq 2 ] ||
            die "random topology requires 2 additional argument(s): <min_edges> <max_edges>"
        is_whole_number "$1" ||
            die "random topology requires min_edges to be a non-negative integer, but found min_edges: $1"
        is_whole_number "$2" ||
            die "random topology requires max_edges to be a non-negative integer, but found max_edges: $2"
        [ "$1" -ge 1 ] ||
            die "random requires min_edges ($1) >= 1"
        [ "$2" -ge "$1" ] ||
            die "random requires max_edges ($2) >= min_edges ($1)"
        [ "$2" -lt "$N_ISLANDS" ] ||
            die "random requires max_edges ($2) < n_islands ($N_ISLANDS)"
        TOPOLOGY_TAG="${TOPOLOGY}_${1}_${2}"
        ;;
esac

# The topology's arguments are validated whole numbers, so they are kept as a
# plain string and split on whitespace where they are used. Holding them in the
# positional parameters instead would mean rebuilding those parameters on every
# iteration of the loop below, which is easy to get subtly wrong.
TOPOLOGY_ARGUMENTS="$*"

SCRIPT_DIR=$(dirname "$0")
JOB_SCRIPT="$SCRIPT_DIR/exaqc_rl_job.sh"

[ -f "$JOB_SCRIPT" ] ||
    die "the job script is missing: $JOB_SCRIPT"

if [ -z "${DRY_RUN:-}" ]; then
    command -v sbatch >/dev/null 2>&1 ||
        die "sbatch is not available, so these jobs cannot be submitted from here"
    mkdir -p "$LOG_DIR"
fi

# The runs of an experiment differ only by their index, which separates their
# archives. Note that a run whose archive already exists is *continued* rather
# than restarted from scratch, since --restart defaults to auto: submitting the
# same experiment twice resumes it instead of starting over.
run=1
while [ "$run" -le "$RUNS" ]; do
    name="exaqc_${ENVIRONMENT}_${TOPOLOGY_TAG}_${run}"

    # The topology and its arguments go last, and stay last all the way into the
    # search: --topology takes any number of values, so anything after it would
    # be swallowed as one of them. TOPOLOGY_ARGUMENTS is deliberately unquoted,
    # so its whole numbers become separate arguments.
    set -- sbatch \
        -J "$name" \
        -o "${LOG_DIR}/${name}_%j.o" \
        -e "${LOG_DIR}/${name}_%j.e" \
        "$JOB_SCRIPT" \
        "$ENVIRONMENT" \
        "$INPUT_QUBITS" \
        "$OUTPUT_QUBITS" \
        "$run" \
        "$N_ISLANDS" \
        "$MAX_ISLAND_SIZE" \
        "$TOPOLOGY_TAG" \
        "$TOPOLOGY" \
        $TOPOLOGY_ARGUMENTS

    if [ -n "${DRY_RUN:-}" ]; then
        printf '%s\n' "$*"
    else
        "$@"
    fi

    run=$((run + 1))
done
