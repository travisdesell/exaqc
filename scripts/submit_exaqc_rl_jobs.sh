#!/bin/sh
#
# Submits one Slurm job per EXAQC reinforcement-learning run.
#
#   sh scripts/submit_exaqc_rl_jobs.sh <min_run> <max_run> <tag> <env> \
#                                      <input_qubits> <output_qubits> \
#                                      <topology> [topology arguments...]
#
# for example:
#
#   sh scripts/submit_exaqc_rl_jobs.sh 1 5 healthy02 walker2d 6 6 2d_mesh 4 5
#
# The run range is inclusive, so a finished experiment can be extended with more
# runs later without resubmitting the ones already done.
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
#   HEALTHY_REWARD   per-step upright bonus for the MuJoCo environments that
#                    have one (hopper, walker2d, ant, humanoid); unset leaves
#                    each environment's own Gymnasium value. Read by the job
#                    script, which sbatch reaches through its default
#                    environment export -- so set it on this command line:
#                      HEALTHY_REWARD=0.2 sh scripts/submit_exaqc_rl_jobs.sh ...
#                    Giving it for an environment with no alive bonus is an
#                    error rather than ignored, so tag such runs accordingly
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
usage: sh scripts/submit_exaqc_rl_jobs.sh <min_run> <max_run> <tag> <env> <input_qubits> <output_qubits> <topology> [topology arguments...]

  min_run        first run index to submit (inclusive)
  max_run        last run index to submit (inclusive); one job per index
  tag            keyword distinguishing this run type from others with the
                 same environment and topology (letters, digits, - and _).
                 It names both the job and its archive directory, so two run
                 types cannot resume one another's archives
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
  sh scripts/submit_exaqc_rl_jobs.sh 1 5 healthy02 walker2d 6 6 2d_mesh 4 5
  sh scripts/submit_exaqc_rl_jobs.sh 1 5 baseline walker2d 6 6 ring
  sh scripts/submit_exaqc_rl_jobs.sh 6 10 baseline walker2d 6 6 ring
  N_ISLANDS=30 sh scripts/submit_exaqc_rl_jobs.sh 1 3 stochastic hopper 6 3 tree 2
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

# A tag is used unquoted in a job name and, more importantly, in an archive
# directory name, so it is restricted to characters that are safe in a path and
# cannot be read as shell syntax.
is_tag() {
    case "$1" in
        '' | *[!A-Za-z0-9_-]*) return 1 ;;
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

[ $# -ge 7 ] || usage

MIN_RUN=$1
MAX_RUN=$2
TAG=$3
ENVIRONMENT=$4
INPUT_QUBITS=$5
OUTPUT_QUBITS=$6
TOPOLOGY=$7
shift 7
# whatever is left is the topology's own arguments

is_whole_number "$MIN_RUN" && [ "$MIN_RUN" -gt 0 ] ||
    die "min_run must be a positive integer, but found: $MIN_RUN"

is_whole_number "$MAX_RUN" && [ "$MAX_RUN" -gt 0 ] ||
    die "max_run must be a positive integer, but found: $MAX_RUN"

[ "$MAX_RUN" -ge "$MIN_RUN" ] ||
    die "max_run ($MAX_RUN) must be >= min_run ($MIN_RUN)"

is_tag "$TAG" ||
    die "tag must be non-empty and contain only letters, digits, '-' and '_', but found: $TAG"

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

# What separates one experiment's archives from another's. The tag comes first
# so runs of the same topology but a different run type (a changed reward, a
# different evaluation regime) sort together yet never share a directory: the
# job script builds its --out_dir from this, and --restart defaults to auto, so
# a shared directory would silently resume the wrong experiment.
RUN_TAG="${TAG}_${TOPOLOGY_TAG}"

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
# same experiment twice resumes it instead of starting over. That is what makes
# an inclusive range useful -- runs 6..10 can be added to an experiment later
# without touching runs 1..5.
run=$MIN_RUN
while [ "$run" -le "$MAX_RUN" ]; do
    # This single string names the Slurm job, its two log files and the archive
    # directory the run writes, so all four agree. The island count is part of
    # it because two island counts of the same experiment are different
    # experiments and must not share an archive.
    name="exaqc_${RUN_TAG}_${ENVIRONMENT}_i${N_ISLANDS}_${run}"

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
        "$name" \
        "$N_ISLANDS" \
        "$MAX_ISLAND_SIZE" \
        "$TOPOLOGY" \
        $TOPOLOGY_ARGUMENTS

    if [ -n "${DRY_RUN:-}" ]; then
        printf '%s\n' "$*"
    else
        "$@"
    fi

    run=$((run + 1))
done
