#!/bin/sh
#
# Submits one Slurm job per run of the UCI (tabular) classification batch, on connected islands.
#
#   sh scripts/submit_exaqc_uci_islands_sho_jobs.sh <min_run> <max_run> <tag> \
#                                     <topology> [topology arguments...]
#
# for example:
#
#   sh scripts/submit_exaqc_uci_islands_sho_jobs.sh 1 10 sho ring
#   NAMES=iris sh scripts/submit_exaqc_uci_islands_sho_jobs.sh 1 10 sho 2d_mesh 2 3
#
# Every dataset (iris seeds breast_cancer wine) is submitted unless NAMES picks some.
# The island count, island size and topology are part of each run's job and
# archive name, so runs on different topologies never share an archive. Run i is
# seeded with --seed i, so runs are independent and reproducible. The run range
# is inclusive, so a finished experiment can be extended with more runs later
# without resubmitting the ones already done.
#
# Each run is scheduled as its own job (there is no loop inside the job script),
# so the runs are queued independently and a failure in one does not take the
# others with it. Everything is validated before anything is submitted.
#
# Written for POSIX sh (invoked as `sh ...`), so no arrays and no [[ ]].
#
# Environment variables:
#   NAMES               datasets to submit (default: all of them)
#   N_ISLANDS           islands per run (default 6); the 2d_mesh and
#                       random topologies are validated against it
#   MAX_ISLAND_SIZE     genomes per island (default 5, so 30 genomes
#                       in all, matching the steady-state batch)
#   ISLANDS_TO_EXTINCT  worst islands cleared per extinction event (default 1)
#   TIME_LIMIT  Slurm time limit per job (default: the job script's 5-00:00:00)
#   NTASKS      MPI processes per job (default: the job script's 144)
#   VENV, ARCHIVE_DIR
#               read by the job script, which sbatch reaches through its
#               default environment export -- so set them on this command line
#   DRY_RUN     when set, print the sbatch commands instead of submitting

set -eu

#: Islands per run, genomes per island and islands cleared per extinction.
#: The defaults keep the 30-genome population of the steady-state batch.
N_ISLANDS=${N_ISLANDS:-6}
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-20}
ISLANDS_TO_EXTINCT=${ISLANDS_TO_EXTINCT:-1}

#: The topologies src/evolution/topology.py knows how to build.
TOPOLOGIES="fully_connected ring star 2d_mesh tree random"

#: Where the job script writes Slurm's stdout/stderr, one pair per job.
LOG_DIR=${LOG_DIR:-/home/dk7405/logs/exaqc}

#: The datasets the job script accepts. Kept space-delimited for an
#: exact-token match below.
NAMES_ALLOWED="iris seeds breast_cancer wine"

usage() {
    cat >&2 <<'USAGE'
usage: sh scripts/submit_exaqc_uci_islands_sho_jobs.sh <min_run> <max_run> <tag> <topology> [topology arguments...]

  min_run   first run index to submit (inclusive); also its --seed
  max_run   last run index to submit (inclusive); one job per index and dataset
  tag       keyword distinguishing this run type from others (letters, digits,
            - and _). It names both the job and its archive directory
  topology  how the islands are connected, with its own arguments:

              fully_connected                 (no arguments)
              ring                            (no arguments)
              star                            (no arguments)
              2d_mesh <x_dim> <y_dim>         x_dim * y_dim must equal N_ISLANDS
              tree <n_children>
              random <min_edges> <max_edges>  1 <= min_edges <= max_edges,
                                              max_edges < N_ISLANDS

  NAMES picks the datasets (default: all of iris seeds breast_cancer wine).

examples:
  sh scripts/submit_exaqc_uci_islands_sho_jobs.sh 1 10 sho ring
  sh scripts/submit_exaqc_uci_islands_sho_jobs.sh 1 10 sho 2d_mesh 2 3
  NAMES=iris N_ISLANDS=12 sh scripts/submit_exaqc_uci_islands_sho_jobs.sh 1 10 sho random 1 3
USAGE
    exit 2
}

die() {
    printf 'error: %s\n\n' "$*" >&2
    usage
}

is_whole_number() {
    case "$1" in
        '' | *[!0-9]*) return 1 ;;
    esac
    return 0
}

# A tag is used unquoted in a job name and an archive directory name, so it is
# restricted to characters that are safe in a path.
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

[ $# -ge 4 ] || usage

MIN_RUN=$1
MAX_RUN=$2
TAG=$3
TOPOLOGY=$4
shift 4
# whatever is left is the topology's own arguments

is_whole_number "$MIN_RUN" && [ "$MIN_RUN" -gt 0 ] ||
    die "min_run must be a positive integer, but found: $MIN_RUN"

is_whole_number "$MAX_RUN" && [ "$MAX_RUN" -gt 0 ] ||
    die "max_run must be a positive integer, but found: $MAX_RUN"

[ "$MAX_RUN" -ge "$MIN_RUN" ] ||
    die "max_run ($MAX_RUN) must be >= min_run ($MIN_RUN)"

is_tag "$TAG" ||
    die "tag must be non-empty and contain only letters, digits, '-' and '_', but found: $TAG"

if [ -z "${NAMES:-}" ]; then
    NAMES=$NAMES_ALLOWED
else
    for name in $NAMES; do
        contains_word "$NAMES_ALLOWED" "$name" ||
            die "unknown dataset in NAMES: $name (choose from: $NAMES_ALLOWED)"
    done
fi

for value in "$N_ISLANDS" "$MAX_ISLAND_SIZE"; do
    is_whole_number "$value" && [ "$value" -gt 0 ] ||
        die "N_ISLANDS and MAX_ISLAND_SIZE must be positive integers, but found: $value"
done

is_whole_number "$ISLANDS_TO_EXTINCT" && [ "$ISLANDS_TO_EXTINCT" -lt "$N_ISLANDS" ] ||
    die "ISLANDS_TO_EXTINCT must be a whole number below N_ISLANDS ($N_ISLANDS), but found: $ISLANDS_TO_EXTINCT"

contains_word "$TOPOLOGIES" "$TOPOLOGY" ||
    die "unknown topology: $TOPOLOGY (choose one of: $TOPOLOGIES)"

# Each topology takes its own arguments, and a wrong count is the mistake this
# script exists to catch before a job is queued. The rules follow
# src/evolution/topology.py, which remains the real validator. TOPOLOGY_TAG is
# the topology as it appears in the job and archive names.
case "$TOPOLOGY" in
    fully_connected | ring | star)
        [ $# -eq 0 ] ||
            die "$TOPOLOGY topology takes no additional arguments, but found $#: $*"
        TOPOLOGY_TAG=$TOPOLOGY
        ;;

    2d_mesh)
        [ $# -eq 2 ] ||
            die "2d_mesh topology requires 2 additional argument(s): <x_dim> <y_dim>"
        is_whole_number "$1" && is_whole_number "$2" ||
            die "2d_mesh topology requires whole-number dimensions, but found: $1 $2"
        [ $(($1 * $2)) -eq "$N_ISLANDS" ] ||
            die "2d_mesh requires x_dim ($1) * y_dim ($2) == N_ISLANDS ($N_ISLANDS)"
        TOPOLOGY_TAG="${TOPOLOGY}_${1}x${2}"
        ;;

    tree)
        [ $# -eq 1 ] ||
            die "tree topology requires 1 additional argument(s): <n_children>"
        is_whole_number "$1" ||
            die "tree topology requires n_children to be a whole number, but found: $1"
        TOPOLOGY_TAG="${TOPOLOGY}_${1}"
        ;;

    random)
        [ $# -eq 2 ] ||
            die "random topology requires 2 additional argument(s): <min_edges> <max_edges>"
        is_whole_number "$1" && is_whole_number "$2" ||
            die "random topology requires whole-number edge counts, but found: $1 $2"
        [ "$1" -ge 1 ] ||
            die "random requires min_edges ($1) >= 1"
        [ "$2" -ge "$1" ] ||
            die "random requires max_edges ($2) >= min_edges ($1)"
        [ "$2" -lt "$N_ISLANDS" ] ||
            die "random requires max_edges ($2) < N_ISLANDS ($N_ISLANDS)"
        TOPOLOGY_TAG="${TOPOLOGY}_${1}_${2}"
        ;;
esac

# The topology's arguments are validated whole numbers, so they are kept as a
# plain string and split on whitespace where they are used.
TOPOLOGY_ARGUMENTS="$*"

if [ -n "${NTASKS:-}" ]; then
    is_whole_number "$NTASKS" && [ "$NTASKS" -gt 0 ] ||
        die "NTASKS must be a positive integer, but found: $NTASKS"
fi

SCRIPT_DIR=$(dirname "$0")
JOB_SCRIPT="$SCRIPT_DIR/exaqc_uci_islands_sho_job.sh"

[ -f "$JOB_SCRIPT" ] ||
    die "the job script is missing: $JOB_SCRIPT"

if [ -z "${DRY_RUN:-}" ]; then
    command -v sbatch >/dev/null 2>&1 ||
        die "sbatch is not available, so these jobs cannot be submitted from here"
    mkdir -p "$LOG_DIR"
fi

# Optional resource overrides, kept as a plain string and split where used.
RESOURCE_ARGUMENTS=""
[ -n "${TIME_LIMIT:-}" ] && RESOURCE_ARGUMENTS="$RESOURCE_ARGUMENTS -t $TIME_LIMIT"
[ -n "${NTASKS:-}" ] && RESOURCE_ARGUMENTS="$RESOURCE_ARGUMENTS --ntasks=$NTASKS --ntasks-per-node=$NTASKS"

# A run whose archive already exists is *continued* rather than restarted from
# scratch, since --restart defaults to auto: submitting the same experiment twice
# resumes it instead of starting over.
for name in $NAMES; do
    run=$MIN_RUN
    while [ "$run" -le "$MAX_RUN" ]; do
        # This single string names the Slurm job, its two log files and the
        # archive directory the run writes, so all four agree. It carries the
        # island count and size and the topology, since runs that differ in
        # any of them are different experiments.
        job_name="${name}_exaqc_${TAG}_uci_i${N_ISLANDS}_p${MAX_ISLAND_SIZE}_${TOPOLOGY_TAG}_${run}"

        # RESOURCE_ARGUMENTS and TOPOLOGY_ARGUMENTS are deliberately unquoted,
        # so each flag or number is its own argument. The topology goes last.
        set -- sbatch \
            -J "$job_name" \
            -o "${LOG_DIR}/${job_name}_%j.o" \
            -e "${LOG_DIR}/${job_name}_%j.e" \
            $RESOURCE_ARGUMENTS \
            "$JOB_SCRIPT" \
            "$name" \
            "$run" \
            "$job_name" \
            "$N_ISLANDS" \
            "$MAX_ISLAND_SIZE" \
            "$ISLANDS_TO_EXTINCT" \
            "$TOPOLOGY" \
            $TOPOLOGY_ARGUMENTS

        if [ -n "${DRY_RUN:-}" ]; then
            printf '%s\n' "$*"
        else
            "$@"
        fi

        run=$((run + 1))
    done
done
