#!/bin/sh
#
# Submits one Slurm job per run of the UCI (tabular) classification batch.
#
#   sh scripts/submit_exaqc_uci_sho_jobs.sh <min_run> <max_run> <tag> [dataset ...]
#
# for example:
#
#   sh scripts/submit_exaqc_uci_sho_jobs.sh 1 10 sho
#   sh scripts/submit_exaqc_uci_sho_jobs.sh 1 10 sho iris
#
# Without names, every dataset (iris seeds breast_cancer wine) is submitted. Run i is
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
#   TIME_LIMIT  Slurm time limit per job (default: the job script's 5-00:00:00)
#   NTASKS      MPI processes per job (default: the job script's 144)
#   ARCHIVE_DIR
#               read by the job script, which sbatch reaches through its
#               default environment export -- so set them on this command line
#   DRY_RUN     when set, print the sbatch commands instead of submitting

set -eu

#: Where the job script writes Slurm's stdout/stderr, one pair per job.
LOG_DIR=${LOG_DIR:-/home/dk7405/logs/exaqc}

#: The datasets the job script accepts. Kept space-delimited for an
#: exact-token match below.
NAMES_ALLOWED="iris seeds breast_cancer wine"

usage() {
    cat >&2 <<'USAGE'
usage: sh scripts/submit_exaqc_uci_sho_jobs.sh <min_run> <max_run> <tag> [dataset ...]

  min_run  first run index to submit (inclusive); also its --seed
  max_run  last run index to submit (inclusive); one job per index and dataset
  tag      keyword distinguishing this run type from others (letters, digits,
           - and _). It names both the job and its archive directory, so two
           run types cannot resume one another's archives
  dataset  datasets to submit (default: all of iris seeds breast_cancer wine)

examples:
  sh scripts/submit_exaqc_uci_sho_jobs.sh 1 10 sho
  sh scripts/submit_exaqc_uci_sho_jobs.sh 1 10 sho iris
  TIME_LIMIT=1-00:00:00 sh scripts/submit_exaqc_uci_sho_jobs.sh 1 10 sho
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

[ $# -ge 3 ] || usage

MIN_RUN=$1
MAX_RUN=$2
TAG=$3
shift 3

is_whole_number "$MIN_RUN" && [ "$MIN_RUN" -gt 0 ] ||
    die "min_run must be a positive integer, but found: $MIN_RUN"

is_whole_number "$MAX_RUN" && [ "$MAX_RUN" -gt 0 ] ||
    die "max_run must be a positive integer, but found: $MAX_RUN"

[ "$MAX_RUN" -ge "$MIN_RUN" ] ||
    die "max_run ($MAX_RUN) must be >= min_run ($MIN_RUN)"

is_tag "$TAG" ||
    die "tag must be non-empty and contain only letters, digits, '-' and '_', but found: $TAG"

if [ $# -eq 0 ]; then
    NAMES=$NAMES_ALLOWED
else
    for name in "$@"; do
        contains_word "$NAMES_ALLOWED" "$name" ||
            die "unknown dataset: $name (choose from: $NAMES_ALLOWED)"
    done
    NAMES="$*"
fi

if [ -n "${NTASKS:-}" ]; then
    is_whole_number "$NTASKS" && [ "$NTASKS" -gt 0 ] ||
        die "NTASKS must be a positive integer, but found: $NTASKS"
fi

SCRIPT_DIR=$(dirname "$0")
JOB_SCRIPT="$SCRIPT_DIR/exaqc_uci_sho_job.sh"

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
        # archive directory the run writes, so all four agree.
        job_name="${name}_exaqc_${TAG}_uci_${run}"

        # RESOURCE_ARGUMENTS is deliberately unquoted, so each flag is its own
        # argument.
        set -- sbatch \
            -J "$job_name" \
            -o "${LOG_DIR}/${job_name}_%j.o" \
            -e "${LOG_DIR}/${job_name}_%j.e" \
            $RESOURCE_ARGUMENTS \
            "$JOB_SCRIPT" \
            "$name" \
            "$run" \
            "$job_name"

        if [ -n "${DRY_RUN:-}" ]; then
            printf '%s\n' "$*"
        else
            "$@"
        fi

        run=$((run + 1))
    done
done
