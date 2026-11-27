#!/bin/sh
#
# Submits one Slurm job per multi-objective EXAQC classification run on a UCI
# dataset, by default noise-aware: genomes are trained and scored under a
# device card's noise and NSGA-II trades the noisy loss against the card's
# predicted failure chance (p_fail).
#
#   sh scripts/submit_exaqc_moo_jobs.sh <min_run> <max_run> <tag> <dataset>
#
# for example:
#
#   sh scripts/submit_exaqc_moo_jobs.sh 1 5 card breast_cancer
#   DEVICE_CARD=transmon_detuned_mixed sh scripts/submit_exaqc_moo_jobs.sh 1 5 card wine
#   DEVICE_CARD=none sh scripts/submit_exaqc_moo_jobs.sh 1 5 baseline breast_cancer
#   ALGORITHM=nsga3 OBJECTIVES="loss:1 p_fail:1 total_time_ns:1" \
#       sh scripts/submit_exaqc_moo_jobs.sh 1 3 three_obj iris
#
# The run range is inclusive, and each run is its own job, so an experiment can
# be extended later without resubmitting finished runs. A resubmitted run
# continues its archive (--restart defaults to auto).
#
# Written for POSIX sh (invoked as `sh ...`), so no arrays and no [[ ]].
#
# Environment variables (all optional):
#   INPUT_QUBITS / OUTPUT_QUBITS
#                    the circuit's qubits; default to each dataset's 2026 PPSN
#                    settings (breast_cancer 8/1, iris 4/2, seeds 6/2, wine 4/2)
#   DEVICE_CARD      card name from device_cards/ or a card path (default
#                    transmon_detuned_lindblad_markov); "none" for a noiseless
#                    baseline; "calibration" to build the card from the IBM
#                    chip in NOISE_CALIBRATION / NOISE_QUBITS (both required).
#                    Part of the run name
#   ALGORITHM        nsga2 (default) or nsga3. Part of the run name
#   OBJECTIVES, NOISE_IDLE, NOISE_CALIBRATION, NOISE_QUBITS,
#   MAX_POPULATION_SIZE, NUMBER_GENOMES, EPOCHS, IMPROVEMENT_CUTOFF,
#   BATCH_SIZE, QIM, QOM, ENCODING, DECODING, TRAINING_SAMPLES,
#   VALIDATION_SAMPLES, TEST_FRACTION, VALIDATION_FRACTION
#                    read by the job script (see its comments for defaults) and
#                    reaching it through sbatch's default environment export,
#                    so set them on this command line. They do not change the
#                    run name, so give each combination its own tag
#   N_TASKS          MPI ranks (one master, the rest workers), passed to sbatch
#                    as --ntasks and --ntasks-per-node (default 144, one node)
#   TIME_LIMIT       Slurm wall time (D-HH:MM:SS or HH:MM:SS); unset keeps the
#                    job script's 1 day
#   ACCOUNT          Slurm account (-A); unset keeps the job script's
#   LOG_DIR          where Slurm's stdout/stderr go (default ~/logs/exaqc_moo)
#   ARCHIVE_DIR, VENV
#                    read by the job script (defaults ~/genome_archives and
#                    ~/envs/exaqc)
#   DRY_RUN          when set, print the sbatch commands instead of submitting

set -eu

LOG_DIR=${LOG_DIR:-${HOME}/logs/exaqc_moo}
N_TASKS=${N_TASKS:-144}
DEVICE_CARD=${DEVICE_CARD:-transmon_detuned_lindblad_markov}
ALGORITHM=${ALGORITHM:-nsga2}

usage() {
    cat >&2 <<'USAGE'
usage: sh scripts/submit_exaqc_moo_jobs.sh <min_run> <max_run> <tag> <dataset>

  min_run   first run index to submit (inclusive)
  max_run   last run index to submit (inclusive); one job per index
  tag       keyword for this run type (letters, digits, - and _); it names the
            job and its archive directory
  dataset   breast_cancer, iris, seeds or wine

environment variables (see the comments at the top of this script):
  INPUT_QUBITS, OUTPUT_QUBITS, DEVICE_CARD, ALGORITHM, OBJECTIVES (quoted),
  NOISE_IDLE, NOISE_CALIBRATION, NOISE_QUBITS, MAX_POPULATION_SIZE,
  NUMBER_GENOMES, EPOCHS, IMPROVEMENT_CUTOFF, BATCH_SIZE, QIM, QOM, ENCODING,
  DECODING, TRAINING_SAMPLES, VALIDATION_SAMPLES, TEST_FRACTION,
  VALIDATION_FRACTION, N_TASKS, TIME_LIMIT, ACCOUNT,
  LOG_DIR, ARCHIVE_DIR, VENV, DRY_RUN
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

is_tag() {
    case "$1" in
        '' | *[!A-Za-z0-9_-]*) return 1 ;;
    esac
    return 0
}

[ $# -eq 4 ] || usage

MIN_RUN=$1
MAX_RUN=$2
TAG=$3
DATASET=$4

is_whole_number "$MIN_RUN" && [ "$MIN_RUN" -gt 0 ] ||
    die "min_run must be a positive integer, but found: $MIN_RUN"
is_whole_number "$MAX_RUN" && [ "$MAX_RUN" -ge "$MIN_RUN" ] ||
    die "max_run must be an integer >= min_run ($MIN_RUN), but found: $MAX_RUN"
is_tag "$TAG" ||
    die "tag must contain only letters, digits, '-' and '_', but found: $TAG"

# Each dataset's qubit counts from the 2026 PPSN experiments, unless overridden.
case "$DATASET" in
    breast_cancer) DEFAULT_IN=8 DEFAULT_OUT=1 ;;
    iris) DEFAULT_IN=4 DEFAULT_OUT=2 ;;
    seeds) DEFAULT_IN=6 DEFAULT_OUT=2 ;;
    wine) DEFAULT_IN=4 DEFAULT_OUT=2 ;;
    *) die "dataset must be one of: breast_cancer iris seeds wine, but found: $DATASET" ;;
esac
INPUT_QUBITS=${INPUT_QUBITS:-$DEFAULT_IN}
OUTPUT_QUBITS=${OUTPUT_QUBITS:-$DEFAULT_OUT}

is_whole_number "$INPUT_QUBITS" && [ "$INPUT_QUBITS" -gt 0 ] ||
    die "INPUT_QUBITS must be a positive integer, but found: $INPUT_QUBITS"
is_whole_number "$OUTPUT_QUBITS" && [ "$OUTPUT_QUBITS" -gt 0 ] ||
    die "OUTPUT_QUBITS must be a positive integer, but found: $OUTPUT_QUBITS"
is_whole_number "$N_TASKS" && [ "$N_TASKS" -ge 2 ] ||
    die "N_TASKS must be at least 2 (one master plus a worker), but found: $N_TASKS"

case "$ALGORITHM" in
    nsga2 | nsga3) ;;
    *) die "ALGORITHM must be nsga2 or nsga3, but found: $ALGORITHM" ;;
esac

for setting in MAX_POPULATION_SIZE NUMBER_GENOMES EPOCHS IMPROVEMENT_CUTOFF BATCH_SIZE TRAINING_SAMPLES VALIDATION_SAMPLES; do
    value=$(eval "printf '%s' \"\${$setting:-}\"")
    if [ -n "$value" ]; then
        is_whole_number "$value" ||
            die "$setting must be a non-negative integer, but found: $value"
    fi
done

if [ -n "${TIME_LIMIT:-}" ]; then
    case "$TIME_LIMIT" in
        *[!0-9:-]* | *-*-* | -* | *- | *::* | :* | *:)
            die "TIME_LIMIT must look like D-HH:MM:SS or HH:MM:SS, but found: $TIME_LIMIT"
            ;;
        *-*:*:* | *:*:*) ;;
        *) die "TIME_LIMIT must look like D-HH:MM:SS or HH:MM:SS, but found: $TIME_LIMIT" ;;
    esac
fi

# The card names the run (a path by its file name without device_card_), so
# runs under different cards, or a noiseless baseline, never share an archive.
case "$DEVICE_CARD" in
    */*) CARD_TAG=$(basename "$DEVICE_CARD" .json | sed 's/^device_card_//') ;;
    calibration)
        # a card built from a chip is named by that chip and day
        [ -n "${NOISE_CALIBRATION:-}" ] && [ -n "${NOISE_QUBITS:-}" ] ||
            die "DEVICE_CARD=calibration needs NOISE_CALIBRATION and NOISE_QUBITS"
        case "$NOISE_CALIBRATION" in
            */*) CARD_TAG=calibration_$(basename "$NOISE_CALIBRATION" .json) ;;
            *) CARD_TAG=calibration_$(printf '%s' "$NOISE_CALIBRATION" | tr ':' '_') ;;
        esac
        ;;
    *) CARD_TAG=$DEVICE_CARD ;;
esac
is_tag "$CARD_TAG" ||
    die "DEVICE_CARD must be a card name or a path to a .json card, but found: $DEVICE_CARD"

SCRIPT_DIR=$(dirname "$0")
JOB_SCRIPT="$SCRIPT_DIR/exaqc_moo_job.sh"
[ -f "$JOB_SCRIPT" ] || die "the job script is missing: $JOB_SCRIPT"

if [ -z "${DRY_RUN:-}" ]; then
    command -v sbatch >/dev/null 2>&1 ||
        die "sbatch is not available, so these jobs cannot be submitted from here"
    mkdir -p "$LOG_DIR"
fi

run=$MIN_RUN
while [ "$run" -le "$MAX_RUN" ]; do
    # One string names the job, its logs and its archive directory.
    name="exaqc_moo_${TAG}_${DATASET}_${CARD_TAG}_${ALGORITHM}_${run}"

    set -- sbatch \
        ${TIME_LIMIT:+-t} ${TIME_LIMIT:+"$TIME_LIMIT"} \
        ${ACCOUNT:+-A} ${ACCOUNT:+"$ACCOUNT"} \
        --ntasks "$N_TASKS" --ntasks-per-node "$N_TASKS" \
        -J "$name" \
        -o "${LOG_DIR}/${name}_%j.o" \
        -e "${LOG_DIR}/${name}_%j.e" \
        "$JOB_SCRIPT" \
        "$DATASET" \
        "$INPUT_QUBITS" \
        "$OUTPUT_QUBITS" \
        "$name"

    if [ -n "${DRY_RUN:-}" ]; then
        printf '%s\n' "$*"
    else
        "$@"
    fi

    run=$((run + 1))
done
