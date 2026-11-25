#!/bin/bash -l
#
# Runs a single multi-objective (NSGA-II / NSGA-III) EXAQC classification search
# as a Slurm job, by default under Dr. Vogt's device-card noise (noise-aware
# evolution).
#
#   sbatch scripts/exaqc_moo_job.sh <dataset> <input_qubits> <output_qubits> <run_name>
#
# <run_name> names both the Slurm job and its archive directory, and is built by
# the submitting script so the two cannot disagree. This is normally submitted
# by scripts/submit_exaqc_moo_jobs.sh, which validates the settings and sets the
# job name and log paths per run. One job is one run.
#
# Optional environment variables, normally set on the submitting command line
# (described where they are read below): DEVICE_CARD, NOISE_IDLE,
# NOISE_CALIBRATION, NOISE_QUBITS, OBJECTIVES, ALGORITHM, MAX_POPULATION_SIZE,
# NUMBER_GENOMES, EPOCHS, IMPROVEMENT_CUTOFF, BATCH_SIZE, QIM, QOM, ENCODING,
# DECODING, TRAINING_SAMPLES, VALIDATION_SAMPLES, DRY_RUN, ARCHIVE_DIR and VENV.
#
#SBATCH -J exaqc_moo
#SBATCH -t 1-00:00:00
#SBATCH -A neuroevolution -p tigris
#SBATCH --nodes=1
#SBATCH --ntasks=144
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB

set -eu

#: Where each run's genome archive is written.
ARCHIVE_DIR=${ARCHIVE_DIR:-${HOME}/genome_archives}

#: The project virtual environment the job activates.
VENV=${VENV:-${HOME}/envs/exaqc}

#: The device card whose noise genomes are trained and scored under: a name
#: from device_cards/ or a card file's path. "none" runs noiselessly (a baseline
#: for the same search); the hardware objectives then cannot be used.
#: "calibration" builds the card from the IBM chip named by NOISE_CALIBRATION
#: and NOISE_QUBITS, for when no simulated card is available.
DEVICE_CARD=${DEVICE_CARD:-transmon_detuned_lindblad_markov}

#: Whether waiting wires get the card's idle channel (1) or not (0).
NOISE_IDLE=${NOISE_IDLE:-1}

#: Optional chip wiring for the card: a stored IBM calibration (path or
#: <chip>:<date>) and the physical qubits the wires sit on. Only the chip's
#: coupling map is used; unset, every pair of wires counts as wired.
NOISE_CALIBRATION=${NOISE_CALIBRATION:-}
NOISE_QUBITS=${NOISE_QUBITS:-}

#: The objectives, as space-separated NAME:SIGN (1 minimizes, -1 maximizes).
#: Defaults to the noisy loss against the card's predicted failure chance, or
#: to loss against enabled gates for a noiseless run.
if [ "$DEVICE_CARD" = "none" ]; then
    OBJECTIVES=${OBJECTIVES:-loss:1 n_enabled_gates:1}
else
    OBJECTIVES=${OBJECTIVES:-loss:1 p_fail:1}
fi

#: nsga2, or nsga3 (meant for three or more objectives).
ALGORITHM=${ALGORITHM:-nsga2}

#: Genomes the population keeps.
MAX_POPULATION_SIZE=${MAX_POPULATION_SIZE:-30}

#: Genomes the run evaluates.
NUMBER_GENOMES=${NUMBER_GENOMES:-2000}

#: Training epochs per genome, and epochs without an improved validation loss
#: before a genome's training stops early (0 disables early stopping).
EPOCHS=${EPOCHS:-30}
IMPROVEMENT_CUTOFF=${IMPROVEMENT_CUTOFF:-3}

#: Training batch size.
BATCH_SIZE=${BATCH_SIZE:-3}

#: How inputs enter (-qim) and outputs leave (-qom) the circuit, and the
#: classical encoder and decoder around it.
QIM=${QIM:-ry}
QOM=${QOM:-probs}
ENCODING=${ENCODING:-linear}
DECODING=${DECODING:-linear}

#: Caps on the training and validation sets (unset uses all of them). Noisy
#: simulation is costly, so these are the main knob for a quicker run.
TRAINING_SAMPLES=${TRAINING_SAMPLES:-}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-}

if [ $# -ne 4 ]; then
    echo "usage: $0 <dataset> <input_qubits> <output_qubits> <run_name>" >&2
    exit 2
fi

DATASET=$1
INPUT_QUBITS=$2
OUTPUT_QUBITS=$3
RUN_NAME=$4

# The archive directory *is* the job name, so a job in the queue and the archive
# it writes carry the same label. --restart defaults to auto, so resubmitting a
# run continues its archive rather than starting over.
OUT_DIR="${ARCHIVE_DIR}/${RUN_NAME}"

# Checked again here (the submitting script checks them too) so a job submitted
# directly with sbatch fails with a message rather than an argparse error from
# every rank.
for setting in MAX_POPULATION_SIZE NUMBER_GENOMES EPOCHS IMPROVEMENT_CUTOFF BATCH_SIZE; do
    value=$(eval "printf '%s' \"\$$setting\"")
    case "$value" in
        '' | *[!0-9]*)
            echo "error: $setting must be a non-negative integer, but found: $value" >&2
            exit 2
            ;;
    esac
done

case "$ALGORITHM" in
    nsga2 | nsga3) ;;
    *)
        echo "error: ALGORITHM must be nsga2 or nsga3, but found: $ALGORITHM" >&2
        exit 2
        ;;
esac

case "$NOISE_IDLE" in
    1) IDLE_ARGUMENT=--noise_idle ;;
    0) IDLE_ARGUMENT=--no-noise_idle ;;
    *)
        echo "error: NOISE_IDLE must be 1 or 0, but found: $NOISE_IDLE" >&2
        exit 2
        ;;
esac

# Objective specifications are names, digits, colons and minus signs; checking
# the characters first also means the split below cannot glob.
case "$OBJECTIVES" in
    *[!A-Za-z0-9_:\ -]*)
        echo "error: OBJECTIVES must be space-separated NAME:SIGN entries, but found: $OBJECTIVES" >&2
        exit 2
        ;;
esac
read -r -a OBJECTIVE_LIST <<<"$OBJECTIVES"

NOISE_ARGUMENTS=()
if [ "$DEVICE_CARD" != "none" ]; then
    NOISE_ARGUMENTS+=(--noise_device_card "$DEVICE_CARD" "$IDLE_ARGUMENT")
    [ -z "$NOISE_CALIBRATION" ] || NOISE_ARGUMENTS+=(--noise_calibration "$NOISE_CALIBRATION")
    [ -z "$NOISE_QUBITS" ] || NOISE_ARGUMENTS+=(--noise_qubits "$NOISE_QUBITS")
fi

SAMPLE_ARGUMENTS=()
[ -z "$TRAINING_SAMPLES" ] || SAMPLE_ARGUMENTS+=(--training_samples "$TRAINING_SAMPLES")
[ -z "$VALIDATION_SAMPLES" ] || SAMPLE_ARGUMENTS+=(--validation_samples "$VALIDATION_SAMPLES")

# --objectives, -ms and -ps take any number of values, so another flag always
# follows each of them, and the population sub-command comes last.
#
# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job. Otherwise a crashed master leaves every worker waiting on it and
# the job sits idle until its time limit.
COMMAND=(
    python3.12 -m mpi4py -m src.examples.classification_moo
    --logging_level INFO
    --dataset "$DATASET"
    --input_qubits "$INPUT_QUBITS"
    --output_qubits "$OUTPUT_QUBITS"
    --number_genomes "$NUMBER_GENOMES"
    --epochs "$EPOCHS"
    --improvement_cutoff "$IMPROVEMENT_CUTOFF"
    --batch_size "$BATCH_SIZE"
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 5 5
    -qim "$QIM"
    -qom "$QOM"
    --encoding "$ENCODING"
    --decoding "$DECODING"
    # each expands to nothing when empty; written this way because `set -u`
    # rejects a bare empty-array expansion on bash older than 4.4
    ${NOISE_ARGUMENTS[@]+"${NOISE_ARGUMENTS[@]}"}
    ${SAMPLE_ARGUMENTS[@]+"${SAMPLE_ARGUMENTS[@]}"}
    --objectives "${OBJECTIVE_LIST[@]}"
    --out_dir "$OUT_DIR"
    --shared_file_system
    "$ALGORITHM"
    --max_population_size "$MAX_POPULATION_SIZE"
)

# DRY_RUN prints the search that would be run, which is how the command line is
# checked without a cluster. It comes before the environment is set up, so the
# check works anywhere rather than only where spack and the venv exist.
if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

export PMIX_MCA_psec="^munge"

spack load openmpi /ttqroyz

source "${VENV}/bin/activate"

srun "${COMMAND[@]}"
