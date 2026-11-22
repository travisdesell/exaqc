#!/bin/bash -l
#
# Runs a single EXAQC UCI (tabular) classification search as a Slurm job.
#
#   sbatch scripts/exaqc_uci_sho_job.sh <dataset> <seed> <run_name>
#
# <dataset> is one of: iris seeds breast_cancer wine
#
# Each runs with the settings of its local batch script (run_uci_10_runs.sh),
# training hyperparameters co-evolved by SHO, and --seed <seed>. <run_name> names
# both the Slurm job and its archive directory, and is built by the submitting
# script so the two cannot disagree.
#
# This is normally submitted by scripts/submit_exaqc_uci_sho_jobs.sh, which validates
# the arguments and sets the job name and log paths per run. One job is one run:
# the runs of an experiment are separate jobs rather than a loop inside one.
#
# The resource requests below are defaults for a single run; the submitting
# script overrides -J, -o and -e so each run is named and logged separately.
#
#SBATCH -J exaqc_uci
#SBATCH -t 0-06:00:00
#SBATCH -A cps -p tier3
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

set -eu

#: Where each run's genome archive is written.
ARCHIVE_DIR=${ARCHIVE_DIR:-/home/dk7405/genome_archives}

if [ $# -ne 3 ]; then
    echo "usage: $0 <dataset> <seed> <run_name>" >&2
    exit 2
fi

NAME=$1
SEED=$2
RUN_NAME=$3

# The archive directory *is* the job name, so a job in the queue and the archive
# it writes carry the same label. --restart defaults to auto, so two experiments
# sharing a directory would silently resume one another.
OUT_DIR="${ARCHIVE_DIR}/${RUN_NAME}"

# The SHO settings shared with the other batches: Adam's settings, the
# crossover rates (starting around the configured 0.1 each, free anywhere in
# [0, 1]) and the number of mutations per child (starting in 1-3, up to 10).
OPERATOR_TUNE=(
    binary_crossover_rate=linear:0.05:0.15:0:1
    n_ary_crossover_rate=linear:0.05:0.15:0:1
    exponential_crossover_rate=linear:0.05:0.15:0:1
    mutation_count=int:1:3:1:10
)

# identity encoding needs one input qubit per feature; breast_cancer is
# linearly encoded instead. The decoder is clipped when the class count
# is a power of two and linear otherwise.
case "$NAME" in
    iris) DATASET_ARGUMENTS=(--input_qubits 4 --output_qubits 2 --encoding identity --decoding linear) ;;
    seeds) DATASET_ARGUMENTS=(--input_qubits 7 --output_qubits 2 --encoding identity --decoding linear) ;;
    breast_cancer) DATASET_ARGUMENTS=(--input_qubits 8 --output_qubits 1 --encoding linear --decoding clipped) ;;
    wine) DATASET_ARGUMENTS=(--input_qubits 13 --output_qubits 2 --encoding identity --decoding linear) ;;
    *) echo "error: unknown UCI dataset: $NAME" >&2; exit 2 ;;
esac

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job. Otherwise a crashed master leaves every worker waiting on it and
# the job sits idle until its time limit.
COMMAND=(
    python3.11 -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$NAME"
    --number_genomes 1000
    "${DATASET_ARGUMENTS[@]}"
    --batch_size 3
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 5 5
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1
    -qim ry -qom probs
    --hyperparameter_strategy simplex
    --sho_tune learning_rate=log:1e-3:5e-2:1e-5:0.3
        adam_beta1=linear:0.9:0.99 adam_beta2=linear:0.9:0.99 adam_epsilon=log:1e-9:1e-8
        "${OPERATOR_TUNE[@]}"
    --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5
    --seed "$SEED"
    --out_dir "$OUT_DIR"
    --shared_file_system
    steady_state --max_population_size 30
)

# DRY_RUN prints the search that would be run, which is how the command line is
# checked without a cluster. It comes before the environment is set up, so the
# check works anywhere rather than only where spack and the venv exist.
if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

# export PMIX_MCA_psec="^munge"

# spack load openmpi /ttqroyz
spack env activate default-ml-12-3-aarch64-24050801

source .venv/bin/activate

srun "${COMMAND[@]}"
