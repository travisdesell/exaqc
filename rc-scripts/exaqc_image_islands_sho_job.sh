#!/bin/bash -l
#
# Runs a single EXAQC image classification search on connected islands as a Slurm job.
#
#   sbatch scripts/exaqc_image_islands_sho_job.sh <dataset> <seed> <run_name> \
#                                     <n_islands> <max_island_size> <islands_to_extinct> \
#                                     <topology> [topology arguments...]
#
# <dataset> is one of: mnist fashion_mnist cifar10
#
# Each runs with the settings of its local batch script (run_image_10_runs.sh), but on
# an islands population connected by <topology> instead of one steady-state one,
# with training hyperparameters co-evolved by SHO, and --seed <seed>. <run_name> names
# both the Slurm job and its archive directory, and is built by the submitting
# script so the two cannot disagree.
#
# This is normally submitted by scripts/submit_exaqc_image_islands_sho_jobs.sh, which validates
# the arguments and sets the job name and log paths per run. One job is one run:
# the runs of an experiment are separate jobs rather than a loop inside one.
#
# The resource requests below are defaults for a single run; the submitting
# script overrides -J, -o and -e so each run is named and logged separately.
#
#SBATCH -J exaqc_image_islands
#SBATCH -t 5-00:00:00
#SBATCH -A examm-rnn -p tigris
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=144
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB

set -eu

#: The Python virtual environment to run in. It must be built on the cluster
#: (python3 -m venv ...; pip install -e .): a .venv copied over from a Mac holds
#: macOS binaries, which fail on the Linux nodes with "Exec format error".
VENV=${VENV:-/home/dk7405/envs/exaqc}

#: Where each run's genome archive is written.
ARCHIVE_DIR=${ARCHIVE_DIR:-/home/dk7405/genome_archives/sho02}

#: Where the image datasets are read from (and downloaded to, if missing).
#: Compute nodes may have no internet access, so download them here first.
DATA_DIR=${DATA_DIR:-/home/dk7405/datasets}

#: The CNN encoder configuration; the input channels and size are filled in per
#: dataset.
ENCODER_CONFIG=${ENCODER_CONFIG:-configs/mnist_fc.json}

if [ $# -lt 7 ]; then
    echo "usage: $0 <dataset> <seed> <run_name> <n_islands> <max_island_size> <islands_to_extinct> <topology> [topology arguments...]" >&2
    exit 2
fi

NAME=$1
SEED=$2
RUN_NAME=$3
N_ISLANDS=$4
MAX_ISLAND_SIZE=$5
ISLANDS_TO_EXTINCT=$6
shift 6
# whatever is left is the topology and its own arguments

# The archive directory *is* the job name, so a job in the queue and the archive
# it writes carry the same label. --restart defaults to auto, so two experiments
# sharing a directory would silently resume one another.
OUT_DIR="${ARCHIVE_DIR}/${RUN_NAME}"

case "$NAME" in
    mnist | fashion_mnist | cifar10) ;;
    *) echo "error: unknown image dataset: $NAME" >&2; exit 2 ;;
esac

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job. Otherwise a crashed master leaves every worker waiting on it and
# the job sits idle until its time limit.
# --training_samples 5000
# --validation_samples 1000
COMMAND=(
    python -m mpi4py -m src.examples.classification
    --logging_level INFO
    --dataset "$NAME"
    --data_dir "$DATA_DIR"
    --target pennylane
    --encoding cnn
    --encoder_config "$ENCODER_CONFIG"
    --decoding linear
    --input_qubits 4
    --output_qubits 4
    --quantum_input_mode ry
    --quantum_output_mode probs
    --batch_size 64
    --validation_batch_size 64
    --epochs 20
    --learning_rate 0.001
    --number_genomes 500
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 2 3
    --hyperparameter_strategy simplex
    --sho_tune learning_rate=log:1e-4:1e-2:1e-5:0.1
        weight_decay=linear:0:1e-4:0:1e-3
        adam_beta1=linear:0.9:0.99 adam_beta2=linear:0.9:0.999 adam_epsilon=log:1e-9:1e-8
        binary_crossover_rate=linear:0:0.1:0:1 n_ary_crossover_rate=linear:0.1:0.3:0:1
        exponential_crossover_rate=linear:0:0.2:0:1 mutation_count=int:1:3:1:10
    --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5
    --seed "$SEED"
    --out_dir "$OUT_DIR"
    --shared_file_system
    islands
    --n_islands "$N_ISLANDS"
    --max_island_size "$MAX_ISLAND_SIZE"
    --islands_to_extinct "$ISLANDS_TO_EXTINCT"
    # --topology takes any number of values, so it comes last: a flag after
    # it would be read as another one of its arguments
    --topology "$@"
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

source ".exaqc/bin/activate"

# Fail here, with the reason, rather than on every rank under srun.
python -c "import mpi4py" || {
    echo "error: the venv at $VENV cannot run on $(hostname); rebuild it on the cluster" >&2
    exit 2
}

srun "${COMMAND[@]}"
