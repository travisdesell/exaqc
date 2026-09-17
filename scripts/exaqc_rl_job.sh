#!/bin/bash -l
#
# Runs a single EXAQC reinforcement-learning search as a Slurm job.
#
#   sbatch scripts/exaqc_rl_job.sh <env> <input_qubits> <output_qubits> <run> \
#                                  <n_islands> <max_island_size> <topology_tag> \
#                                  <topology> [topology arguments...]
#
# This is normally submitted by scripts/submit_exaqc_rl_jobs.sh, which validates
# the arguments and sets the job name and log paths per run. One job is one run:
# the runs of an experiment are separate jobs rather than a loop inside one.
#
# The resource requests below are defaults for a single run; the submitting
# script overrides -J, -o and -e so each run is named and logged separately.
#
#SBATCH -J exaqc_rl
#SBATCH -t 5-00:00:00
#SBATCH -A neuroevolution -p tigris
#SBATCH -o /home/tjdvse/logs/exaqc_test/output_%j.o
#SBATCH -e /home/tjdvse/logs/exaqc_test/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=144
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB

set -eu

#: Where each run's genome archive is written.
ARCHIVE_DIR=/home/tjdvse/genome_archives

if [ $# -lt 8 ]; then
    echo "usage: $0 <env> <input_qubits> <output_qubits> <run> <n_islands> <max_island_size> <topology_tag> <topology> [topology arguments...]" >&2
    exit 2
fi

ENVIRONMENT=$1
INPUT_QUBITS=$2
OUTPUT_QUBITS=$3
RUN=$4
N_ISLANDS=$5
MAX_ISLAND_SIZE=$6
TOPOLOGY_TAG=$7
shift 7
# whatever is left is the topology and its own arguments

# The tag carries the topology's arguments (2d_mesh_4x5, tree_2, random_2_4), so
# two shapes of the same topology land in different archives rather than one
# continuing the other -- --restart defaults to auto, so a shared directory would
# silently resume the wrong experiment.
OUT_DIR="${ARCHIVE_DIR}/${ENVIRONMENT}_i${N_ISLANDS}_${TOPOLOGY_TAG}_${RUN}"

# --topology takes any number of values, so it comes last: a flag after it would
# be read as another one of its arguments.
#
# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job. Otherwise a crashed master leaves every worker waiting on it and
# the job sits idle until its time limit.
COMMAND=(
    python3.12 -m mpi4py -m src.examples.reinforcement_learning
    --algo ppo
    --logging_level INFO
    --env "$ENVIRONMENT"
    --learning_rate 0.0005
    --rollout_steps 2048
    --max_steps 1000
    --input_qubits "$INPUT_QUBITS"
    --output_qubits "$OUTPUT_QUBITS"
    --number_genomes 5000
    --entropy_coef 0.015
    --episodes 100
    --ema_alpha 0.1
    --log_every 5
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 2 5
    -qim u3
    -qom probs
    --encoding linear
    --decoding linear
    --out_dir "$OUT_DIR"
    --improvement_cutoff 10
    --shared_file_system
    islands
    --n_islands "$N_ISLANDS"
    --max_island_size "$MAX_ISLAND_SIZE"
    --islands_to_extinct 1
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

source /home/tjdvse/envs/exaqc/bin/activate

srun "${COMMAND[@]}"
