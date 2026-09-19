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

#: The environments whose reward carries an "alive" bonus, and so accept
#: --healthy_reward. Mirrors what
#: src.examples.reinforcement_learning.supported_env_knobs() reads off each
#: Gymnasium environment, and is checked against it by
#: tests/test_exaqc_rl_job_script.py so the two cannot drift apart. Note that
#: HalfCheetah is MuJoCo locomotion but cannot terminate, so it has no healthy
#: bonus at all. Kept space-delimited for an exact-token match below.
HEALTHY_REWARD_ENVIRONMENTS="hopper walker2d ant humanoid"

#: Per-step bonus for staying upright, for the environments that have one. The
#: Gymnasium default of 1.0 makes standing still for a full --max_steps 1000
#: episode worth ~1000, which can swamp the forward-progress reward and leave
#: the search selecting policies that balance rather than walk.
HEALTHY_REWARD=0.2

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

# Only the environments with an alive bonus take --healthy_reward: the entry
# point rejects it for the others rather than quietly ignoring it, so it is
# added only where it applies and this script stays usable for every --env.
HEALTHY_REWARD_ARGUMENTS=()
case " ${HEALTHY_REWARD_ENVIRONMENTS} " in
    *" ${ENVIRONMENT} "*)
        HEALTHY_REWARD_ARGUMENTS=(--healthy_reward "$HEALTHY_REWARD")
        ;;
esac

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
    --number_genomes 10000
    --entropy_coef 0.015
    --episodes 100
    # expands to nothing for an environment without an alive bonus; written
    # this way because `set -u` rejects a bare empty-array expansion on bash
    # older than 4.4, which the cluster may still be running
    ${HEALTHY_REWARD_ARGUMENTS[@]+"${HEALTHY_REWARD_ARGUMENTS[@]}"}
    --eval_episodes 20
    --ema_alpha 0.1
    --log_every 5
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 2 5
    -qim u3
    -qom probs
    --encoding linear
    --decoding linear
    --out_dir "$OUT_DIR"
    --improvement_cutoff 30
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
