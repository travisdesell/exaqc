#!/bin/bash -l
#
# Runs a single EXAQC reinforcement-learning search as a Slurm job.
#
#   sbatch scripts/exaqc_rl_job.sh <env> <input_qubits> <output_qubits> <run_name> \
#                                  <n_islands> <max_island_size> \
#                                  <topology> [topology arguments...]
#
# <run_name> names both the Slurm job and its archive directory, and is built by
# the submitting script so the two cannot disagree.
#
# This is normally submitted by scripts/submit_exaqc_rl_jobs.sh, which validates
# the arguments and sets the job name and log paths per run. One job is one run:
# the runs of an experiment are separate jobs rather than a loop inside one.
#
# The resource requests below are defaults for a single run; the submitting
# script overrides -J, -o and -e so each run is named and logged separately.
#
# Optional environment variables, normally set on the submitting command line
# (described where they are read below): HEALTHY_REWARD, EPISODES,
# IMPROVEMENT_CUTOFF, LOG_EVERY, NUMBER_GENOMES, EVAL_SEED, USE_ONLY and DRY_RUN.
#
#SBATCH -J exaqc_rl
#SBATCH -t 1-00:00:00
#SBATCH -A examm-rnn -p tigris
#SBATCH -o /home/dk7405/logs/exaqc/output_%j.o
#SBATCH -e /home/dk7405/logs/exaqc/error_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=144
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB

set -eu

#: Where each run's genome archive is written.
ARCHIVE_DIR=/home/dk7405/genome_archives

#: The environments whose reward carries an "alive" bonus, and so accept
#: --healthy_reward. Mirrors what
#: src.objectives.reinforcement_learning_objective.supported_env_knobs() reads off each
#: Gymnasium environment, and is checked against it by
#: tests/test_exaqc_rl_job_script.py so the two cannot drift apart. Note that
#: HalfCheetah is MuJoCo locomotion but cannot terminate, so it has no healthy
#: bonus at all. Kept space-delimited for an exact-token match below.
HEALTHY_REWARD_ENVIRONMENTS="hopper walker2d ant humanoid"

#: Per-step bonus for staying upright, for the environments that have one.
#: Unset by default, so --healthy_reward is not passed at all and each
#: environment's own Gymnasium value applies -- which keeps a run comparable
#: with the published baselines. Set HEALTHY_REWARD to override it:
#:
#:     HEALTHY_REWARD=0.2 sbatch scripts/exaqc_rl_job.sh ...
#:
#: Worth overriding because the Gymnasium default of 1.0 makes standing still
#: for a full --max_steps 1000 episode worth ~1000, which can swamp the
#: forward-progress reward and leave the search selecting policies that balance
#: rather than walk.
HEALTHY_REWARD=${HEALTHY_REWARD:-}

#: Training episodes per genome. 0 evaluates each genome's inherited weights
#: without training it.
EPISODES=${EPISODES:-200}

#: Episodes without an improved evaluation before a genome's training stops
#: early; 0 disables early stopping.
IMPROVEMENT_CUTOFF=${IMPROVEMENT_CUTOFF:-60}

#: Evaluate (and log) every this many training episodes; 1 evaluates after every
#: episode. The cutoff is counted in episodes, so it allows
#: IMPROVEMENT_CUTOFF / LOG_EVERY evaluations without improvement.
LOG_EVERY=${LOG_EVERY:-5}

#: Genomes the run evaluates.
NUMBER_GENOMES=${NUMBER_GENOMES:-10000}

#: Base seed for every genome's evaluation episodes and their action sampling.
#: Unset by default, so each genome draws its own; setting it scores every genome
#: on the same episodes with the same sampling noise. Training seeds stay random
#: per genome either way (--training_seed is never passed).
EVAL_SEED=${EVAL_SEED:-}

#: The gates the search may use, as space-separated gate method names passed to
#: --use_only. Set USE_ONLY to search a different gate set, quoting it so the
#: shell keeps the names together as one value:
#:
#:     USE_ONLY="rx ry rz cx" sbatch scripts/exaqc_rl_job.sh ...
USE_ONLY=${USE_ONLY:-rx ry rz p u crx cry crz cp rxx ryy rzz cx cy cz}

if [ $# -lt 7 ]; then
    echo "usage: $0 <env> <input_qubits> <output_qubits> <run_name> <n_islands> <max_island_size> <topology> [topology arguments...]" >&2
    exit 2
fi

ENVIRONMENT=$1
INPUT_QUBITS=$2
OUTPUT_QUBITS=$3
RUN_NAME=$4
N_ISLANDS=$5
MAX_ISLAND_SIZE=$6
shift 6
# whatever is left is the topology and its own arguments

# The archive directory *is* the job name, so a job in the queue and the archive
# it writes carry the same label. The submitting script builds that name from
# everything that has to separate one experiment from another -- the run type,
# the topology and its arguments, the environment, the island count and the run
# index -- because --restart defaults to auto, and two experiments sharing a
# directory would silently resume one another.
OUT_DIR="${ARCHIVE_DIR}/${RUN_NAME}"

# --healthy_reward is passed only when HEALTHY_REWARD asks for it, so by
# default the environment's own value stands.
#
# Asking for it on an environment that has no alive bonus is an error rather
# than something to quietly drop: the run would otherwise be tagged as one
# reward setting while having been trained under another. This mirrors
# src.objectives.reinforcement_learning_objective.environment_knob_kwargs(), which raises
# instead of ignoring -- but catching it here costs an error message rather
# than a scheduled job that dies on startup.
HEALTHY_REWARD_ARGUMENTS=()
if [ -n "$HEALTHY_REWARD" ]; then
    case "$HEALTHY_REWARD" in
        '' | *[!0-9.-]* | *.*.* | *-*-*)
            echo "error: HEALTHY_REWARD must be a number, but found: $HEALTHY_REWARD" >&2
            exit 2
            ;;
    esac

    case " ${HEALTHY_REWARD_ENVIRONMENTS} " in
        *" ${ENVIRONMENT} "*)
            HEALTHY_REWARD_ARGUMENTS=(--healthy_reward "$HEALTHY_REWARD")
            ;;
        *)
            echo "error: HEALTHY_REWARD=$HEALTHY_REWARD was given, but the $ENVIRONMENT environment has no healthy (alive) bonus; it applies to: $HEALTHY_REWARD_ENVIRONMENTS" >&2
            exit 2
            ;;
    esac
fi

# The submitting script validates these too; they are checked again here so a
# job submitted directly with sbatch fails with a message rather than an
# argparse error from every rank.
case "$EPISODES" in
    '' | *[!0-9]*)
        echo "error: EPISODES must be a non-negative integer, but found: $EPISODES" >&2
        exit 2
        ;;
esac

case "$IMPROVEMENT_CUTOFF" in
    '' | *[!0-9]*)
        echo "error: IMPROVEMENT_CUTOFF must be a non-negative integer, but found: $IMPROVEMENT_CUTOFF" >&2
        exit 2
        ;;
esac

case "$LOG_EVERY" in
    '' | *[!0-9]* | 0)
        echo "error: LOG_EVERY must be a positive integer, but found: $LOG_EVERY" >&2
        exit 2
        ;;
esac

case "$NUMBER_GENOMES" in
    '' | *[!0-9]* | 0)
        echo "error: NUMBER_GENOMES must be a positive integer, but found: $NUMBER_GENOMES" >&2
        exit 2
        ;;
esac

# --eval_seed is passed only when EVAL_SEED asks for it, so by default every
# genome draws its own evaluation seed.
EVAL_SEED_ARGUMENTS=()
if [ -n "$EVAL_SEED" ]; then
    case "$EVAL_SEED" in
        *[!0-9]*)
            echo "error: EVAL_SEED must be a non-negative integer, but found: $EVAL_SEED" >&2
            exit 2
            ;;
    esac
    EVAL_SEED_ARGUMENTS=(--eval_seed "$EVAL_SEED")
fi

#: Whether to switch on the PPO training stabilizers: running observation
#: normalization, reward scaling, linear learning-rate decay, KL early stopping
#: and a tighter log-std clamp. On by default; STABILIZE=0 leaves the code's own
#: defaults.
STABILIZE=${STABILIZE:-1}
STABILIZER_ARGUMENTS=()
if [ "$STABILIZE" = "1" ]; then
    STABILIZER_ARGUMENTS=(--normalize_observations --ppo_reward_scaling --lr_anneal linear --ppo_target_kl 0.015 --log_std_max 0.5 --ppo_passes 10)
fi
# Gate names are lowercase letters and digits, so anything else (a comma, a
# shell character) is a malformed list rather than a gate. Checking the
# characters first also means the split below cannot glob.
case "$USE_ONLY" in
    *[!a-z0-9\ ]*)
        echo "error: USE_ONLY must be space-separated gate names (lowercase letters and digits), but found: $USE_ONLY" >&2
        exit 2
        ;;
    *[a-z0-9]*) ;;
    *)
        echo "error: USE_ONLY must name at least one gate, but found: '$USE_ONLY'" >&2
        exit 2
        ;;
esac
read -r -a USE_ONLY_GATES <<<"$USE_ONLY"

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
    --learning_rate 0.003
    --rollout_steps 2048
    --max_steps 1000
    --input_qubits "$INPUT_QUBITS"
    --output_qubits "$OUTPUT_QUBITS"
    --number_genomes "$NUMBER_GENOMES"
    --entropy_coef 0.0
    --episodes "$EPISODES"
    # expands to nothing unless HEALTHY_REWARD asked for it; written this way
    # because `set -u` rejects a bare empty-array expansion on bash older than
    # 4.4, which the cluster may still be running
    ${HEALTHY_REWARD_ARGUMENTS[@]+"${HEALTHY_REWARD_ARGUMENTS[@]}"}
    --eval_episodes 20
    # expands to nothing unless EVAL_SEED asked for it (written as above)
    ${EVAL_SEED_ARGUMENTS[@]+"${EVAL_SEED_ARGUMENTS[@]}"}
    --ema_alpha 0.1
    --log_every "$LOG_EVERY"
    --mutation_strategy uniform 1 7
    --parent_strategy uniform 2 7
    # expands to nothing when STABILIZE=0 (written as above)
    ${STABILIZER_ARGUMENTS[@]+"${STABILIZER_ARGUMENTS[@]}"}
    -qim ry
    -qom expval
    --encoding linear
    --decoding linear
    # --use_only takes any number of values, so another flag must follow it
    --use_only "${USE_ONLY_GATES[@]}"
    --out_dir "$OUT_DIR"
    --improvement_cutoff "$IMPROVEMENT_CUTOFF"
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

source .exaqc/bin/activate

srun "${COMMAND[@]}"
