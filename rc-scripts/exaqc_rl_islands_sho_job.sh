#!/bin/bash -l
#
# Runs a single EXAQC reinforcement-learning search on connected islands as a Slurm job.
#
#   sbatch scripts/exaqc_rl_islands_sho_job.sh <environment> <seed> <run_name> \
#                                     <n_islands> <max_island_size> <islands_to_extinct> \
#                                     <topology> [topology arguments...]
#
# <environment> is one of: cartpole acrobot mountaincar frozenlake
# mountaincar_continuous pendulum hopper walker2d halfcheetah ant humanoid
#
# Each runs with the settings of its local batch script (run_rl_10_runs.sh), but on
# an islands population connected by <topology> instead of one steady-state one,
# with training hyperparameters co-evolved by SHO, and --seed <seed>. <run_name> names
# both the Slurm job and its archive directory, and is built by the submitting
# script so the two cannot disagree.
#
# This is normally submitted by scripts/submit_exaqc_rl_islands_sho_jobs.sh, which validates
# the arguments and sets the job name and log paths per run. One job is one run:
# the runs of an experiment are separate jobs rather than a loop inside one.
#
# The resource requests below are defaults for a single run; the submitting
# script overrides -J, -o and -e so each run is named and logged separately.
#
#SBATCH -J exaqc_rl_islands
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

if [ $# -lt 7 ]; then
    echo "usage: $0 <environment> <seed> <run_name> <n_islands> <max_island_size> <islands_to_extinct> <topology> [topology arguments...]" >&2
    exit 2
fi

NAME=$1
SEED=$2
RUN_NAME=$3
N_ISLANDS=$4
MAX_ISLAND_SIZE=$5
ISLANDS_TO_EXTINCT=$6
shift 6

#: The quantum readout: probs (2^output_qubits values) or expval (one value per
#: output qubit). It decides each environment's output qubit count below.
QOM=${QOM:-expval}
case "$QOM" in
    probs | expval) ;;
    *) echo "error: QOM must be probs or expval, but found: $QOM" >&2; exit 2 ;;
esac
# whatever is left is the topology and its own arguments

# The archive directory *is* the job name, so a job in the queue and the archive
# it writes carry the same label. --restart defaults to auto, so two experiments
# sharing a directory would silently resume one another.
OUT_DIR="${ARCHIVE_DIR}/${RUN_NAME}_${QOM}"

# The SHO settings shared with the other batches: Adam's settings, the
# crossover rates (starting around the configured 0.1 each, free anywhere in
# [0, 1]) and the number of mutations per child (starting in 1-3, up to 10).
OPERATOR_TUNE=(
    binary_crossover_rate=linear:0.05:0.15:0:1
    n_ary_crossover_rate=linear:0.05:0.15:0:1
    exponential_crossover_rate=linear:0.05:0.15:0:1
    mutation_count=int:1:3:1:10
    add_gate_weight=linear:8:14:0:30
    reorder_gate_weight=linear:1:3:0:10
    qubit_swap_weight=linear:1:3:0:10
    enable_gate_weight=linear:0.5:1.5:0:10
    disable_gate_weight=linear:1:3:0:10
    clone_weight=linear:1:3:0:10
    mutate_some_weights_weight=linear:1:3:0:10
    mutate_all_weights_weight=linear:1:3:0:10
    gamma=linear:0.97:0.999:0.9:0.9999
    entropy_coef=linear:0.005:0.03:0:0.1
    gae_lambda=linear:0.9:0.98:0.8:0.999
    ppo_clip=linear:0.1:0.3:0.05:0.4
)

# the learning rate's SHO burn-in sits around each env's configured rate
# (the default 0.01, or 3e-4 for the MuJoCo and continuous-control ones).
#
# The output qubits must carry the policy's outputs: one per discrete action, or
# a mean and log-std per continuous action dimension, plus one state value for
# PPO. probs gives 2^output_qubits values, so it needs
# ceil(log2(outputs)) qubits; expval gives one value per qubit, so it needs one
# qubit per output, capped at 8 (halfcheetah and walker2d have 13 outputs, ant
# 17 and humanoid 35, which would otherwise mean a 13- to 35-qubit circuit; the
# linear decoder maps the 8 values onto them). The MuJoCo tasks share
# walker2d's settings, with --healthy_reward only for those that have an alive
# bonus (not halfcheetah).
MUJOCO_GATES=(--use_only rx ry rz p u crx cry crz cp rxx ryy rzz cx cy cz)
MUJOCO_ARGUMENTS=(--learning_rate 3e-4 --rollout_steps 2048 --max_steps 1000 --episodes 200 "${MUJOCO_GATES[@]}")
case "$NAME" in
    cartpole)
        # 2 actions + value = 3 outputs
        ENV_ARGUMENTS=(--input_qubits 4 --episodes 200)
        OUTPUT_QUBITS_PROBS=2
        OUTPUT_QUBITS_EXPVAL=3
        LR_TUNE=learning_rate=log:1e-3:5e-2:1e-5:0.3
        ;;
    acrobot)
        # 3 actions + value = 4 outputs
        ENV_ARGUMENTS=(--input_qubits 4 --episodes 200)
        OUTPUT_QUBITS_PROBS=2
        OUTPUT_QUBITS_EXPVAL=4
        LR_TUNE=learning_rate=log:1e-3:5e-2:1e-5:0.3
        ;;
    mountaincar)
        # 3 actions + value = 4 outputs
        ENV_ARGUMENTS=(--input_qubits 2 --episodes 200)
        OUTPUT_QUBITS_PROBS=2
        OUTPUT_QUBITS_EXPVAL=4
        LR_TUNE=learning_rate=log:1e-3:5e-2:1e-5:0.3
        ;;
    frozenlake)
        # 4 actions + value = 5 outputs
        ENV_ARGUMENTS=(--map_name 4x4 --is_slippery --input_qubits 4 --episodes 200 --eval_episodes 10)
        OUTPUT_QUBITS_PROBS=3
        OUTPUT_QUBITS_EXPVAL=5
        LR_TUNE=learning_rate=log:1e-3:5e-2:1e-5:0.3
        ;;
    walker2d)
        # 6 action dimensions x (mean, log-std) + value = 13 outputs
        ENV_ARGUMENTS=("${MUJOCO_ARGUMENTS[@]}" --healthy_reward 0.1 --input_qubits 6)
        OUTPUT_QUBITS_PROBS=4
        OUTPUT_QUBITS_EXPVAL=8
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    hopper)
        # 3 action dimensions x (mean, log-std) + value = 7 outputs
        ENV_ARGUMENTS=("${MUJOCO_ARGUMENTS[@]}" --healthy_reward 0.1 --input_qubits 4)
        OUTPUT_QUBITS_PROBS=3
        OUTPUT_QUBITS_EXPVAL=7
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    halfcheetah)
        # 6 action dimensions x (mean, log-std) + value = 13 outputs; no alive bonus
        ENV_ARGUMENTS=("${MUJOCO_ARGUMENTS[@]}" --input_qubits 6)
        OUTPUT_QUBITS_PROBS=4
        OUTPUT_QUBITS_EXPVAL=8
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    ant)
        # 8 action dimensions x (mean, log-std) + value = 17 outputs
        ENV_ARGUMENTS=("${MUJOCO_ARGUMENTS[@]}" --healthy_reward 0.1 --input_qubits 8)
        OUTPUT_QUBITS_PROBS=5
        OUTPUT_QUBITS_EXPVAL=8
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    humanoid)
        # 17 action dimensions x (mean, log-std) + value = 35 outputs
        ENV_ARGUMENTS=("${MUJOCO_ARGUMENTS[@]}" --healthy_reward 0.1 --input_qubits 8)
        OUTPUT_QUBITS_PROBS=6
        OUTPUT_QUBITS_EXPVAL=8
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    mountaincar_continuous)
        # 1 action dimension x (mean, log-std) + value = 3 outputs
        ENV_ARGUMENTS=(--episodes 200 --learning_rate 3e-4 --rollout_steps 1024 --max_steps 1000 --input_qubits 2)
        OUTPUT_QUBITS_PROBS=2
        OUTPUT_QUBITS_EXPVAL=3
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    pendulum)
        # 1 action dimension x (mean, log-std) + value = 3 outputs; episodes
        # end at Gymnasium's 200-step limit
        ENV_ARGUMENTS=(--episodes 200 --learning_rate 3e-4 --rollout_steps 1024 --max_steps 1000 --input_qubits 2)
        OUTPUT_QUBITS_PROBS=2
        OUTPUT_QUBITS_EXPVAL=3
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    *) echo "error: unknown RL environment: $NAME" >&2; exit 2 ;;
esac

if [ "$QOM" = "probs" ]; then
    ENV_ARGUMENTS+=(--output_qubits "$OUTPUT_QUBITS_PROBS")
else
    ENV_ARGUMENTS+=(--output_qubits "$OUTPUT_QUBITS_EXPVAL")
fi

#: Whether to switch on the PPO training stabilizers: running observation
#: normalization, reward scaling, linear learning-rate decay, KL early stopping
#: (which is what makes the 10 passes per rollout safe) and a tighter log-std
#: clamp. Off by default; STABILIZE=1 switches them on.
STABILIZE=${STABILIZE:-0}
STABILIZER_ARGUMENTS=()
if [ "$STABILIZE" = "1" ]; then
    STABILIZER_ARGUMENTS=(--normalize_observations --ppo_reward_scaling --lr_anneal linear --ppo_target_kl 0.015 --log_std_max 0.5 --ppo_passes 10)
fi

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job. Otherwise a crashed master leaves every worker waiting on it and
# the job sits idle until its time limit.
COMMAND=(
    python3.12 -m mpi4py -m src.examples.reinforcement_learning
    --algo ppo
    --logging_level INFO
    --env "$NAME"
    "${ENV_ARGUMENTS[@]}"
    --number_genomes 10000
    --mutation_strategy uniform 1 7
    --parent_strategy uniform 2 7
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1
    ${STABILIZER_ARGUMENTS[@]+"${STABILIZER_ARGUMENTS[@]}"}
    -qim ry -qom $QOM --encoding linear --decoding linear
    --hyperparameter_strategy simplex
    --sho_tune "$LR_TUNE"
        adam_beta1=linear:0.9:0.99 adam_beta2=linear:0.9:0.999 adam_epsilon=log:1e-9:1e-8
        "${OPERATOR_TUNE[@]}"
    --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5
    --training_seed "$SEED"
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
# spack env activate default-ml-12-3-aarch64-24050801

source ".exaqc/bin/activate"

# Fail here, with the reason, rather than on every rank under srun.
# python -c "import mpi4py" || {
#     echo "error: the venv at $VENV cannot run on $(hostname); rebuild it on the cluster" >&2
#     exit 2
# }

srun "${COMMAND[@]}"
