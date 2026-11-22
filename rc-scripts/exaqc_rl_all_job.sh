#!/bin/bash -l
#
# Runs a single EXAQC reinforcement-learning search as a Slurm job (RIT Tigris).
#
#   sbatch exaqc_rl_all_job.sh <environment> <population> <mode> <run>
#
#   <environment>  cartpole | acrobot | mountaincar | frozenlake |
#                  mountaincar_continuous | pendulum | hopper | walker2d |
#                  halfcheetah | ant | humanoid
#   <population>   fully_connected | ring | star | 2d_mesh | tree | random | steady_state
#   <mode>         fixed          use the configured hyperparameters
#                  sho            SHO on the training settings, plus the operator
#                                 settings on islands (see below)
#                  sho_training   SHO on the training settings only
#                  sho_operators  SHO on the operator settings only (islands only):
#                                 crossover rates, mutation count, mutation weights
#                  sho_mutations  SHO on the mutation settings only (islands only):
#                                 mutation count and mutation weights
#   <run>          run number; also the --training_seed
#
# Normally submitted by submit_rl_all.sh, which loops over runs, environments
# and populations and sets the job name and log files. One job is one run.
# Search and cluster settings come from environment variables (sbatch passes
# the submitting shell's environment through), with the defaults below.
#
# The defaults are the settings of the sho02/sho03 SHO islands runs (commit
# 98bccff): PPO, 200 training episodes per genome, greedy evaluation, 10,000
# genomes, 10 islands of 10, expval readout.
#
# With sho, every run evolves the training settings (learning rate, Adam, and
# PPO's gamma, entropy coefficient, GAE lambda and clip), and islands runs also
# evolve the operator settings (crossover rates, mutation count, mutation
# weights). EXAQC only allows operator settings to be tuned on islands, so
# steady-state runs evolve the training settings only. SHO_OPERATORS=0 keeps
# islands to the training settings too.
#
# Search settings:
#   N_ISLANDS         islands per run              [10]
#   MAX_ISLAND_SIZE   genomes per island           [10]
#   MESH              2d_mesh dimensions, must multiply to N_ISLANDS [2 5]
#   TREE_CHILDREN     children per node for tree   [2]
#   RANDOM_EDGES      min and max edges for random [1 3]
#   POP_SIZE          steady-state population size [N_ISLANDS * MAX_ISLAND_SIZE]
#   NUMBER_GENOMES    genomes per run              [10000]
#   EPISODES          training episodes per genome [200]
#   MUTATIONS         min and max mutations per child (uniform) [1 7]
#   PARENTS           min and max n-ary crossover parents (uniform) [2 7]
#   QOM               readout, expval or probs; sets the output qubits [expval]
#   STABILIZE         1 = PPO stabilizers on (observation normalization,
#                     reward scaling, lr annealing, KL stop, log-std clamp) [0]
#   SHO_OPERATORS     1 = also tune operator settings on islands [1]
#
# Cluster settings:
#   REPO              repository on the cluster    [/home/dk7405/Quantum/exaqc]
#   VENV              venv to activate, relative to REPO or absolute [.exaqc]
#   PYTHON            python inside the venv       [python3.12]
#   OUT_ROOT          where the archives go        [/home/dk7405/genome_archives/rl_runs]
#   OPENMPI           spack spec for OpenMPI       [/ttqroyz]
#   DRY_RUN           1 = print the command instead of running it
#
# The resource requests below are defaults; submit_rl_all.sh overrides them
# on the sbatch command line.
#
#SBATCH -J exaqc_rl_all
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

if [ $# -ne 4 ]; then
    echo "usage: $0 <environment> <population> <fixed|sho|sho_training|sho_operators|sho_mutations> <run>" >&2
    exit 2
fi
ENVIRONMENT=$1
POPULATION=$2
MODE=$3
RUN=$4
case "$MODE" in
    fixed|sho|sho_training|sho_operators|sho_mutations) ;;
    *) echo "error: mode must be fixed, sho, sho_training, sho_operators or sho_mutations, not '$MODE'" >&2; exit 2 ;;
esac

N_ISLANDS=${N_ISLANDS:-10}
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-10}
read -r -a MESH <<< "${MESH:-2 5}"
TREE_CHILDREN=${TREE_CHILDREN:-2}
read -r -a RANDOM_EDGES <<< "${RANDOM_EDGES:-1 3}"
# equal total capacity by default, so steady-state is not just a smaller population
POP_SIZE=${POP_SIZE:-$((N_ISLANDS * MAX_ISLAND_SIZE))}
NUMBER_GENOMES=${NUMBER_GENOMES:-10000}
EPISODES=${EPISODES:-200}
read -r -a MUTATIONS <<< "${MUTATIONS:-1 7}"
read -r -a PARENTS <<< "${PARENTS:-2 7}"
QOM=${QOM:-expval}
STABILIZE=${STABILIZE:-0}
SHO_OPERATORS=${SHO_OPERATORS:-1}

REPO=${REPO:-/home/dk7405/Quantum/exaqc}
VENV=${VENV:-.exaqc}
PYTHON=${PYTHON:-python3.12}
OUT_ROOT=${OUT_ROOT:-/home/dk7405/genome_archives/rl_runs}
OPENMPI=${OPENMPI:-/ttqroyz}

case "$QOM" in
    probs|expval) ;;
    *) echo "error: QOM must be probs or expval, not '$QOM'" >&2; exit 2 ;;
esac

# The output qubits must carry the policy's outputs: one per discrete action,
# or a mean and log-std per continuous action dimension, plus one state value
# for PPO. probs gives 2^output_qubits values, so it needs ceil(log2(outputs))
# qubits; expval gives one value per qubit, capped at 8 (the linear decoder
# maps the 8 values onto the larger MuJoCo outputs). The learning rate's SHO
# burn-in sits around each environment's configured rate (the default 0.01,
# or 3e-4 for the MuJoCo and continuous-control ones).
MUJOCO_ARGS=(--learning_rate 3e-4 --rollout_steps 2048 --max_steps 1000
             --use_only rx ry rz p u crx cry crz cp rxx ryy rzz cx cy cz)
DISCRETE_LR=learning_rate=log:1e-3:5e-2:1e-5:0.3
CONTINUOUS_LR=learning_rate=log:1e-4:1e-3:1e-5:0.1
case "$ENVIRONMENT" in
    cartpole)  # 2 actions + value
        ENV_ARGS=(--input_qubits 4); QUBITS_PROBS=2; QUBITS_EXPVAL=3; LR_TUNE=$DISCRETE_LR ;;
    acrobot)  # 3 actions + value
        ENV_ARGS=(--input_qubits 4); QUBITS_PROBS=2; QUBITS_EXPVAL=4; LR_TUNE=$DISCRETE_LR ;;
    mountaincar)  # 3 actions + value
        ENV_ARGS=(--input_qubits 2); QUBITS_PROBS=2; QUBITS_EXPVAL=4; LR_TUNE=$DISCRETE_LR ;;
    frozenlake)  # 4 actions + value
        ENV_ARGS=(--map_name 4x4 --is_slippery --input_qubits 4 --eval_episodes 10)
        QUBITS_PROBS=3; QUBITS_EXPVAL=5; LR_TUNE=$DISCRETE_LR ;;
    mountaincar_continuous)  # 1 action dimension x (mean, log-std) + value
        ENV_ARGS=(--learning_rate 3e-4 --rollout_steps 1024 --max_steps 1000 --input_qubits 2)
        QUBITS_PROBS=2; QUBITS_EXPVAL=3; LR_TUNE=$CONTINUOUS_LR ;;
    pendulum)  # 1 action dimension x (mean, log-std) + value; Gymnasium ends episodes at 200 steps
        ENV_ARGS=(--learning_rate 3e-4 --rollout_steps 1024 --max_steps 200 --input_qubits 2)
        QUBITS_PROBS=2; QUBITS_EXPVAL=3; LR_TUNE=$CONTINUOUS_LR ;;
    hopper)  # 3 action dimensions x (mean, log-std) + value
        ENV_ARGS=("${MUJOCO_ARGS[@]}" --healthy_reward 0.1 --input_qubits 4)
        QUBITS_PROBS=3; QUBITS_EXPVAL=7; LR_TUNE=$CONTINUOUS_LR ;;
    walker2d)  # 6 action dimensions x (mean, log-std) + value
        ENV_ARGS=("${MUJOCO_ARGS[@]}" --healthy_reward 0.1 --input_qubits 6)
        QUBITS_PROBS=4; QUBITS_EXPVAL=8; LR_TUNE=$CONTINUOUS_LR ;;
    halfcheetah)  # 6 action dimensions x (mean, log-std) + value; no alive bonus
        ENV_ARGS=("${MUJOCO_ARGS[@]}" --input_qubits 6)
        QUBITS_PROBS=4; QUBITS_EXPVAL=8; LR_TUNE=$CONTINUOUS_LR ;;
    ant)  # 8 action dimensions x (mean, log-std) + value
        ENV_ARGS=("${MUJOCO_ARGS[@]}" --healthy_reward 0.1 --input_qubits 8)
        QUBITS_PROBS=5; QUBITS_EXPVAL=8; LR_TUNE=$CONTINUOUS_LR ;;
    humanoid)  # 17 action dimensions x (mean, log-std) + value
        ENV_ARGS=("${MUJOCO_ARGS[@]}" --healthy_reward 0.1 --input_qubits 8)
        QUBITS_PROBS=6; QUBITS_EXPVAL=8; LR_TUNE=$CONTINUOUS_LR ;;
    *) echo "error: unknown RL environment: $ENVIRONMENT" >&2; exit 2 ;;
esac
if [ "$QOM" = probs ]; then
    ENV_ARGS+=(--output_qubits "$QUBITS_PROBS")
else
    ENV_ARGS+=(--output_qubits "$QUBITS_EXPVAL")
fi

ISLANDS=(islands --n_islands "$N_ISLANDS" --max_island_size "$MAX_ISLAND_SIZE" --islands_to_extinct 1)
TAG="i${N_ISLANDS}_p${MAX_ISLAND_SIZE}"
case "$POPULATION" in
    fully_connected|ring|star)
        TAG="${TAG}_$POPULATION"; POPULATION_ARGS=("${ISLANDS[@]}" --topology "$POPULATION") ;;
    2d_mesh)
        if [ $((MESH[0] * MESH[1])) -ne "$N_ISLANDS" ]; then
            echo "error: MESH=${MESH[*]} does not multiply to N_ISLANDS=$N_ISLANDS" >&2
            exit 2
        fi
        TAG="${TAG}_2d_mesh_${MESH[0]}x${MESH[1]}"; POPULATION_ARGS=("${ISLANDS[@]}" --topology 2d_mesh "${MESH[@]}") ;;
    tree)
        TAG="${TAG}_tree_${TREE_CHILDREN}"; POPULATION_ARGS=("${ISLANDS[@]}" --topology tree "$TREE_CHILDREN") ;;
    random)
        TAG="${TAG}_random_${RANDOM_EDGES[0]}_${RANDOM_EDGES[1]}"; POPULATION_ARGS=("${ISLANDS[@]}" --topology random "${RANDOM_EDGES[@]}") ;;
    steady_state)
        TAG="pop${POP_SIZE}_steady_state"; POPULATION_ARGS=(steady_state --max_population_size "$POP_SIZE") ;;
    *) echo "error: unknown population: $POPULATION" >&2; exit 2 ;;
esac

# The ranges of the sho02/sho03 runs: burn-in range, then the full allowed range.
TRAINING_TUNE=(
    "$LR_TUNE"
    adam_beta1=linear:0.9:0.99
    adam_beta2=linear:0.9:0.999
    adam_epsilon=log:1e-9:1e-8
    gamma=linear:0.97:0.999:0.9:0.9999
    entropy_coef=linear:0.005:0.03:0:0.1
    gae_lambda=linear:0.9:0.98:0.8:0.999
    ppo_clip=linear:0.1:0.3:0.05:0.4
)
CROSSOVER_TUNE=(
    binary_crossover_rate=linear:0.05:0.15:0:1
    n_ary_crossover_rate=linear:0.05:0.15:0:1
    exponential_crossover_rate=linear:0.05:0.15:0:1
)
MUTATION_TUNE=(
    mutation_count=int:1:3:1:10
    add_gate_weight=linear:8:14:0:30
    reorder_gate_weight=linear:1:3:0:10
    qubit_swap_weight=linear:1:3:0:10
    enable_gate_weight=linear:0.5:1.5:0:10
    disable_gate_weight=linear:1:3:0:10
    clone_weight=linear:1:3:0:10
    mutate_some_weights_weight=linear:1:3:0:10
    mutate_all_weights_weight=linear:1:3:0:10
)
case "$MODE" in
    sho_operators|sho_mutations)
        if [ "$POPULATION" = steady_state ]; then
            echo "error: $MODE tunes operator settings, which EXAQC only allows on islands" >&2
            exit 2
        fi ;;
esac
SHO_ARGS=()
TUNE=()
case "$MODE" in
    sho)
        TUNE=("${TRAINING_TUNE[@]}")
        if [ "$POPULATION" != steady_state ] && [ "$SHO_OPERATORS" = 1 ]; then
            TUNE+=("${CROSSOVER_TUNE[@]}" "${MUTATION_TUNE[@]}")
        fi ;;
    sho_training) TUNE=("${TRAINING_TUNE[@]}") ;;
    sho_operators) TUNE=("${CROSSOVER_TUNE[@]}" "${MUTATION_TUNE[@]}") ;;
    sho_mutations) TUNE=("${MUTATION_TUNE[@]}") ;;
esac
if [ ${#TUNE[@]} -gt 0 ]; then
    SHO_ARGS=(--hyperparameter_strategy simplex --sho_tune "${TUNE[@]}"
              --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5)
fi

STABILIZER_ARGS=()
if [ "$STABILIZE" = 1 ]; then
    STABILIZER_ARGS=(--normalize_observations --ppo_reward_scaling --lr_anneal linear
                     --ppo_target_kl 0.015 --log_std_max 0.5 --ppo_passes 10)
fi

# The archive directory carries the same label as the job name that
# submit_rl_all.sh gives this job.
OUT_DIR="$OUT_ROOT/${ENVIRONMENT}_${MODE}_${TAG}_${QOM}_${RUN}"

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job instead of leaving workers waiting until the time limit. The
# population subcommand comes last: --topology takes any number of values.
COMMAND=(
    "$PYTHON" -m mpi4py -m src.examples.reinforcement_learning
    --algo ppo
    --logging_level INFO
    --env "$ENVIRONMENT"
    "${ENV_ARGS[@]}"
    --episodes "$EPISODES"
    --eval_policy greedy
    --number_genomes "$NUMBER_GENOMES"
    --mutation_strategy uniform "${MUTATIONS[@]}"
    --parent_strategy uniform "${PARENTS[@]}"
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1
    ${STABILIZER_ARGS[@]+"${STABILIZER_ARGS[@]}"}
    -qim ry -qom "$QOM" --encoding linear --decoding linear
    ${SHO_ARGS[@]+"${SHO_ARGS[@]}"}
    --training_seed "$RUN"
    --out_dir "$OUT_DIR"
    --shared_file_system
    "${POPULATION_ARGS[@]}"
)

if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

# spack and venv activation scripts read unset variables
set +u
export PMIX_MCA_psec="^munge"
spack load openmpi "$OPENMPI"
cd "$REPO"
source "$VENV/bin/activate"
set -u

srun "${COMMAND[@]}"