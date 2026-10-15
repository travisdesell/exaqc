#!/bin/bash -l
#
# Runs a single EXAQC reinforcement-learning search as a Slurm job.
#
#   sbatch scripts/exaqc_rl_sho_job.sh <environment> <seed> <run_name>
#
# <environment> is one of: cartpole frozenlake walker2d mountaincar_continuous
#
# Each runs with the settings of its local batch script (run_rl_10_runs.sh),
# training hyperparameters co-evolved by SHO, and --seed <seed>. <run_name> names
# both the Slurm job and its archive directory, and is built by the submitting
# script so the two cannot disagree.
#
# This is normally submitted by scripts/submit_exaqc_rl_sho_jobs.sh, which validates
# the arguments and sets the job name and log paths per run. One job is one run:
# the runs of an experiment are separate jobs rather than a loop inside one.
#
# The resource requests below are defaults for a single run; the submitting
# script overrides -J, -o and -e so each run is named and logged separately.
#
#SBATCH -J exaqc_rl
#SBATCH -t 2-00:00:00
#SBATCH -A cps -p tier3
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
ARCHIVE_DIR=${ARCHIVE_DIR:-/home/dk7405/genome_archives}

if [ $# -ne 3 ]; then
    echo "usage: $0 <environment> <seed> <run_name>" >&2
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

# the learning rate's SHO burn-in sits around each env's configured rate
# (the default 0.01, or 3e-4 for the MuJoCo and continuous-control ones)
case "$NAME" in
    cartpole)
        ENV_ARGUMENTS=(--number_genomes 1000 --input_qubits 4 --output_qubits 2 --episodes 100)
        LR_TUNE=learning_rate=log:1e-3:5e-2:1e-5:0.3
        ;;
    frozenlake)
        ENV_ARGUMENTS=(--number_genomes 1000 --map_name 4x4 --is_slippery --input_qubits 4 --output_qubits 4 --episodes 1000 --eval_episodes 100)
        LR_TUNE=learning_rate=log:1e-3:5e-2:1e-5:0.3
        ;;
    walker2d)
        ENV_ARGUMENTS=(--learning_rate 3e-4 --rollout_steps 2048 --max_steps 1000 --healthy_reward 0.1 --input_qubits 6 --output_qubits 6 --episodes 100)
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    mountaincar_continuous)
        ENV_ARGUMENTS=(--number_genomes 1000 --episodes 100 --learning_rate 3e-4 --rollout_steps 1024 --max_steps 1000 --input_qubits 2 --output_qubits 2)
        LR_TUNE=learning_rate=log:1e-4:1e-3:1e-5:0.1
        ;;
    *) echo "error: unknown RL environment: $NAME" >&2; exit 2 ;;
esac

# Running under `-m mpi4py` makes an uncaught exception on any rank abort the
# whole job. Otherwise a crashed master leaves every worker waiting on it and
# the job sits idle until its time limit.
COMMAND=(
    python -m mpi4py -m src.examples.reinforcement_learning
    --algo reinforce
    --logging_level INFO
    --env "$NAME"
    "${ENV_ARGUMENTS[@]}"
    --mutation_strategy uniform 1 3
    --parent_strategy uniform 5 5
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1
    -qim amplitude -qom probs --encoding identity --decoding clipped
    --hyperparameter_strategy simplex
    --sho_tune "$LR_TUNE"
        adam_beta1=linear:0.9:0.99 adam_beta2=linear:0.9:0.999 adam_epsilon=log:1e-9:1e-8
        "${OPERATOR_TUNE[@]}"
    --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5
    --seed "$SEED"
    --out_dir "$OUT_DIR"
    --shared_file_system
    steady_state --max_population_size 50
)

# DRY_RUN prints the search that would be run, which is how the command line is
# checked without a cluster. It comes before the environment is set up, so the
# check works anywhere rather than only where spack and the venv exist.
if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

export PMIX_MCA_psec="^munge"

# spack load openmpi /ttqroyz
spack env activate default-ml-12-3-aarch64-24050801

source "$VENV/bin/activate"

# Fail here, with the reason, rather than on every rank under srun.
python -c "import mpi4py" || {
    echo "error: the venv at $VENV cannot run on $(hostname); rebuild it on the cluster" >&2
    exit 2
}

srun "${COMMAND[@]}"
