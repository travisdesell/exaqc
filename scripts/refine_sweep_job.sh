#!/bin/bash -l
#
# Runs one refine_genome task of the fine-tuning sweep as a Slurm array element.
#
#   sbatch --array=1-<n_tasks> scripts/refine_sweep_job.sh <sweep_dir>/tasks.tsv
#
# The task list, and the command above, are written by
# scripts/refine_sweep_prepare.py. Array element N runs the task whose task_id
# is N. A task whose refined genome already exists is skipped, so resubmitting
# the same array only reruns the tasks that did not finish.
#
# Submit from the repository root, since refine_genome is run as a module.
#
#SBATCH -J exaqc_refine_sweep
#SBATCH -t 0-08:00:00
#SBATCH -A neuroevolution -p tigris
#SBATCH -o /home/tjdvse/logs/exaqc_refine_sweep/output_%A_%a.o
#SBATCH -e /home/tjdvse/logs/exaqc_refine_sweep/error_%A_%a.e
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB

set -eu

if [ $# -ne 1 ]; then
    echo "usage: sbatch --array=1-<n_tasks> $0 <tasks.tsv>" >&2
    exit 2
fi

TASKS=$1
TASK_ID=${SLURM_ARRAY_TASK_ID:?"run this as a Slurm array job (sbatch --array=...)"}

# the row whose first column is this task's id; the header never matches a number
ROW=$(awk -F '\t' -v id="$TASK_ID" '$1 == id' "$TASKS")
if [ -z "$ROW" ]; then
    echo "error: no task $TASK_ID in $TASKS" >&2
    exit 2
fi

IFS=$'\t' read -r _ GENOME_JSON OUT_DIR RUN_NAME GENOME_NUMBER TIER REPLICATE ARM LEARNING_RATE ENTROPY_COEF <<<"$ROW"

if [ -f "${OUT_DIR}/refined_genome_${GENOME_NUMBER}.json" ]; then
    echo "task $TASK_ID ($RUN_NAME genome $GENOME_NUMBER, $TIER, replicate $REPLICATE, $ARM) already done"
    exit 0
fi

COMMAND=(
    python3.12 -m src.examples.refine_genome
    --genome_json "$GENOME_JSON"
    --out_dir "$OUT_DIR"
    --set "learning_rate=${LEARNING_RATE}"
    --set "entropy_coef=${ENTROPY_COEF}"
    --no-save_circuit
    --logging_level INFO
)

# DRY_RUN prints the command instead of running it, so the task list can be
# checked without a cluster
if [ -n "${DRY_RUN:-}" ]; then
    printf '%s\n' "${COMMAND[*]}"
    exit 0
fi

# one CPU per task: keep PyTorch and the numeric libraries from oversubscribing it
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# refine_genome imports the search entry points, which import mpi4py, and
# mpi4py calls MPI_Init on import. Inside a Slurm job that initialization tries
# to reach Slurm's PMI and aborts. refine_genome never uses MPI, so tell mpi4py
# (4.0 or later) not to initialize it at all.
export MPI4PY_RC_INITIALIZE=0

# In case the cluster's mpi4py predates that variable, also set up MPI the way
# scripts/exaqc_rl_job.sh does, which is known to initialize under srun.
export PMIX_MCA_psec="^munge"
spack load openmpi /ttqroyz

source /home/tjdvse/envs/exaqc/bin/activate

echo "task $TASK_ID: $RUN_NAME genome $GENOME_NUMBER ($TIER), replicate $REPLICATE, arm $ARM"
srun "${COMMAND[@]}"
