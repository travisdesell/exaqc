#!/bin/bash
#
# Submits EXAQC reinforcement-learning runs to Slurm (RIT Tigris): every environment
# under every island topology and a steady-state population, one job per run.
#
#   bash submit_rl_all.sh <min_run> <max_run> <mode>
#
#   <min_run> <max_run>  run numbers to do (inclusive); run i uses --training_seed i
#   <mode>               fixed, sho, sho_training, sho_operators or sho_mutations
#                        (see exaqc_rl_all_job.sh); sho_operators and
#                        sho_mutations run on islands only
#
# Each job runs exaqc_rl_all_job.sh (kept next to this script), which builds
# the search command. Search and cluster settings are environment variables
# documented there (N_ISLANDS, MAX_ISLAND_SIZE, NUMBER_GENOMES, EPISODES, QOM,
# POP_SIZE, REPO, OUT_ROOT, ...); sbatch passes them through to the job.
# A run whose archive already exists is resumed (--restart defaults to auto),
# so resubmitting the same command continues unfinished runs.
#
# Settings used only here:
#   ENVIRONMENTS      environments to run          [acrobot cartpole frozenlake mountaincar_continuous walker2d]
#   POPULATIONS       populations to run           [fully_connected ring star 2d_mesh tree random steady_state]
#   LOG_DIR           Slurm output and error files [/home/dk7405/logs/exaqc]
#   ACCOUNT           Slurm account                [examm-rnn]
#   PARTITION         Slurm partition              [tigris]
#   NTASKS            MPI ranks per job (1 master + workers) [144]
#   TIME_LIMIT        per job                      [5-00:00:00]
#   MEM               per job                      [256GB]
#   DRY_RUN           1 = print each job's sbatch line and search command without submitting

set -eu

if [ $# -ne 3 ]; then
    echo "usage: $0 <min_run> <max_run> <fixed|sho|sho_training|sho_operators|sho_mutations>" >&2
    exit 2
fi
MIN_RUN=$1
MAX_RUN=$2
MODE=$3
case "$MODE" in
    fixed|sho|sho_training|sho_operators|sho_mutations) ;;
    *) echo "error: unknown mode '$MODE'" >&2; exit 2 ;;
esac

JOB_SCRIPT="$(cd "$(dirname "$0")" && pwd)/exaqc_rl_all_job.sh"
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "error: $JOB_SCRIPT not found; keep it next to this script" >&2
    exit 2
fi

read -r -a ENVIRONMENTS <<< "${ENVIRONMENTS:-acrobot cartpole frozenlake mountaincar_continuous walker2d}"
read -r -a POPULATIONS <<< "${POPULATIONS:-fully_connected ring star 2d_mesh tree random steady_state}"
LOG_DIR=${LOG_DIR:-/home/dk7405/logs/exaqc}
ACCOUNT=${ACCOUNT:-examm-rnn}
PARTITION=${PARTITION:-tigris}
NTASKS=${NTASKS:-144}
TIME_LIMIT=${TIME_LIMIT:-5-00:00:00}
MEM=${MEM:-256GB}

[ -n "${DRY_RUN:-}" ] || mkdir -p "$LOG_DIR"
submitted=0
for run in $(seq "$MIN_RUN" "$MAX_RUN"); do
    for environment in "${ENVIRONMENTS[@]}"; do
        for population in "${POPULATIONS[@]}"; do
            # the job script's dry run prints the command, whose --out_dir
            # holds the archive name; the job is named after it so the queue
            # and the archive directory carry the same label. This also checks
            # the arguments before anything is submitted.
            command=$(DRY_RUN=1 bash "$JOB_SCRIPT" "$environment" "$population" "$MODE" "$run")
            out_dir=$(printf '%s\n' "$command" | sed -E 's/.*--out_dir ([^ ]+).*/\1/')
            name=$(basename "$out_dir")

            sbatch_args=(
                -J "$name"
                -A "$ACCOUNT" -p "$PARTITION"
                -t "$TIME_LIMIT"
                -o "$LOG_DIR/${name}_%j.o"
                -e "$LOG_DIR/${name}_%j.e"
                --nodes=1 --ntasks="$NTASKS" --ntasks-per-node="$NTASKS"
                --cpus-per-task=1 --mem="$MEM"
            )
            if [ -n "${DRY_RUN:-}" ]; then
                echo "=== $name"
                echo "sbatch ${sbatch_args[*]} $JOB_SCRIPT $environment $population $MODE $run"
                echo "$command"
            else
                sbatch "${sbatch_args[@]}" "$JOB_SCRIPT" "$environment" "$population" "$MODE" "$run"
            fi
            submitted=$((submitted + 1))
        done
    done
done
if [ -n "${DRY_RUN:-}" ]; then
    echo "$submitted job(s) printed"
else
    echo "$submitted job(s) submitted"
fi