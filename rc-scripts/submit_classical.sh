#!/bin/bash
#
# Submits the classical image baselines: every dataset in DATASETS x every
# model in MODELS for seeds <min_seed>..<max_seed>, one Slurm job each.
#
#   bash submit_classical.sh <min_seed> <max_seed>
#
# Each job runs classical_job.sh (kept next to this script); its training
# settings are environment variables documented there.
#
# Settings used only here:
#   DATASETS     [cifar10]
#   MODELS       [resnet18 resnet50]
#   GRES         GPU request, e.g. gpu:a100:1; empty = no GPU   [gpu:1]
#                (with GRES= also set DEVICE=cpu)
#   LOG_DIR      [/home/dk7405/logs/exaqc]
#   ACCOUNT [examm-rnn]  PARTITION [tigris]  CPUS [4]  MEM [64GB]
#   TIME_LIMIT   [3-00:00:00]
#   DRY_RUN      1 = print each job's sbatch line and command without submitting

set -eu

if [ $# -ne 2 ]; then
    echo "usage: $0 <min_seed> <max_seed>" >&2
    exit 2
fi
MIN_SEED=$1
MAX_SEED=$2

JOB_SCRIPT="$(cd "$(dirname "$0")" && pwd)/classical_job.sh"
[ -f "$JOB_SCRIPT" ] || { echo "error: $JOB_SCRIPT not found" >&2; exit 2; }

read -r -a DATASETS <<< "${DATASETS:-cifar10}"
read -r -a MODELS <<< "${MODELS:-resnet18 resnet50}"
GRES=${GRES-gpu:gh200:1}
LOG_DIR=${LOG_DIR:-/home/dk7405/logs/exaqc}
ACCOUNT=${ACCOUNT:-examm-rnn}
PARTITION=${PARTITION:-tigris}
CPUS=${CPUS:-4}
MEM=${MEM:-64GB}
TIME_LIMIT=${TIME_LIMIT:-3-00:00:00}

[ -n "${DRY_RUN:-}" ] || mkdir -p "$LOG_DIR"
submitted=0
for seed in $(seq "$MIN_SEED" "$MAX_SEED"); do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            name="classical_${dataset}_${model}_${seed}"
            sbatch_args=(
                -J "$name" -A "$ACCOUNT" -p "$PARTITION" -t "$TIME_LIMIT"
                -o "$LOG_DIR/${name}_%j.o" -e "$LOG_DIR/${name}_%j.e"
                --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM"
            )
            [ -z "$GRES" ] || sbatch_args+=(--gres="$GRES")
            if [ -n "${DRY_RUN:-}" ]; then
                echo "=== $name"
                echo "sbatch ${sbatch_args[*]} $JOB_SCRIPT $dataset $model $seed"
                DRY_RUN=1 bash "$JOB_SCRIPT" "$dataset" "$model" "$seed"
            else
                sbatch "${sbatch_args[@]}" "$JOB_SCRIPT" "$dataset" "$model" "$seed"
            fi
            submitted=$((submitted + 1))
        done
    done
done
echo "$submitted job(s) $([ -n "${DRY_RUN:-}" ] && echo printed || echo submitted)"
