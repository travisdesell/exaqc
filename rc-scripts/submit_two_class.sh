#!/bin/bash
#
# Submits the 2-class EXAQC runs: every dataset in DATASETS for runs
# <min_run>..<max_run>, one Slurm job per run.
#
#   bash submit_two_class.sh <min_run> <max_run>
#
# Each job runs exaqc_two_class_job.sh (kept next to this script); its settings
# (CLASSES, POP_SIZE, NUMBER_GENOMES, QUBITS, ENCODER_CONFIG, ...) are
# environment variables documented there. CLASSES applies to every dataset in
# the call, so set it only when submitting one dataset.
#
# Settings used only here:
#   DATASETS     datasets to run               [mnist fashion_mnist]
#                CIFAR-10 needs its own encoder config, so submit it separately:
#                DATASETS=cifar10 ENCODER_CONFIG=<3-layer CNN config> bash submit_two_class.sh 1 10
#   LOG_DIR      Slurm output and error files  [/home/dk7405/logs/exaqc]
#   ACCOUNT      [examm-rnn]   PARTITION [tigris]   NTASKS [144]
#   TIME_LIMIT   [5-00:00:00]  MEM [256GB]
#   DRY_RUN      1 = print each job's sbatch line and command without submitting

set -eu

if [ $# -ne 2 ]; then
    echo "usage: $0 <min_run> <max_run>" >&2
    exit 2
fi
MIN_RUN=$1
MAX_RUN=$2

JOB_SCRIPT="$(cd "$(dirname "$0")" && pwd)/exaqc_two_class_job.sh"
[ -f "$JOB_SCRIPT" ] || { echo "error: $JOB_SCRIPT not found" >&2; exit 2; }

read -r -a DATASETS <<< "${DATASETS:-mnist fashion_mnist}"
LOG_DIR=${LOG_DIR:-/home/dk7405/logs/exaqc}
ACCOUNT=${ACCOUNT:-examm-rnn}
PARTITION=${PARTITION:-tigris}
NTASKS=${NTASKS:-144}
TIME_LIMIT=${TIME_LIMIT:-5-00:00:00}
MEM=${MEM:-256GB}

[ -n "${DRY_RUN:-}" ] || mkdir -p "$LOG_DIR"
submitted=0
for run in $(seq "$MIN_RUN" "$MAX_RUN"); do
    for dataset in "${DATASETS[@]}"; do
        command=$(DRY_RUN=1 bash "$JOB_SCRIPT" "$dataset" "$run")
        out_dir=$(printf '%s\n' "$command" | sed -E 's/.*--out_dir ([^ ]+).*/\1/')
        name=$(basename "$out_dir")
        sbatch_args=(
            -J "$name" -A "$ACCOUNT" -p "$PARTITION" -t "$TIME_LIMIT"
            -o "$LOG_DIR/${name}_%j.o" -e "$LOG_DIR/${name}_%j.e"
            --nodes=1 --ntasks="$NTASKS" --ntasks-per-node="$NTASKS"
            --cpus-per-task=1 --mem="$MEM"
        )
        if [ -n "${DRY_RUN:-}" ]; then
            echo "=== $name"
            echo "sbatch ${sbatch_args[*]} $JOB_SCRIPT $dataset $run"
            echo "$command"
        else
            sbatch "${sbatch_args[@]}" "$JOB_SCRIPT" "$dataset" "$run"
        fi
        submitted=$((submitted + 1))
    done
done
echo "$submitted job(s) $([ -n "${DRY_RUN:-}" ] && echo printed || echo submitted)"
