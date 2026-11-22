#!/usr/bin/env bash
# Acrobot (Acrobot-v1) with a steady-state population on the Mac: one run with
# fixed hyperparameters, then one with SHO tuning the training settings.
# Settings mirror the acrobot case in rc-scripts/exaqc_rl_islands_sho_job.sh.
# Operator settings (crossover rates, mutation_count, mutation weights) are not
# tuned because --sho_tune rejects them under steady_state.
#
# Usage (from the repo root):
#   bash run_acrobot_steady_state.sh [SEED] [OUT_DIR]
# Overrides: NPROCS (MPI ranks, default 8), NUMBER_GENOMES (default 1000),
#   MAX_POPULATION_SIZE (default 50, same as 10 islands x 5), QOM (expval|probs),
#   RUNS ("fixed sho", or just one of them).

set -euo pipefail

SEED=${1:-1}
OUT_DIR=${2:-artifacts/acrobot}
NPROCS=${NPROCS:-8}
NUMBER_GENOMES=${NUMBER_GENOMES:-1000}
MAX_POPULATION_SIZE=${MAX_POPULATION_SIZE:-50}
QOM=${QOM:-expval}
RUNS=${RUNS:-"fixed sho"}

# 3 actions + value = 4 outputs: expval needs one qubit per output, probs
# needs ceil(log2(4)) = 2.
if [ "$QOM" = "probs" ]; then OUTPUT_QUBITS=2; else OUTPUT_QUBITS=4; fi

source .venv/bin/activate

COMMON=(
    -m src.examples.reinforcement_learning
    --algo ppo
    --logging_level INFO
    --env acrobot
    --input_qubits 4 --output_qubits "$OUTPUT_QUBITS"
    --episodes 200
    --number_genomes "$NUMBER_GENOMES"
    --mutation_strategy uniform 1 7
    --parent_strategy uniform 2 7
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1
    -qim ry -qom "$QOM" --encoding linear --decoding linear
    --training_seed "$SEED"
)

for RUN in $RUNS; do
    case "$RUN" in
        fixed)
            EXTRA=(--hyperparameter_strategy fixed)
            ;;
        sho)
            EXTRA=(
                --hyperparameter_strategy simplex
                --sho_tune learning_rate=log:1e-3:5e-2:1e-5:0.3
                    adam_beta1=linear:0.9:0.99 adam_beta2=linear:0.9:0.999 adam_epsilon=log:1e-9:1e-8
                --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5
            )
            ;;
        *) echo "error: unknown run: $RUN (use fixed or sho)" >&2; exit 2 ;;
    esac
    RUN_DIR="$OUT_DIR/acrobot_ss_p${MAX_POPULATION_SIZE}_${RUN}_${SEED}"
    echo "== $RUN -> $RUN_DIR"
    # The population strategy is a subcommand, so steady_state and its
    # flags come last.
    mpiexec --oversubscribe -n "$NPROCS" python3 "${COMMON[@]}" "${EXTRA[@]}" \
        --out_dir "$RUN_DIR" \
        steady_state --max_population_size "$MAX_POPULATION_SIZE"
done
