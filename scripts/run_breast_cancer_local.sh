#!/usr/bin/env bash
# Local single-objective breast-cancer runs. Each population strategy is
# launched separately, one after another. Four MPI ranks. Outputs under
# artifacts/.
#
#   bash scripts/run_breast_cancer_local.sh
#
# Cluster / 12-rank replica: scripts/run_breast_cancer.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

REPLICATE="${1:-1}"
LOSS_LABEL="${2:-per_class}"
OUT_DIR="${3:-artifacts}"
NPROC="${NPROC:-4}"
GENOMES="${GENOMES:-1000}"
POP_SIZE="${POP_SIZE:-30}"
SPECIES_THRESHOLD="${SPECIES_THRESHOLD:-0.6}"
DATASET="breast_cancer"
INPUT_QUBITS=8
OUTPUT_QUBITS=1

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="python3"
fi

run_classification() {
  local run_dir="$1"
  shift
  mkdir -p "$run_dir"
  echo "===== ${DATASET} -> ${run_dir} ====="
  mpiexec --oversubscribe -n "$NPROC" "$PY" -m src.examples.classification \
    --logging_level INFO \
    --dataset "$DATASET" \
    --number_genomes "$GENOMES" \
    --input_qubits "$INPUT_QUBITS" \
    --output_qubits "$OUTPUT_QUBITS" \
    --batch_size 3 \
    --mutation_strategy uniform 1 3 \
    --parent_strategy uniform 5 5 \
    --binary_crossover_rate 0.1 \
    --n_ary_crossover_rate 0.1 \
    --exponential_crossover_rate 0.1 \
    -qim amplitude \
    -qom probs \
    --encoding identity \
    --decoding clipped \
    --out_dir "$run_dir" \
    "$@"
}

# 1) default population
run_classification \
  "${OUT_DIR}/breast_steady_i${POP_SIZE}_${LOSS_LABEL}_${REPLICATE}" \
  steady_state --max_population_size "$POP_SIZE"

# 2) islands, one layout at a time (same total capacity as pop 30)
run_classification \
  "${OUT_DIR}/breast_islands_i5_p6_best_${LOSS_LABEL}_${REPLICATE}" \
  islands \
  --n_islands 5 \
  --max_island_size 6 \
  --islands_to_extinct 1 \
  --genomes_before_extinction 50 \
  --genomes_for_next_extinction 200 \
  --primary_parent best \
  --intra_island_crossover_rate 0.5 \
  --topology fully_connected

run_classification \
  "${OUT_DIR}/breast_islands_i10_p3_best_${LOSS_LABEL}_${REPLICATE}" \
  islands \
  --n_islands 10 \
  --max_island_size 3 \
  --islands_to_extinct 2 \
  --genomes_before_extinction 50 \
  --genomes_for_next_extinction 200 \
  --primary_parent best \
  --intra_island_crossover_rate 0.5 \
  --topology fully_connected

run_classification \
  "${OUT_DIR}/breast_islands_i30_p1_best_${LOSS_LABEL}_${REPLICATE}" \
  islands \
  --n_islands 30 \
  --max_island_size 1 \
  --islands_to_extinct 3 \
  --genomes_before_extinction 50 \
  --genomes_for_next_extinction 200 \
  --primary_parent best \
  --intra_island_crossover_rate 0.5 \
  --topology fully_connected

# 3) historical speciation
run_classification \
  "${OUT_DIR}/breast_speciation_i${POP_SIZE}_t${SPECIES_THRESHOLD}_${LOSS_LABEL}_${REPLICATE}" \
  steady_state_speciation \
  --max_population_size "$POP_SIZE" \
  --species_threshold "$SPECIES_THRESHOLD"

echo "===== all breast_cancer population-strategy runs finished ====="
