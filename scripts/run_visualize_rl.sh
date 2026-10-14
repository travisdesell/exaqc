#!/usr/bin/env bash
# Roll out the best genome from a walker2d run with src.examples.visualize_rl.
# Run from the exaqc repo root:  bash visualize_walker2d_minibatch.sh
#
# Overrides (env vars):
#   RUN_DIR        run directory        (default: artifacts/exaqc_minibatch_2d_mesh_4x5_walker2d_i20_1)
#   GENOME_NUMBER  load this genome from RUN_DIR/genomes.sqlar instead of best_fitness.json
#   EPISODES       episodes to play     (default: 3)
#   EVAL_POLICY    match|greedy|stochastic (default: match = regime it was scored under)
#   MAX_STEPS      steps per episode    (default: from the genome)
#   SEED           env seed             (default: random)
#   OUTPUT         GIF path             (default: RUN_DIR/rollout_<genome>.gif)
#   LIVE=1         open a live MuJoCo window instead of writing a GIF
set -euo pipefail

RUN_DIR="${RUN_DIR:-artifacts/exaqc_minibatch_2d_mesh_4x5_walker2d_i20_1}"
EPISODES="${EPISODES:-3}"
EVAL_POLICY="${EVAL_POLICY:-match}"
PYTHON="${PYTHON:-.venv/bin/python3.11}"

if [[ -n "${GENOME_NUMBER:-}" ]]; then
    SOURCE=(--archive "$RUN_DIR" --genome_number "$GENOME_NUMBER")
    TAG="genome_${GENOME_NUMBER}"
else
    SOURCE=(--genome_json "$RUN_DIR/best_fitness.json")
    TAG="best_fitness"
fi

EXTRA=()
[[ -n "${MAX_STEPS:-}" ]] && EXTRA+=(--max_steps "$MAX_STEPS")
[[ -n "${SEED:-}" ]] && EXTRA+=(--seed "$SEED")
if [[ "${LIVE:-0}" != "1" ]]; then
    EXTRA+=(--output_file "${OUTPUT:-$RUN_DIR/rollout_${TAG}.gif}")
fi

"$PYTHON" -m src.examples.visualize_rl \
    "${SOURCE[@]}" \
    --env walker2d \
    --episodes "$EPISODES" \
    --eval_policy "$EVAL_POLICY" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --logging_level INFO
