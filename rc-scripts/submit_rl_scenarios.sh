#!/bin/bash
#
# Submits the RL scenarios for the SHO paper to Slurm (RIT Tigris), with the
# settings of the sho02/sho03 SHO islands runs: PPO, 200 training episodes per
# genome, greedy evaluation, 10,000 genomes, 1-7 mutations per child, 2-7
# n-ary crossover parents, 10 islands of 10, and the five paper environments.
#
#   bash submit_rl_scenarios.sh <min_run> <max_run> [scenario ...]
#
# Scenarios (fixed_pop, sho_pop and fixed_islands when none is given):
#   fixed_pop      fixed hyperparameters, steady-state population of POP_SIZE
#   sho_pop        SHO (training settings only), steady-state population of POP_SIZE
#   fixed_islands  fixed hyperparameters, islands under every topology
#                  (fully connected, ring, star, 2 x 5 mesh, tree with 2
#                  children, random with 1-3 edges)
#   sho_islands    SHO (training and operator settings), islands under every
#                  topology. The sho02/sho03 runs already cover these with the
#                  same settings; this writes new archives under OUT_ROOT, so
#                  use it only to add seeds or redo them in one place.
#   training_islands   SHO on the training settings only (learning rate, Adam,
#                      gamma, entropy coefficient, GAE lambda, PPO clip), same
#                      islands
#   operators_islands  SHO on the operator settings only (crossover rates,
#                      mutation count, mutation weights), same islands
#   mutations_islands  SHO on the mutation settings only (mutation count and
#                      mutation weights), same islands
# Only the first three are in the default set; name the others to submit them.
#
# Uses submit_rl_all.sh and exaqc_rl_all_job.sh, which must sit next to this
# script. Other settings (POP_SIZE, NTASKS, TIME_LIMIT, OUT_ROOT, ENVIRONMENTS,
# TOPOLOGIES, DRY_RUN, ...) pass through as environment variables; see those
# scripts.
#   POP_SIZE    steady-state population size [100, equal to 10 islands x 10]
#   TOPOLOGIES  island topologies for the islands scenarios
#               [fully_connected ring star 2d_mesh tree random]

set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <min_run> <max_run> [fixed_pop sho_pop fixed_islands sho_islands training_islands operators_islands mutations_islands]" >&2
    exit 2
fi
MIN_RUN=$1
MAX_RUN=$2
shift 2
SCENARIOS=("$@")
[ ${#SCENARIOS[@]} -gt 0 ] || SCENARIOS=(fixed_pop sho_pop fixed_islands)

SUBMIT="$(cd "$(dirname "$0")" && pwd)/submit_rl_all.sh"
if [ ! -f "$SUBMIT" ]; then
    echo "error: $SUBMIT not found; keep it next to this script" >&2
    exit 2
fi

# shared by every scenario
export NUMBER_GENOMES=10000
export EPISODES=200
export MUTATIONS="1 7"
export PARENTS="2 7"
export N_ISLANDS=10
export MAX_ISLAND_SIZE=10
export MESH="2 5"
export TREE_CHILDREN=2
export RANDOM_EDGES="1 3"
POP_SIZE=${POP_SIZE:-100}
TOPOLOGIES=${TOPOLOGIES:-fully_connected ring star 2d_mesh tree random}

steady_state() {  # <mode>
    POP_SIZE=$POP_SIZE POPULATIONS=steady_state bash "$SUBMIT" "$MIN_RUN" "$MAX_RUN" "$1"
}

islands() {  # <mode>
    POPULATIONS=$TOPOLOGIES bash "$SUBMIT" "$MIN_RUN" "$MAX_RUN" "$1"
}

for scenario in "${SCENARIOS[@]}"; do
    echo "##### $scenario"
    case "$scenario" in
        fixed_pop) steady_state fixed ;;
        sho_pop) steady_state sho ;;
        fixed_islands) islands fixed ;;
        sho_islands) islands sho ;;
        training_islands) islands sho_training ;;
        operators_islands) islands sho_operators ;;
        mutations_islands) islands sho_mutations ;;
        *) echo "error: unknown scenario: $scenario" >&2; exit 2 ;;
    esac
done