#!/bin/bash
#
# Submits the UCI scenarios for the SHO paper to Slurm (RIT Tigris), with the
# settings of the earlier sho/ runs: 2,000 genomes, 1-5 mutations per child,
# 2-5 n-ary crossover parents, all four datasets.
#
#   bash submit_uci_scenarios.sh <min_run> <max_run> [scenario ...]
#
# Scenarios (fixed_pop, sho_pop and fixed_islands when none is given):
#   fixed_pop      fixed hyperparameters, steady-state population of POP_SIZE
#   sho_pop        SHO (training settings only), steady-state population of POP_SIZE
#   fixed_islands  fixed hyperparameters, islands: 5 x 10 for fully connected,
#                  ring, tree (2 children) and random (1-2 edges); 6 x 10 for
#                  the 2 x 3 mesh
#   training_islands   SHO on the training settings only (learning rate, Adam),
#                      same islands
#   operators_islands  SHO on the operator settings only (crossover rates,
#                      mutation count, mutation weights), same islands
#   mutations_islands  SHO on the mutation settings only (mutation count and
#                      mutation weights), same islands
# The last three are not in the default set; name them to submit them.
#
# Uses submit_uci_all.sh and exaqc_uci_all_job.sh, which must sit next to this
# script. Other settings (POP_SIZE, NTASKS, TIME_LIMIT, OUT_ROOT, DATASETS,
# DRY_RUN, ...) pass through as environment variables; see those scripts.
#   POP_SIZE  steady-state population size [30, as in the earlier runs]

set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <min_run> <max_run> [fixed_pop sho_pop fixed_islands training_islands operators_islands mutations_islands]" >&2
    exit 2
fi
MIN_RUN=$1
MAX_RUN=$2
shift 2
SCENARIOS=("$@")
[ ${#SCENARIOS[@]} -gt 0 ] || SCENARIOS=(fixed_pop sho_pop fixed_islands)

SUBMIT="$(cd "$(dirname "$0")" && pwd)/submit_uci_all.sh"
if [ ! -f "$SUBMIT" ]; then
    echo "error: $SUBMIT not found; keep it next to this script" >&2
    exit 2
fi

# shared by every scenario
export NUMBER_GENOMES=2000
export MUTATIONS="1 5"
export PARENTS="2 5"
POP_SIZE=${POP_SIZE:-30}

steady_state() {  # <mode>
    POP_SIZE=$POP_SIZE POPULATIONS=steady_state bash "$SUBMIT" "$MIN_RUN" "$MAX_RUN" "$1"
}

islands() {  # <mode>
    # MESH only has to pass the mesh check here; this call runs no mesh
    N_ISLANDS=5 MAX_ISLAND_SIZE=10 MESH="1 5" TREE_CHILDREN=2 RANDOM_EDGES="1 2" \
        POPULATIONS="fully_connected ring star tree random" bash "$SUBMIT" "$MIN_RUN" "$MAX_RUN" "$1"
    N_ISLANDS=6 MAX_ISLAND_SIZE=10 MESH="2 3" \
        POPULATIONS=2d_mesh bash "$SUBMIT" "$MIN_RUN" "$MAX_RUN" "$1"
}

for scenario in "${SCENARIOS[@]}"; do
    echo "##### $scenario"
    case "$scenario" in
        fixed_pop) steady_state fixed ;;
        sho_pop) steady_state sho ;;
        fixed_islands) islands fixed ;;
        training_islands) islands sho_training ;;
        operators_islands) islands sho_operators ;;
        mutations_islands) islands sho_mutations ;;
        *) echo "error: unknown scenario: $scenario" >&2; exit 2 ;;
    esac
done