#!/bin/bash
#
# Submits the image scenarios for the SHO paper to Slurm (RIT Tigris), with the
# settings of the sho04 SHO islands runs: linear encoder, 1,000 genomes, 1-5
# mutations per child, 2-5 n-ary crossover parents, 6 islands of 10, MNIST and
# Fashion-MNIST.
#
#   bash submit_image_scenarios.sh <min_run> <max_run> [scenario ...]
#
# Scenarios (fixed_pop, sho_pop and fixed_islands when none is given):
#   fixed_pop      fixed hyperparameters, steady-state population of POP_SIZE
#   sho_pop        SHO (training settings only), steady-state population of POP_SIZE
#   fixed_islands  fixed hyperparameters, islands under every topology
#                  (fully connected, ring, star, 2 x 3 mesh, tree with 2
#                  children, random with 1-3 edges)
#   sho_islands    SHO (training and operator settings), islands under every
#                  topology. The sho04 runs already cover these with the same
#                  settings; this writes new archives under OUT_ROOT.
#
# Uses submit_image_all.sh and exaqc_image_all_job.sh, which must sit next to
# this script. Other settings (POP_SIZE, NTASKS, TIME_LIMIT, OUT_ROOT,
# DATASETS, TOPOLOGIES, DRY_RUN, ...) pass through as environment variables;
# see those scripts.
#   POP_SIZE    steady-state population size [60, equal to 6 islands x 10]
#   TOPOLOGIES  island topologies for the islands scenarios
#               [fully_connected ring star 2d_mesh tree random]

set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 <min_run> <max_run> [fixed_pop sho_pop fixed_islands sho_islands]" >&2
    exit 2
fi
MIN_RUN=$1
MAX_RUN=$2
shift 2
SCENARIOS=("$@")
[ ${#SCENARIOS[@]} -gt 0 ] || SCENARIOS=(fixed_pop sho_pop fixed_islands)

SUBMIT="$(cd "$(dirname "$0")" && pwd)/submit_image_all.sh"
if [ ! -f "$SUBMIT" ]; then
    echo "error: $SUBMIT not found; keep it next to this script" >&2
    exit 2
fi

# shared by every scenario
export NUMBER_GENOMES=${NUMBER_GENOMES:-1000}
export MUTATIONS="1 5"
export PARENTS="2 5"
export N_ISLANDS=6
export MAX_ISLAND_SIZE=10
export MESH="2 3"
export TREE_CHILDREN=2
export RANDOM_EDGES="1 3"
POP_SIZE=${POP_SIZE:-60}
TOPOLOGIES=${TOPOLOGIES:-fully_connected ring star 2d_mesh tree random}

steady_state() {  # <sho|fixed>
    POP_SIZE=$POP_SIZE POPULATIONS=steady_state bash "$SUBMIT" "$MIN_RUN" "$MAX_RUN" "$1"
}

islands() {  # <sho|fixed>
    POPULATIONS=$TOPOLOGIES bash "$SUBMIT" "$MIN_RUN" "$MAX_RUN" "$1"
}

for scenario in "${SCENARIOS[@]}"; do
    echo "##### $scenario"
    case "$scenario" in
        fixed_pop) steady_state fixed ;;
        sho_pop) steady_state sho ;;
        fixed_islands) islands fixed ;;
        sho_islands) islands sho ;;
        *) echo "error: unknown scenario: $scenario" >&2; exit 2 ;;
    esac
done