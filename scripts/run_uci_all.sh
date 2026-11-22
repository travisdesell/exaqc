#!/bin/bash
#
# Runs EXAQC on all four UCI datasets under every island topology and a
# steady-state population, locally with mpiexec.
#
#   bash run_uci_all.sh <min_run> <max_run> <sho|fixed>
#
#   <min_run> <max_run>  run numbers to do (inclusive); run i uses --seed i
#   sho                  co-evolve hyperparameters with SHO
#   fixed                use the configured hyperparameters for every genome
#
# Run from the repository root with the venv active, e.g.
#   source .venv/bin/activate
#   bash run_uci_all.sh 1 10 sho
#
# With sho, islands runs evolve the training settings (learning rate, Adam)
# and the operator settings (crossover rates, mutation count, mutation
# weights). EXAQC only allows operator settings to be tuned on islands, so
# steady-state runs evolve the training settings only. Set SHO_OPERATORS=0 to
# evolve the training settings only on islands too, so both match.
#
# Loop order is run -> dataset -> population, so each seed is finished across
# every configuration before the next one starts. A run whose archive already
# exists is resumed (--restart defaults to auto), so the script can be
# stopped and started again.
#
# Overrides (environment variables, defaults in brackets):
#   DATASETS          datasets to run              [iris seeds wine breast_cancer]
#   POPULATIONS       populations to run           [fully_connected ring star 2d_mesh tree random steady_state]
#   N_ISLANDS         islands per run              [6]
#   MAX_ISLAND_SIZE   genomes per island           [10]
#   MESH              2d_mesh dimensions, must multiply to N_ISLANDS [2 3]
#   TREE_CHILDREN     children per node for tree   [2]
#   RANDOM_EDGES      min and max edges for random [1 3]
#   POP_SIZE          steady-state population size [N_ISLANDS * MAX_ISLAND_SIZE]
#   NUMBER_GENOMES    genomes per run              [1000]
#   NP                MPI processes (1 master + workers) [8]
#   OUT_ROOT          where the archives go        [artifacts/uci_runs]
#   SHO_OPERATORS     1 = also tune operator settings on islands [1]
#   PYTHON            python to run                [python3]
#   DRY_RUN           1 = print the commands without running them

set -eu

if [ $# -ne 3 ]; then
    echo "usage: $0 <min_run> <max_run> <sho|fixed>" >&2
    exit 2
fi
MIN_RUN=$1
MAX_RUN=$2
MODE=$3
case "$MODE" in
    sho|fixed) ;;
    *) echo "error: third argument must be sho or fixed, not '$MODE'" >&2; exit 2 ;;
esac

read -r -a DATASETS <<< "${DATASETS:-iris seeds wine breast_cancer}"
read -r -a POPULATIONS <<< "${POPULATIONS:-fully_connected ring star 2d_mesh tree random steady_state}"
N_ISLANDS=${N_ISLANDS:-6}
MAX_ISLAND_SIZE=${MAX_ISLAND_SIZE:-10}
read -r -a MESH <<< "${MESH:-2 3}"
TREE_CHILDREN=${TREE_CHILDREN:-2}
read -r -a RANDOM_EDGES <<< "${RANDOM_EDGES:-1 3}"
# equal total capacity by default, so steady-state is not just a smaller population
POP_SIZE=${POP_SIZE:-$((N_ISLANDS * MAX_ISLAND_SIZE))}
NUMBER_GENOMES=${NUMBER_GENOMES:-1000}
NP=${NP:-8}
OUT_ROOT=${OUT_ROOT:-artifacts/uci_runs}
SHO_OPERATORS=${SHO_OPERATORS:-1}
PYTHON=${PYTHON:-python3.11}

if [ $((MESH[0] * MESH[1])) -ne "$N_ISLANDS" ]; then
    echo "error: MESH=${MESH[*]} does not multiply to N_ISLANDS=$N_ISLANDS" >&2
    exit 2
fi

# The ranges used for the paper's runs (Kini et al., GECCO '23 for the
# training settings): burn-in range, then the full allowed range.
TRAINING_TUNE=(
    learning_rate=log:1e-3:5e-2:1e-5:0.3
    adam_beta1=linear:0.9:0.99
    adam_beta2=linear:0.9:0.99
    adam_epsilon=log:1e-9:1e-8
)
OPERATOR_TUNE=(
    binary_crossover_rate=linear:0.05:0.15:0:1
    n_ary_crossover_rate=linear:0.05:0.15:0:1
    exponential_crossover_rate=linear:0.05:0.15:0:1
    mutation_count=int:1:3:1:10
    add_gate_weight=linear:8:14:0:30
    reorder_gate_weight=linear:1:3:0:10
    qubit_swap_weight=linear:1:3:0:10
    enable_gate_weight=linear:0.5:1.5:0:10
    disable_gate_weight=linear:1:3:0:10
    clone_weight=linear:1:3:0:10
    mutate_some_weights_weight=linear:1:3:0:10
    mutate_all_weights_weight=linear:1:3:0:10
)

# identity encoding needs one input qubit per feature; breast_cancer is
# linearly encoded instead, with a clipped decoder for its two classes
dataset_arguments() {
    case "$1" in
        iris) echo "--input_qubits 4 --output_qubits 2 --encoding identity --decoding linear" ;;
        seeds) echo "--input_qubits 7 --output_qubits 2 --encoding identity --decoding linear" ;;
        wine) echo "--input_qubits 13 --output_qubits 2 --encoding identity --decoding linear" ;;
        breast_cancer) echo "--input_qubits 8 --output_qubits 1 --encoding linear --decoding clipped" ;;
        *) echo "error: unknown UCI dataset: $1" >&2; exit 2 ;;
    esac
}

# prints "<archive tag> <population arguments...>" for a population name
population_arguments() {
    local islands="islands --n_islands $N_ISLANDS --max_island_size $MAX_ISLAND_SIZE --islands_to_extinct 1"
    local tag="i${N_ISLANDS}_p${MAX_ISLAND_SIZE}"
    case "$1" in
        fully_connected|ring|star) echo "${tag}_$1 $islands --topology $1" ;;
        2d_mesh) echo "${tag}_2d_mesh_${MESH[0]}x${MESH[1]} $islands --topology 2d_mesh ${MESH[0]} ${MESH[1]}" ;;
        tree) echo "${tag}_tree_${TREE_CHILDREN} $islands --topology tree $TREE_CHILDREN" ;;
        random) echo "${tag}_random_${RANDOM_EDGES[0]}_${RANDOM_EDGES[1]} $islands --topology random ${RANDOM_EDGES[0]} ${RANDOM_EDGES[1]}" ;;
        steady_state) echo "pop${POP_SIZE}_steady_state steady_state --max_population_size $POP_SIZE" ;;
        *) echo "error: unknown population: $1" >&2; exit 2 ;;
    esac
}

for run in $(seq "$MIN_RUN" "$MAX_RUN"); do
    for dataset in "${DATASETS[@]}"; do
        for population in "${POPULATIONS[@]}"; do
            read -r -a dataset_args <<< "$(dataset_arguments "$dataset")"
            read -r -a population_args <<< "$(population_arguments "$population")"
            tag=${population_args[0]}
            population_args=("${population_args[@]:1}")

            sho_args=()
            if [ "$MODE" = sho ]; then
                tune=("${TRAINING_TUNE[@]}")
                if [ "$population" != steady_state ] && [ "$SHO_OPERATORS" = 1 ]; then
                    tune+=("${OPERATOR_TUNE[@]}")
                fi
                sho_args=(--hyperparameter_strategy simplex --sho_tune "${tune[@]}"
                          --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5)
            fi

            out_dir="$OUT_ROOT/${dataset}_${MODE}_${tag}_${run}"
            # the population subcommand and its arguments come last:
            # --topology takes any number of values
            command=(
                mpiexec --oversubscribe -n "$NP" "$PYTHON" -m mpi4py -m src.examples.classification
                --logging_level INFO
                --dataset "$dataset"
                --number_genomes "$NUMBER_GENOMES"
                "${dataset_args[@]}"
                --batch_size 3
                --mutation_strategy uniform 1 3
                --parent_strategy uniform 2 5
                --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1
                -qim ry -qom probs
                ${sho_args[@]+"${sho_args[@]}"}
                --seed "$run"
                --out_dir "$out_dir"
                "${population_args[@]}"
            )

            echo "=== run $run: $dataset, $population, $MODE -> $out_dir"
            if [ -n "${DRY_RUN:-}" ]; then
                printf '%s\n' "${command[*]}"
            else
                "${command[@]}"
            fi
        done
    done
done
