#!/bin/bash -l
#SBATCH -J exaqc_gate_param_plots
#SBATCH -t 0-05:00:00
#SBATCH -A cps
#SBATCH -p tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

# set -euo pipefail

# ============================================================
# Environment
# ============================================================

spack env activate default-ml-x86_64-25052701
source .venv/bin/activate


# ============================================================
# Configuration
# ============================================================

ARTIFACT_ROOT="./artifacts"
OUTPUT_ROOT="./outs/gate_parameter_evolution"
METRIC="target_metric"

mkdir -p "${OUTPUT_ROOT}"


# ============================================================
# Process each top-level experiment folder independently
#
# Supported layouts:
#
# 1. Single/base run:
#
#    artifacts/<experiment>/all_genomes/
#
# 2. Multiple repeated runs:
#
#    artifacts/<experiment>/runs/0/all_genomes/
#    artifacts/<experiment>/runs/1/all_genomes/
#    artifacts/<experiment>/runs/2/all_genomes/
#
# If BOTH exist, they are processed separately.
# ============================================================

for EXPERIMENT_DIR in "${ARTIFACT_ROOT}"/*; do

    [[ -d "${EXPERIMENT_DIR}" ]] || continue

    EXPERIMENT_NAME=$(basename "${EXPERIMENT_DIR}")

    BASE_ALL_GENOMES="${EXPERIMENT_DIR}/all_genomes"
    RUNS_DIR="${EXPERIMENT_DIR}/runs"

    echo
    echo "============================================================"
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "============================================================"


    # ========================================================
    # Case 1: Base all_genomes/
    #
    # artifacts/<experiment>/all_genomes/
    #
    # This is treated as ONE independent run.
    # It is NOT combined with runs/*.
    # ========================================================

    if [[ -d "${BASE_ALL_GENOMES}" ]]; then

        BASE_OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
        mkdir -p "${BASE_OUTPUT_DIR}"

        BASE_OUTPUT_FILE="${BASE_OUTPUT_DIR}/gates_parameters.png"

        echo
        echo "------------------------------------------------------------"
        echo "Base run found"
        echo "------------------------------------------------------------"
        echo "Input:"
        echo "  ${EXPERIMENT_DIR}"
        echo
        echo "all_genomes:"
        echo "  ${BASE_ALL_GENOMES}"
        echo
        echo "Output:"
        echo "  ${BASE_OUTPUT_FILE}"
        echo

        python3.11 -m src.analysis.plot_gate_parameter_evolution \
            --input_directories "${EXPERIMENT_DIR}" \
            --title "EXAQC Gates and Parameters" \
            --metric ${METRIC} \
            --output "${BASE_OUTPUT_FILE}"

        echo
        echo "Finished base run: ${EXPERIMENT_NAME}"
    fi


    # ========================================================
    # Case 2: runs/<i>/all_genomes/
    #
    # All repeated runs belonging to THIS experiment are
    # aggregated into one plot.
    #
    # The base all_genomes/ above is NOT included.
    # ========================================================

    if [[ -d "${RUNS_DIR}" ]]; then

        RUN_DIRS=()

        for RUN_DIR in "${RUNS_DIR}"/*; do

            [[ -d "${RUN_DIR}" ]] || continue

            if [[ -d "${RUN_DIR}/all_genomes" ]]; then
                RUN_DIRS+=("${RUN_DIR}")
            fi

        done


        if [[ ${#RUN_DIRS[@]} -gt 0 ]]; then

            RUNS_OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
            mkdir -p "${RUNS_OUTPUT_DIR}"

            # Use a different filename so that if both a base run
            # and runs/ exist, neither plot overwrites the other.
            RUNS_OUTPUT_FILE="${RUNS_OUTPUT_DIR}/gates_parameters_runs.png"

            echo
            echo "------------------------------------------------------------"
            echo "Repeated runs found"
            echo "------------------------------------------------------------"
            echo "Number of runs: ${#RUN_DIRS[@]}"
            echo

            for RUN_DIR in "${RUN_DIRS[@]}"; do
                echo "  ${RUN_DIR}/all_genomes"
            done

            echo
            echo "Output:"
            echo "  ${RUNS_OUTPUT_FILE}"
            echo

            python3.11 -m src.analysis.plot_gate_parameter_evolution \
                --input_directories "${RUN_DIRS[@]}" \
                --title "EXAQC Gates and Parameters" \
                --metric ${METRIC} \
                --output "${RUNS_OUTPUT_FILE}"

            echo
            echo "Finished repeated runs: ${EXPERIMENT_NAME}"

        else
            echo
            echo "runs/ exists, but no valid run/all_genomes directories found."
        fi
    fi


    # ========================================================
    # Nothing usable
    # ========================================================

    if [[ ! -d "${BASE_ALL_GENOMES}" ]] &&
       [[ ! -d "${RUNS_DIR}" ]]; then

        echo "No all_genomes/ or runs/ found. Skipping."

    fi

done


echo
echo "============================================================"
echo "All gate/parameter evolution plots completed."
echo "Output root:"
echo "  ${OUTPUT_ROOT}"
echo "============================================================"