#!/bin/bash -l
#SBATCH -J exaqc_progress_mnist
#SBATCH -t 0-03:00:00
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASET="mnist"
COLUMN_MAX_LENGTH=500

ARTIFACT_ROOT="./artifacts"
OUT_ROOT="./outs/${DATASET}/progress"
LOG_ROOT="./logs/${DATASET}/progress"

mkdir -p "$OUT_ROOT"
mkdir -p "$LOG_ROOT"

# shopt -s nullglob

for RESULT_PATH in "${ARTIFACT_ROOT}"/"${DATASET}"*; do

    [[ -d "$RESULT_PATH" ]] || continue

    RESULT_FOLDER=$(basename "$RESULT_PATH")
    RUNS_PATH="${RESULT_PATH}/runs"

    echo "============================================================"
    echo "Processing experiment: ${RESULT_FOLDER}"
    echo "============================================================"

    INPUT_DIRS=()

    # ---------------------------------------------------------
    # Layout 1: experiment/runs/<run>/
    # ---------------------------------------------------------
    if [[ -d "$RUNS_PATH" ]]; then

        echo "Found runs directory: ${RUNS_PATH}"

        for RUN_PATH in "${RUNS_PATH}"/*; do

            [[ -d "$RUN_PATH" ]] || continue

            RUN=$(basename "$RUN_PATH")

            if [[ -d "${RUN_PATH}/all_genomes" ]] && \
               [[ -f "${RUN_PATH}/exaqc_history.csv" ]]; then

                INPUT_DIRS+=("$RUN_PATH")
                echo "Adding run ${RUN}: ${RUN_PATH}"

            else
                echo "Skipping run ${RUN}: missing all_genomes and/or exaqc_history.csv"
            fi

        done

    # ---------------------------------------------------------
    # Layout 2: experiment/all_genomes/
    # ---------------------------------------------------------
    elif [[ -d "${RESULT_PATH}/all_genomes" ]]; then

        echo "No runs directory found."
        echo "Found all_genomes directly under experiment."

        if [[ -f "${RESULT_PATH}/exaqc_history.csv" ]]; then

            INPUT_DIRS+=("$RESULT_PATH")
            echo "Adding: ${RESULT_PATH}"

        else

            echo "Missing ${RESULT_PATH}/exaqc_history.csv"
            echo "Skipping ${RESULT_FOLDER}."
            continue

        fi

    else

        echo "No runs/*/all_genomes or all_genomes found."
        echo "Skipping ${RESULT_FOLDER}."
        continue

    fi

    if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
        echo "No valid input directories found for ${RESULT_FOLDER}. Skipping."
        continue
    fi

    OUTPUT_FILE="${OUT_ROOT}/${RESULT_FOLDER}_progress.o"
    ERROR_FILE="${LOG_ROOT}/${RESULT_FOLDER}_progress.e"

    echo
    echo "Experiment        : ${RESULT_FOLDER}"
    echo "Inputs            : ${#INPUT_DIRS[@]}"
    echo "Column max length : ${COLUMN_MAX_LENGTH}"
    echo "Output            : ${OUTPUT_FILE}"
    echo "Error             : ${ERROR_FILE}"

    echo "Input directories:"
    printf '  %s\n' "${INPUT_DIRS[@]}"
    echo

    srun python3.11 -m src.analysis.plot_search_progress \
        --input_directories "${INPUT_DIRS[@]}" \
        --column_max_length "$COLUMN_MAX_LENGTH" \
        --output_filename "search_progress.png" \
        > "$OUTPUT_FILE" \
        2> "$ERROR_FILE"

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "${RESULT_FOLDER} progress analysis completed successfully."
    else
        echo "${RESULT_FOLDER} progress analysis FAILED with exit code ${EXIT_CODE}."
    fi

done

echo
echo "All ${DATASET} experiment folders processed."