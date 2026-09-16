#!/bin/bash -l
#SBATCH -J exaqc_analysis
#SBATCH -t 0-08:00:00
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

DATASETS=("mnist" "fashion_mnist" "cifar10")
METRIC="target_metric"

ARTIFACT_ROOT="./artifacts"

# shopt -s nullglob


for DATASET in "${DATASETS[@]}"; do

    OUT_ROOT="./outs/${DATASET}/analysis"
    LOG_ROOT="./logs/${DATASET}/analysis"

    mkdir -p "$OUT_ROOT"
    mkdir -p "$LOG_ROOT"
    
    for RESULT_PATH in "${ARTIFACT_ROOT}"/"${DATASET}"*; do

        [[ -d "$RESULT_PATH" ]] || continue

        RESULT_FOLDER=$(basename "$RESULT_PATH")
        RUNS_PATH="${RESULT_PATH}/runs"

        echo "============================================================"
        echo "Processing experiment: ${RESULT_FOLDER}"
        echo "============================================================"

        INPUT_DIRS=()

        # ---------------------------------------------------------
        # Layout 1: experiment/runs/<run>/all_genomes/
        # ---------------------------------------------------------
        if [[ -d "$RUNS_PATH" ]]; then

            echo "Found runs directory: ${RUNS_PATH}"

            for RUN_PATH in "${RUNS_PATH}"/*; do

                [[ -d "$RUN_PATH" ]] || continue

                RUN=$(basename "$RUN_PATH")

                if [[ -d "${RUN_PATH}/all_genomes" ]]; then
                    INPUT_DIRS+=("$RUN_PATH")
                    echo "Adding run ${RUN}: ${RUN_PATH}"
                else
                    echo "Skipping run ${RUN}: no all_genomes directory"
                fi

            done

        # ---------------------------------------------------------
        # Layout 2: experiment/all_genomes/
        # ---------------------------------------------------------
        elif [[ -d "${RESULT_PATH}/all_genomes" ]]; then

            echo "No runs directory found."
            echo "Found all_genomes directly under experiment."
            echo "Adding: ${RESULT_PATH}"

            INPUT_DIRS+=("$RESULT_PATH")

        else

            echo "No runs/*/all_genomes or all_genomes found."
            echo "Skipping ${RESULT_FOLDER}."
            continue

        fi

        if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
            echo "No valid input directories found for ${RESULT_FOLDER}. Skipping."
            continue
        fi

        OUTPUT_FILE="${OUT_ROOT}/${RESULT_FOLDER}_${METRIC}.o"
        ERROR_FILE="${LOG_ROOT}/${RESULT_FOLDER}_${METRIC}.e"

        echo
        echo "Experiment : ${RESULT_FOLDER}"
        echo "Metric     : ${METRIC}"
        echo "Inputs     : ${#INPUT_DIRS[@]}"
        echo "Output     : ${OUTPUT_FILE}"
        echo "Error      : ${ERROR_FILE}"

        echo "Input directories:"
        printf '  %s\n' "${INPUT_DIRS[@]}"
        echo

        srun python3.11 -m src.analysis.analyze_genome_generation \
            --input_directories "${INPUT_DIRS[@]}" \
            --metric "$METRIC" \
            > "$OUTPUT_FILE" \
            2> "$ERROR_FILE"

        EXIT_CODE=$?

        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "${RESULT_FOLDER} analysis completed successfully."
        else
            echo "${RESULT_FOLDER} analysis FAILED with exit code ${EXIT_CODE}."
        fi

    done
done

echo
echo "All ${DATASET} experiment folders processed."