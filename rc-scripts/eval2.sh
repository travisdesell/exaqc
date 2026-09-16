#!/bin/bash -l
#SBATCH -J exaqc_eval
#SBATCH -t 0-08:00:00
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate


# ============================================================
# Configuration
# ============================================================

DATASETS=(
    "mnist"
    "fashion_mnist"
    "cifar10"
)

BATCH_SIZE=1
ARTIFACT_ROOT="./artifacts"

# shopt -s nullglob


# ============================================================
# Evaluate the highest-numbered genome in a directory
# ============================================================

evaluate_directory() {

    SEARCH_PATH="$1"
    RESULT_FOLDER="$2"
    RUN_LABEL="$3"
    DATASET="$4"
    OUT_ROOT="$5"
    LOG_ROOT="$6"

    MAX_GENOME_ID=-1
    GENOME_PATH=""

    # Find genome_<i>.json with the largest numeric i.
    for genome in "${SEARCH_PATH}"/genome_*.json; do

        [[ -f "$genome" ]] || continue

        filename=$(basename "$genome")

        if [[ "$filename" =~ ^genome_([0-9]+)\.json$ ]]; then

            genome_id="${BASH_REMATCH[1]}"

            if (( genome_id > MAX_GENOME_ID )); then
                MAX_GENOME_ID=$genome_id
                GENOME_PATH="$genome"
            fi

        fi
    done

    if [[ -z "$GENOME_PATH" ]]; then
        echo "No genome_*.json found in ${SEARCH_PATH}. Skipping."
        return
    fi

    MODEL_FILENAME=$(basename "$GENOME_PATH" .json)

    # --------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------

    if [[ -n "$RUN_LABEL" ]]; then

        OUTPUT_FILE="${OUT_ROOT}/${RESULT_FOLDER}_r${RUN_LABEL}_${MODEL_FILENAME}.o"
        ERROR_FILE="${LOG_ROOT}/${RESULT_FOLDER}_r${RUN_LABEL}_${MODEL_FILENAME}.e"

    else

        OUTPUT_FILE="${OUT_ROOT}/${RESULT_FOLDER}_${MODEL_FILENAME}.o"
        ERROR_FILE="${LOG_ROOT}/${RESULT_FOLDER}_${MODEL_FILENAME}.e"

    fi

    echo
    echo "------------------------------------------------------------"
    echo "Dataset    : ${DATASET}"
    echo "Experiment : ${RESULT_FOLDER}"

    if [[ -n "$RUN_LABEL" ]]; then
        echo "Run        : ${RUN_LABEL}"
    else
        echo "Run        : direct experiment directory"
    fi

    echo "Directory  : ${SEARCH_PATH}"
    echo "Genome     : ${GENOME_PATH}"
    echo "Genome ID  : ${MAX_GENOME_ID}"
    echo "Output     : ${OUTPUT_FILE}"
    echo "Error      : ${ERROR_FILE}"
    echo "------------------------------------------------------------"
    echo

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    srun python3.11 -m src.examples.evaluate \
        --dataset "$DATASET" \
        --genome "$GENOME_PATH" \
        --device cuda \
        --batch_size "$BATCH_SIZE" \
        > "$OUTPUT_FILE" \
        2> "$ERROR_FILE"

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "${RESULT_FOLDER} ${RUN_LABEL:+run ${RUN_LABEL} }completed successfully."
    else
        echo "${RESULT_FOLDER} ${RUN_LABEL:+run ${RUN_LABEL} }FAILED with exit code ${EXIT_CODE}."
    fi
}


# ============================================================
# Dataset loop
# ============================================================

for DATASET in "${DATASETS[@]}"; do

    echo
    echo "############################################################"
    echo "# DATASET: ${DATASET}"
    echo "############################################################"
    echo

    OUT_ROOT="./outs/${DATASET}"
    LOG_ROOT="./logs/${DATASET}"

    mkdir -p "$OUT_ROOT"
    mkdir -p "$LOG_ROOT"


    # ========================================================
    # Experiment loop
    # ========================================================

    for RESULT_PATH in "${ARTIFACT_ROOT}"/"${DATASET}"*; do

        [[ -d "$RESULT_PATH" ]] || continue

        RESULT_FOLDER=$(basename "$RESULT_PATH")
        RUNS_PATH="${RESULT_PATH}/runs"

        echo
        echo "============================================================"
        echo "Processing experiment: ${RESULT_FOLDER}"
        echo "============================================================"


        # ----------------------------------------------------
        # Layout 1:
        #
        # experiment/
        # └── runs/
        #     ├── 1/
        #     │   ├── genome_100.json
        #     │   └── genome_500.json
        #     └── 2/
        # ----------------------------------------------------

        if [[ -d "$RUNS_PATH" ]]; then

            echo "Found runs directory: ${RUNS_PATH}"

            FOUND_RUN=false

            for RUN_PATH in "${RUNS_PATH}"/*; do

                [[ -d "$RUN_PATH" ]] || continue

                FOUND_RUN=true

                RUN=$(basename "$RUN_PATH")

                evaluate_directory \
                    "$RUN_PATH" \
                    "$RESULT_FOLDER" \
                    "$RUN" \
                    "$DATASET" \
                    "$OUT_ROOT" \
                    "$LOG_ROOT"

            done

            if [[ "$FOUND_RUN" == false ]]; then
                echo "runs/ exists but contains no run directories."
            fi


        # ----------------------------------------------------
        # Layout 2:
        #
        # experiment/
        # ├── genome_100.json
        # ├── genome_500.json
        # ├── all_genomes/
        # ├── exaqc_history.csv
        # └── ...
        # ----------------------------------------------------

        elif [[ -d "${RESULT_PATH}/all_genomes" ]]; then

            echo "No runs directory found."
            echo "Found all_genomes directly under experiment."

            evaluate_directory \
                "$RESULT_PATH" \
                "$RESULT_FOLDER" \
                "" \
                "$DATASET" \
                "$OUT_ROOT" \
                "$LOG_ROOT"

        else

            echo "No runs directory or direct all_genomes directory found."
            echo "Skipping ${RESULT_FOLDER}."

        fi

    done


    echo
    echo "============================================================"
    echo "Finished dataset: ${DATASET}"
    echo "============================================================"

done


echo
echo "############################################################"
echo "# ALL DATASETS PROCESSED"
echo "############################################################"