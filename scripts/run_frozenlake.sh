MIN_COUNT=$1
MAX_COUNT=$2
LOSS=$3
OUT_DIR=$4

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 12 python -m src.examples.pl_reinforce \
        --algo reinforce \
        --logging_level INFO \
        --env frozenlake \
        --number_genomes 1000 \
        --map_name 4x4 \
        --is_slippery \
        --input_qubits 4 \
        --output_qubits 4 \
        --episodes 1000 \
        --eval_episodes 100 \
        --mutation_strategy uniform 1 3 \
        --out_dir $OUT_DIR/frozenlake_i50_$3_${i} \
        steady_state --max_population_size 50
done

