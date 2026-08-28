MIN_COUNT=$1
MAX_COUNT=$2
LOSS=$3
OUT_DIR=$4

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 12 python -m src.examples.pl_reinforce \
        --algo reinforce \
        --logging_level INFO \
        --env walker2d \
        --learning_rate 3e-4 \
        --rollout_steps 2048 \
        --max_steps 1000 \
        --input_qubits 6 \
        --output_qubits 6 \
        --episodes 100 \
        --mutation_strategy uniform 1 3 \
        --out_dir $OUT_DIR/walker2d_i50_$3_${i} \
        steady_state --max_population_size 50
done

