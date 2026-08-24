MIN_COUNT=$1
MAX_COUNT=$2
OUT_DIR=$3

arguments=(
    --oversubscribe 
    -n 12 
    python3 -m src.examples.pl_classification
    --logging_level INFO
    --dataset seeds
    --number_genomes 1000
    --input_qubits 6
    --batch_size 3
    --mutation_strategy uniform 2 3
    --output_qubits 3
    --parent_strategy uniform 2 3
)

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec "${arguments[@]}" --out_dir $OUT_DIR/seeds_i30_$3_${i} steady_state --max_population_size 30
done

