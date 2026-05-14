MIN_COUNT=$1
MAX_COUNT=$2
LOSS=$3
OUT_DIR=$4

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset iris --number_genomes 1000 --input_qubits 4 --batch_size 3 --mutation_strategy uniform 1 3 --loss $LOSS --out_dir $OUT_DIR/iris_i30_$3_${i} steady_state --max_population_size 30
done

