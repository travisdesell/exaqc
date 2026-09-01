MIN_COUNT=$1
MAX_COUNT=$2
LOSS=$3
OUT_DIR=$4

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 12 python3 -m src.examples.classification --logging_level INFO --dataset breast_cancer --number_genomes 1000 --input_qubits 8 --output_qubits 1 --batch_size 3 --mutation_strategy uniform 1 3 --parent_strategy uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir $OUT_DIR/breast_i30_$3_${i} steady_state --max_population_size 30
done
