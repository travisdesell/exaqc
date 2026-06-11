MIN_COUNT=$1
MAX_COUNT=$2
LOSS=$3
OUT_DIR=$4

N_ISLANDS=1
ISLAND_SIZE=64
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset wine --number_genomes 2000 --input_qubits 4 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/wine_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done
