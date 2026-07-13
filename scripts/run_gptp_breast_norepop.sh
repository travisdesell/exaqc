MIN_COUNT=$1
MAX_COUNT=$2
LOSS=$3
OUT_DIR=$4

N_ISLANDS=64
ISLAND_SIZE=1
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 2000 --input_qubits 8 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/breast_cancer_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done

N_ISLANDS=32
ISLAND_SIZE=2
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 2000 --input_qubits 8 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/breast_cancer_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done


N_ISLANDS=16
ISLAND_SIZE=4
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 2000 --input_qubits 8 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/breast_cancer_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done


N_ISLANDS=8
ISLAND_SIZE=8
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 2000 --input_qubits 8 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/breast_cancer_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done


N_ISLANDS=4
ISLAND_SIZE=16
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 2000 --input_qubits 8 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/breast_cancer_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done


N_ISLANDS=2
ISLAND_SIZE=32
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 2000 --input_qubits 8 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/breast_cancer_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done


N_ISLANDS=1
ISLAND_SIZE=64
for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 24 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 2000 --input_qubits 8 --batch_size 3 --encoding amplitude --mutation_strategy exponential 0.5 --loss $LOSS --out_dir $OUT_DIR/breast_cancer_i${N_ISLANDS}_p${ISLAND_SIZE}_$3_${i} islands --n_islands ${N_ISLANDS} --max_island_size ${ISLAND_SIZE} --islands_to_extinct 0 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done
