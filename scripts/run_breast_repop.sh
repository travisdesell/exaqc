for i in $(seq $1 $2); do
    mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 1000 --input_qubits 8 --batch_size 6 --loss $3 --out_dir ./breast_islands_repop_$3_${i} islands --n_islands 50 --max_island_size 1 --islands_to_extinct 3 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done
