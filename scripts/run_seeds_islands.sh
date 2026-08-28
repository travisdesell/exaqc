for i in $(seq $1 $2); do
    mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset seeds --number_genomes 1000 --input_qubits 6 --batch_size 6 --loss per_class --out_dir ~/Data/2026_ppsn_exaqc/seeds_i5_p6_per_class_${i} islands --n_islands 5 --max_island_size 6 --islands_to_extinct 1 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done

for i in $(seq $1 $2); do
    mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset seeds --number_genomes 1000 --input_qubits 6 --batch_size 6 --loss per_class --out_dir ~/Data/2026_ppsn_exaqc/seeds_i10_p3_per_class_${i} islands --n_islands 10 --max_island_size 3 --islands_to_extinct 2 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done

for i in $(seq $1 $2); do
    mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset seeds --number_genomes 1000 --input_qubits 6 --batch_size 6 --loss per_class --out_dir ~/Data/2026_ppsn_exaqc/seeds_i30_p1_per_class_${i} islands --n_islands 30 --max_island_size 1 --islands_to_extinct 3 --genomes_before_extinction 50 --genomes_for_next_extinction 200
done
