#source ~/Code/exaqc/bin/activate
for i in $(seq $1 $2); do
    mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 3000 --input_qubits 8 --batch_size 3 --loss $3 --out_dir ~/Data/2026_gptp_exaqc/breast_i30_$3_${i} steady_state --max_population_size 30
done
