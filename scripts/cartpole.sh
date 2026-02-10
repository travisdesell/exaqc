mpiexec -n 4 python -m src.examples.pl_reinforce --logging_level INFO \
 --env cartpole \
 --number_genomes 500 \
 --input_qubits 4 \
 --out_qubits 2 \
 --episodes 80 \
 --out_dir artifacts/cartpole