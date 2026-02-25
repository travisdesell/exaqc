mpiexec -n 4 python -m src.examples.pl_reinforce \
 --algo reinforce \
 --logging_level INFO \
 --env frozenlake \
 --map_name 4x4 \
 --is_slippery \
 --input_qubits 4 \
 --output_qubits 4 \
 --episodes 300 \
 --out_dir artifacts/frozenlake
