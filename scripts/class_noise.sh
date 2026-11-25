python -m src.examples.pl_classification_noisy \
  --dataset iris \
  --loss ce \
  --epochs 30 \
  --learning_rate 5e-4 \
  --number_genomes 500 \
  --input_qubits 4 \
  --batch_size 16 \
  --encoding angle \
  --noise_type depolarizing \
  --noise_p_1q 0.001 \
  --noise_p_2q 0.01 \
  --noise_after_gates \
  --out_dir artifacts/iris_noisy \
  steady_state \
  --max_population_size 30