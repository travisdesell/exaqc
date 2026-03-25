# for i in $(seq 1 10); do
#   mpiexec -n 4 python -m src.examples.pl_reinforce \
#   --logging_level INFO \
#   --algo reinforce \
#   --env cartpole \
#   --number_genomes 500 \
#   --input_qubits 4 \
#   --output_qubits 2 \
#   --episodes 80 \
#   --out_dir artifacts/cartpole_mc/runs/${i}
# done

for i in $(seq 1 10); do
  mpiexec -n 4 python -m src.examples.pl_reinforce \
    --env cartpole \
    --algo q_learning \
    --out_dir artifacts/cartpole_q_learning/runs/${i} \
    --logging_level INFO \
    --number_genomes 1000 \
    --input_qubits 4 \
    --output_qubits 2 \
    --episodes 300 \
    --max_steps 500 \
    --gamma 0.99 \
    --lr 1e-2 \
    --epsilon 0.20 \
    --epsilon_min 0.02 \
    --epsilon_decay 0.995 \
    --eval_episodes 10 \
    --seed 0 \
    steady_state --max_population_size 30
done




# mpiexec -n 4 python -m src.examples.pl_reinforce \
#   --env cartpole \
#   --algo a2c \
#   --out_dir artifacts/cartpole_actor_critic \
#   --logging_level INFO \
#   --max_population_size 30 \
#   --number_genomes 1200 \
#   --input_qubits 4 \
#   --output_qubits 2 \
#   --episodes 200 \
#   --max_steps 500 \
#   --gamma 0.99 \
#   --lr 3e-3 \
#   --entropy_coef 1e-2 \
#   --baseline mean \
#   --eval_episodes 10 \
#   --seed 0
