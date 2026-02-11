mpiexec -n 4 python -m src.examples.pl_reinforce \
 --logging_level INFO \
 --algo reinforce \
 --env cartpole \
 --number_genomes 500 \
 --input_qubits 4 \
 --output_qubits 2 \
 --episodes 80 \
 --out_dir artifacts/cartpole


python -m src.examples.pl_reinforce \
  --env cartpole \
  --algo q_learning \
  --out_dir artifacts/cartpole_q_learning \
  --logging_level INFO \
  --max_population_size 30 \
  --number_genomes 1200 \
  --input_qubits 4 \
  --out_qubits 2 \
  --episodes 300 \
  --max_steps 500 \
  --gamma 0.99 \
  --q_lr 1e-2 \
  --epsilon 0.20 \
  --epsilon_min 0.02 \
  --epsilon_decay 0.995 \
  --target_update 25 \
  --batch_size 32 \
  --replay_size 10000 \
  --eval_episodes 10 \
  --seed 0




python -m src.examples.pl_reinforce \
  --env cartpole \
  --algo actor_critic \
  --out_dir artifacts/cartpole_actor_critic \
  --logging_level INFO \
  --max_population_size 30 \
  --number_genomes 1200 \
  --input_qubits 4 \
  --out_qubits 2 \
  --episodes 200 \
  --max_steps 500 \
  --gamma 0.99 \
  --actor_lr 3e-3 \
  --critic_lr 1e-2 \
  --entropy_coef 1e-2 \
  --baseline mean \
  --eval_episodes 10 \
  --seed 0
