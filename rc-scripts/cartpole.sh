#!/bin/bash -l
#SBATCH -J cartpole_rf
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/cartpole_rf/pop/runs/output.o
#SBATCH -e ./logs/cartpole_rf/pop/runs/error.e
#SBATCH -A cps -p tier3
#SBATCH --ntasks=12
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=12 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=16GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

for i in $(seq 1 10); do
  echo "Starting run ${i}"
  srun python3 -m src.examples.pl_reinforce \
  --logging_level INFO \
  --algo reinforce \
  --env cartpole \
  --number_genomes 1000 \
  --input_qubits 4 \
  --output_qubits 2 \
  --episodes 100 \
  --out_dir artifacts/cartpole_rf/pop/runs/${i} \
  steady_state --max_population_size 50
  echo "Completed run ${i}"
done

# for i in $(seq 1 10); do
#   srun python3 -m src.examples.pl_reinforce \
#     --env cartpole \
#     --algo q_learning \
#     --out_dir artifacts/cartpole_q_learning/pop/runs/${i} \
#     --logging_level INFO \
#     --number_genomes 1000 \
#     --input_qubits 4 \
#     --output_qubits 2 \
#     --episodes 300 \
#     --max_steps 500 \
#     --gamma 0.99 \
#     --lr 1e-2 \
#     --epsilon 0.20 \
#     --epsilon_min 0.02 \
#     --epsilon_decay 0.995 \
#     --eval_episodes 30 \
#     --seed 0 \
#     steady_state --max_population_size 50
# done

# islands --n_islands 10 --max_island_size 3


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



# for i in $(seq 1 10); do
#   srun python -m src.examples.pl_reinforce \
#     --env cartpole \
#     --algo ppo \
#     --out_dir artifacts/cartpole_ppo/pop/runs/${i} \
#     --number_genomes 1000 \
#     --input_qubits 4 \
#     --output_qubits 2 \
#     --episodes 100 \
#     --eval_episodes 50 \
#     --max_steps 500 \
#     --gamma 0.99 \
#     --learning_rate 0.01 \
#     --entropy_coef 0.01 \
#     --seed 0 \
#     --log_every 10 \
#     --logging_level INFO \
#     steady_state --max_population_size 30
# done


