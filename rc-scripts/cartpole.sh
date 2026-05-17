#!/bin/bash -l
#SBATCH -J cartpole
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/cartpole/runs/output.o
#SBATCH -e ./logs/cartpole/runs/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

for i in $(seq 1 10); do
  echo "Starting run ${i}"
  srun python3 -m src.examples.pl_reinforce \
  --algo reinforce \
  --logging_level INFO \
  --env cartpole \
  --number_genomes 1000 \
  --input_qubits 4 \
  --output_qubits 2 \
  --episodes 100 \
  --mutation_strategy uniform 1 3 \
  --out_dir artifacts/cartpole_reinforce/islands/runs/${i} \
  islands \
  --n_islands 5 \
  --max_island_size 6 \
  --islands_to_extinct 1 \
  --genomes_before_extinction 50 \
  --genomes_for_next_extinction 200
done

# mpiexec -n 4 python -m src.examples.pl_reinforce \
#   --env cartpole \
#   --algo q_learning \
#   --out_dir artifacts/cartpole_q_learning \
#   --logging_level INFO \
#   --max_population_size 30 \
#   --number_genomes 1200 \
#   --input_qubits 4 \
#   --out_qubits 2 \
#   --episodes 300 \
#   --max_steps 500 \
#   --gamma 0.99 \
#   --lr 1e-2 \
#   --epsilon 0.20 \
#   --epsilon_min 0.02 \
#   --epsilon_decay 0.995 \
#   --target_update 25 \
#   --batch_size 32 \
#   --replay_size 10000 \
#   --eval_episodes 10 \
#   --seed 0




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
