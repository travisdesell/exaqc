#!/bin/bash -l
#SBATCH -J minigrid
#SBATCH -t 2-00:00:00
#SBATCH -o ./outs/minigrid_empty/output.o
#SBATCH -e ./logs/minigrid_empty/error.e
#SBATCH -A cps -p tier3
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a100:1

spack env activate default-ml-x86_64-25052701

source .venv/bin/activate

for i in $(seq 1 1); do
    echo "Starting run ${i}"
    srun python -m src.examples.pl_reinforce \
    --env minigrid \
    --minigrid_env_id MiniGrid-Empty-8x8-v0 \
    --minigrid_obs_wrapper flat \
    --algo ppo \
    --episodes 50 \
    --rollout_steps 512 \
    --ppo_epochs 4 \
    --ppo_minibatch 64 \
    --input_qubits 6 \
    --output_qubits 3 \
    --out_dir artifacts/minigrid_empty/runs/${i}
    echo "Completed run ${i}"
done