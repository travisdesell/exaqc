#!/bin/bash -l

#SBATCH --job-name=vqc-layers
#SBATCH --account=YOUR_RIT_RC_ACCOUNT
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=0-01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node list:   ${SLURM_JOB_NODELIST}"
echo "MPI tasks:   ${SLURM_NTASKS}"
echo "Working dir: $(pwd)"

# Avoid oversubscribing each MPI process with extra CPU threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# RIT's MPI examples load OpenMPI before launching with srun.
spack load openmpi

# Build this environment on TIGRIS itself (ARM/aarch64).
source ~/venvs/vqc-tigris/bin/activate

python - <<'PY'
import sys
import pennylane
import mpi4py
print("Python:", sys.version)
print("PennyLane:", pennylane.__version__)
print("mpi4py:", mpi4py.__version__)
PY

srun --cpu-bind=cores \
    python -u distributed_layered_ground_state.py \
    --problem h2 \
    --max-layers 10 \
    --steps 300 \
    --stepsize 0.1 \
    --seed 0 \
    --output "layered_vqc_${SLURM_JOB_ID}.csv"
