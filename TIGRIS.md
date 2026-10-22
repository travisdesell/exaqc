# download and install uv:

curl -LsSf https://astral.sh/uv/install.sh | sh

# install python 3.12 with UV

uv python install 3.12

# load MPI version:

spack load openmpi /ttqroyz

# create venv:

cd exaqc

python3.12 -m venv ~/envs/exaqc

source ~/envs/exaqc/bin/activate

# install mpi4py linked to the loaded openmpi:

python3 -m pip install --no-cache-dir --no-binary=mpi4py mpi4py

# install the rest of the dependencies

python3 -m pip install -e .

# fix munge/PMIX error, make sure this is in the submission scripts too

export PMIX_MCA_psec="^munge"
