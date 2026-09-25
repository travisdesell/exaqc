# activate default quantum env, make venv, activate it, install pennylane

spack env activate default-quantum-aarch64-24112201

mkdir -p ~/venvs

python -m venv --system-site-packages ~/venvs/vqc-tigris
source ~/venvs/vqc-tigris/bin/activate

python -m pip install --upgrade pip
python -m pip install pennylane
