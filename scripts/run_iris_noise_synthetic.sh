#!/bin/sh
#
# Iris classification search under SYNTHETIC noise: one hand-set channel type
# added after every evolved gate, with separate error rates for one-qubit and
# multi-qubit gates. Same search settings as run_iris_noiseless.sh (the
# baseline) and run_iris_noise_chip.sh; only the lines marked "noise" differ.
#
#   sh scripts/run_iris_noise_synthetic.sh <min_run> <max_run> <out_dir>
#
# writes one genome archive per run at
# <out_dir>/iris_i30_<NOISE_TYPE>_p1q<NOISE_P_1Q>_p2q<NOISE_P_2Q>_<run>, so
# runs at different noise strengths never share a directory. The run range is
# inclusive.
#
# Environment variables (defaults in brackets):
#   NUMBER_GENOMES  genomes each run evaluates [1000]; use 2 for a quick check
#   N_PROCS         MPI processes [12]: one master plus N_PROCS-1 workers
#   NOISE_TYPE      the channel [depolarizing]: depolarizing, bit_flip,
#                   phase_flip, amplitude_damping, phase_damping, mixed
#                   (thermal_relaxation needs the T1/T2 flags instead; add
#                   them below if you sweep it)
#   NOISE_P_1Q      error probability after a one-qubit gate [0.001]
#   NOISE_P_2Q      error probability after a multi-qubit gate [0.01]
#
# On the Tigris cluster, run it inside a Slurm job that names the account and
# partition (sbatch ... -A examm-rnn -p tigris), with openmpi loaded and the
# project venv activated, as scripts/exaqc_rl_job.sh does for its own search.

set -eu

if [ $# -ne 3 ]; then
    echo "usage: $0 <min_run> <max_run> <out_dir>" >&2
    exit 2
fi

MIN_COUNT=$1
MAX_COUNT=$2
OUT_DIR=$3

NUMBER_GENOMES=${NUMBER_GENOMES:-1000}
N_PROCS=${N_PROCS:-12}
NOISE_TYPE=${NOISE_TYPE:-depolarizing}
NOISE_P_1Q=${NOISE_P_1Q:-0.001}
NOISE_P_2Q=${NOISE_P_2Q:-0.01}

for i in $(seq "$MIN_COUNT" "$MAX_COUNT"); do
    mpiexec --oversubscribe -n "$N_PROCS" python3 -m src.examples.classification \
        --logging_level INFO \
        --dataset iris \
        --number_genomes "$NUMBER_GENOMES" \
        --input_qubits 4 \
        --output_qubits 2 \
        --batch_size 3 \
        --mutation_strategy uniform 1 3 \
        --parent_strategy uniform 5 5 \
        --binary_crossover_rate 0.1 \
        --n_ary_crossover_rate 0.1 \
        --exponential_crossover_rate 0.1 \
        -qim amplitude \
        -qom probs \
        --encoding identity \
        --decoding clipped \
        --noise_type "$NOISE_TYPE" \
        --noise_p_1q "$NOISE_P_1Q" \
        --noise_p_2q "$NOISE_P_2Q" \
        --out_dir "$OUT_DIR/iris_i30_${NOISE_TYPE}_p1q${NOISE_P_1Q}_p2q${NOISE_P_2Q}_${i}" \
        steady_state \
        --max_population_size 30
    # noise: the three --noise_* lines above. They switch the circuits to the
    # density-matrix simulator and add the channel after every evolved gate;
    # every genome records the type and both rates.
done
