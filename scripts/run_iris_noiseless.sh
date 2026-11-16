#!/bin/sh
#
# Iris classification search with NO simulated noise: the baseline that the two
# noisy scripts (run_iris_noise_synthetic.sh, run_iris_noise_chip.sh) are
# compared against. The three scripts share every search setting and differ
# only in the lines marked "noise" below, so a diff between them shows exactly
# what each noise setting adds.
#
#   sh scripts/run_iris_noiseless.sh <min_run> <max_run> <out_dir>
#
# writes one genome archive per run at <out_dir>/iris_i30_noiseless_<run>. The
# run range is inclusive, so an experiment can be extended later with more runs
# without redoing the ones already done.
#
# Environment variables (defaults in brackets):
#   NUMBER_GENOMES  genomes each run evaluates [1000]; use 2 for a quick check
#   N_PROCS         MPI processes [12]: one master plus N_PROCS-1 workers
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
        --out_dir "$OUT_DIR/iris_i30_noiseless_${i}" \
        steady_state \
        --max_population_size 30
    # noise: none. No noise flag is given, so the circuits run on the perfect
    # (state-vector) simulator and every genome records noise_type "none".
done
