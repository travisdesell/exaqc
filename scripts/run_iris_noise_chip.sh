#!/bin/sh
#
# Iris classification search under CHIP noise: each circuit wire gets the
# calibrated errors of its own physical qubit on a real IBM chip, read from one
# stored calibration file, and every genome is translated onto those qubits so
# gates between unwired qubits are charged their routing. Same search settings
# as run_iris_noiseless.sh (the baseline) and run_iris_noise_synthetic.sh;
# only the lines marked "noise" differ.
#
#   sh scripts/run_iris_noise_chip.sh <min_run> <max_run> <out_dir>
#
# writes one genome archive per run at
# <out_dir>/iris_i30_<CHIP>_<DATE>_q<QUBITS>_<run> (commas in QUBITS become
# dashes), so runs on different chips, days or qubits never share a directory.
# The run range is inclusive.
#
# Environment variables (defaults in brackets):
#   NUMBER_GENOMES  genomes each run evaluates [1000]; use 2 for a quick check
#   N_PROCS         MPI processes [12]: one master plus N_PROCS-1 workers
#   CHIP            the IBM backend [ibm_fez]
#   DATE            the calibration day, YYYY-MM-DD [2026-11-06]
#   QUBITS          the physical qubits wires 0..3 map to, a wired line of
#                   exactly 4 (the larger of --input_qubits and
#                   --output_qubits) [10,11,12,13]
#
# The calibration is named as <CHIP>:<DATE>. If that day's file is stored
# under data/calibration/<CHIP>/ it is used as is; if not, the run fetches it
# once from IBM with the Qiskit account saved on this machine
# (QiskitRuntimeService.save_account) and stores it there, or stops with the
# collect_calibration command to run by hand.
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
CHIP=${CHIP:-ibm_fez}
DATE=${DATE:-2026-11-06}
QUBITS=${QUBITS:-10,11,12,13}

QUBITS_TAG=$(printf '%s' "$QUBITS" | tr ',' '-')

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
        --noise_calibration "${CHIP}:${DATE}" \
        --noise_qubits "$QUBITS" \
        --out_dir "$OUT_DIR/iris_i30_${CHIP}_${DATE}_q${QUBITS_TAG}_${i}" \
        steady_state \
        --max_population_size 30
    # noise: the two --noise_* lines above. They select the calibrated
    # ibm_backend channel; every genome records the file, chip, date, qubits
    # and the per-qubit values it was evolved under.
done
