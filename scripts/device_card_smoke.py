"""Smoke test for DeviceCardNoiseModel + hardware_objectives on a hand-built
Bell-preparation genome  CNOT . (H (x) I).

Runs the same genome under every device card found in --cards, printing the
noisy |00>,|01>,|10>,|11> probabilities, the tier-0 p_fail estimate and the
schedule.  Cards lacking a `cx` entry (the CNOT campaign has not run yet) are
patched in-memory with a clearly-labelled placeholder so the plumbing can be
exercised end-to-end; the printout says so.

    .venv_win/Scripts/python scripts/device_card_smoke.py \
        --cards ../../Arch_Aware_Circuit_Evolution/device_cards/*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.circuits.registers import expand_registers  # noqa: E402
from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics, asap_schedule  # noqa: E402


PLACEHOLDER_CX = {
    "T_ns": 200.0, "infidelity": 5e-3, "leakage": 0.0, "process_infidelity": 6e-3,
    "trace_loss": 0.0, "wires": 2, "PLACEHOLDER": "no CNOT campaign yet",
    "pauli_error": {"II": 0.994, "IX": 0.0005, "IY": 0.0005, "IZ": 0.0005,
                    "XI": 0.0005, "XX": 0.0005, "XY": 0.0005, "XZ": 0.0005,
                    "YI": 0.0005, "YX": 0.0005, "YY": 0.0005, "YZ": 0.0005,
                    "ZI": 0.0005, "ZX": 0.0005, "ZY": 0.0005, "ZZ": 0.0005},
    "ptm_error_diag": {},
}


def bell_genome() -> CircuitGenome:
    inp = expand_registers({"q": 2})
    g = CircuitGenome(genome_number=0, target="pennylane",
                      input_qubits=inp.copy(), output_qubits=inp.copy())
    g.add_gate(depth=0.2, method_name="h", qubits=[("q", 0)])
    g.add_gate(depth=0.6, method_name="cx", qubits=[("q", 0), ("q", 1)])
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", nargs="+", required=True)
    args = ap.parse_args()
    paths = [p for pat in args.cards for p in glob.glob(pat)]
    genome = bell_genome()
    print("Bell genome:", [(g.method_name, g.qubits) for g in genome.gates])
    for path in sorted(paths):
        card = DeviceCard.load(path)
        placeholder = False
        if not card.has("cx"):
            card.gates["cx"] = dict(PLACEHOLDER_CX)
            placeholder = True
        nm = DeviceCardNoiseModel(card)
        dev, qnode = genome.generate_pennylane_circuit(
            noise_model=nm, return_probs=True, input_mode="basis")
        zeros = np.zeros(len(genome.input_indexes))
        probs = np.asarray(qnode(zeros, None))
        m = hardware_metrics(genome, card)
        sched = asap_schedule(genome, card)
        tag = " [cx PLACEHOLDER]" if placeholder else ""
        print(f"\n== {card.architecture} / {card.model}{tag}")
        print(f"   probs |00>,|01>,|10>,|11> = {np.round(probs, 6)}   "
              f"(ideal 0.5, 0, 0, 0.5)   1 - P(Bell) = {1 - probs[0] - probs[3]:.3e}")
        print(f"   p_fail={m['p_fail']:.3e}  n_entanglers={m['n_entanglers']}  "
              f"total_time={m['total_time_ns']:.0f} ns  layers={m['n_layers']}  "
              f"shots/success={m['expected_shots_per_success']:.4f}")
        print(f"   idle per wire: {sched.idle_ns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
