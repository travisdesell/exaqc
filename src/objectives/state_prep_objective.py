"""State-preparation objective for an arbitrary target state vector.

Generalises ``GHZPrepObjective`` (which is this with the GHZ target) so the four
Bell states, and any other named state, can be used as tasks.  Objectives
written into ``genome.fitness`` are the same set: task_loss (noiseless, the
feasibility constraint), p_fail / n_entanglers / total_time_ns from the device
card, and the tier-1 fidelity under ``DeviceCardNoiseModel``.

The four Bell states are included by name.  Note that they are not
interchangeable for this study: the textbook circuits that prepare them from
|0...0> differ in single-qubit-gate content, so the idle structure of the
reference circuit differs too, and any conclusion drawn from one of them ought
to be checked on the others.  That is the point of running all four.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from src.circuits.circuit import CircuitGenome
from src.evolution.objective import Objective
from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_R2 = 1.0 / np.sqrt(2.0)


def bell_state(name: str) -> np.ndarray:
    """Wire order is (q0, q1); index = 2*b0 + b1."""
    v = np.zeros(4, dtype=complex)
    if name == "phi_plus":       # (|00> + |11>)/sqrt2
        v[0b00], v[0b11] = _R2, _R2
    elif name == "phi_minus":    # (|00> - |11>)/sqrt2
        v[0b00], v[0b11] = _R2, -_R2
    elif name == "psi_plus":     # (|01> + |10>)/sqrt2
        v[0b01], v[0b10] = _R2, _R2
    elif name == "psi_minus":    # (|01> - |10>)/sqrt2
        v[0b01], v[0b10] = _R2, -_R2
    else:
        raise ValueError(f"unknown Bell state {name!r}")
    return v


def ghz_state(n: int) -> np.ndarray:
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = v[-1] = _R2
    return v


TARGETS: Dict[str, np.ndarray] = {
    "phi_plus": bell_state("phi_plus"),
    "phi_minus": bell_state("phi_minus"),
    "psi_plus": bell_state("psi_plus"),
    "psi_minus": bell_state("psi_minus"),
}
# GHZ_n for every n the benchmark sweeps. ghz2 is phi_plus by definition; it is
# registered under both names so the n-indexed sweep does not special-case it.
TARGETS.update({f"ghz{n}": ghz_state(n) for n in range(2, 13)})
TARGET_WIRES = {"phi_plus": 2, "phi_minus": 2, "psi_plus": 2, "psi_minus": 2}
TARGET_WIRES.update({f"ghz{n}": n for n in range(2, 13)})


def state_fidelity(state: np.ndarray, target: np.ndarray, n_wires: int,
                   target_wires: Optional[Sequence[int]] = None,
                   strict_ancilla: bool = False) -> float:
    """<target| rho_reduced |target> for a pure vector or density matrix."""
    state = np.asarray(state)
    rho = np.outer(state, state.conj()) if state.ndim == 1 else state
    n_t = int(round(np.log2(len(target))))
    keep = list(range(n_t)) if target_wires is None else list(target_wires)
    others = [w for w in range(n_wires) if w not in keep]

    leak = 0.0
    if strict_ancilla and others:
        probs = np.real(np.einsum("ii->i", rho))
        for idx, p in enumerate(probs):
            bits = [(idx >> (n_wires - 1 - w)) & 1 for w in range(n_wires)]
            if any(bits[w] for w in others):
                leak += p

    if others:
        R = rho.reshape([2] * n_wires + [2] * n_wires)
        row = [_LETTERS[w] for w in range(n_wires)]
        col = [_LETTERS[w] if w in others else _LETTERS[w].upper()
               for w in range(n_wires)]
        out = [_LETTERS[w] for w in keep] + [_LETTERS[w].upper() for w in keep]
        R = np.einsum("".join(row + col) + "->" + "".join(out), R)
        rho_keep = R.reshape(2 ** len(keep), 2 ** len(keep))
    else:
        rho_keep = rho

    F = float(np.real(target.conj() @ rho_keep @ target))
    return max(0.0, F - leak)


class StatePrepObjective(Objective):
    def __init__(self, card: DeviceCard, target: str | np.ndarray,
                 noisy: bool = True, strict_ancilla: bool = True,
                 virtual_z_family: bool = False):
        self.card = card
        if isinstance(target, str):
            self.target_name = target
            self.target = TARGETS[target]
        else:
            self.target_name = "custom"
            self.target = np.asarray(target, dtype=complex)
        self.n_target = int(round(np.log2(len(self.target))))
        self.noisy = noisy
        self.strict_ancilla = strict_ancilla
        self.noise_model = (DeviceCardNoiseModel(card, virtual_z_family=virtual_z_family)
                            if noisy else None)

    def _state(self, genome: CircuitGenome, noise_model) -> np.ndarray:
        dev, qnode = genome.generate_pennylane_circuit(
            noise_model=noise_model, input_mode="basis", return_probs=False,
            measure_registers=False)
        zeros = torch.zeros(len(genome.input_indexes), dtype=torch.int64)
        out = qnode(zeros, None)
        return out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)

    def __call__(self, genome: CircuitGenome) -> None:
        n = len(genome.qubits)
        wires = list(range(self.n_target))
        ideal = self._state(genome, None)
        F_ideal = state_fidelity(ideal, self.target, n, wires, self.strict_ancilla)
        m = hardware_metrics(genome, self.card)
        fitness = {
            "task_loss": 1.0 - F_ideal,
            "p_fail": m["p_fail"],
            "n_entanglers": float(m["n_entanglers"]),
            "total_time_ns": m["total_time_ns"],
            "n_gates": float(m["n_gates"]),
            "n_layers": float(m["n_layers"]),
        }
        if self.noisy:
            noisy = self._state(genome, self.noise_model)
            fitness["noisy_fidelity"] = state_fidelity(
                noisy, self.target, n, wires, self.strict_ancilla)
            fitness["noisy_loss"] = 1.0 - fitness["noisy_fidelity"]
        genome.fitness = fitness
