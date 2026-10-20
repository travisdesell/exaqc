"""GHZ_n state-preparation objective with hardware-aware co-objectives.

Generalises ``BellPrepObjective`` (GHZ_2 = |Phi+>) to

    |GHZ_n> = (|0...0> + |1...1>) / sqrt(2)

on the first ``n_target`` wires.  Objectives written into ``genome.fitness``:

  task_loss      1 - <GHZ_n| rho_target |GHZ_n>, noiseless (default.qubit),
                 with the non-target wires traced out; ``strict_ancilla``
                 additionally subtracts any population they hold outside |0>
                 (so the search cannot park garbage on a helper wire).
  p_fail         tier-0 failure probability from the device card
  n_entanglers   number of two-qubit gates
  total_time_ns  ASAP-schedule makespan using the card's gate durations
  noisy_fidelity <GHZ_n| rho_noisy |GHZ_n> under DeviceCardNoiseModel (tier 1)

GHZ_n is the right next task after the Bell pair because the entangler count
is genuinely free: the linear chain CNOT(0,1), CNOT(1,2), ... uses n-1
entanglers in n-1 layers, while the balanced-tree ordering uses the same
n-1 entanglers in ceil(log2 n) layers.  Depth and entangler count therefore
decouple, and which one the optimum prefers depends on the card -- an
architecture-dependent trade-off rather than a single feasible point.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from src.circuits.circuit import CircuitGenome
from src.evolution.objective import Objective
from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def ghz_vector(n: int) -> np.ndarray:
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = v[-1] = 1.0 / np.sqrt(2.0)
    return v


def ghz_fidelity_from_state(state: np.ndarray, n_wires: int,
                            target_wires: Optional[Sequence[int]] = None,
                            strict_ancilla: bool = False) -> float:
    """<GHZ_k| rho_target |GHZ_k> for a pure vector or density matrix on
    ``n_wires`` wires, reduced to ``target_wires`` (default: all of them)."""
    state = np.asarray(state)
    rho = np.outer(state, state.conj()) if state.ndim == 1 else state
    keep = list(range(n_wires)) if target_wires is None else list(target_wires)
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

    g = ghz_vector(len(keep))
    F = float(np.real(g.conj() @ rho_keep @ g))
    return max(0.0, F - leak)


class GHZPrepObjective(Objective):
    """``n_target = 2`` reproduces BellPrepObjective's numbers exactly."""

    def __init__(self, card: DeviceCard, n_target: int, noisy: bool = True,
                 strict_ancilla: bool = True, virtual_z_family: bool = False):
        self.card = card
        self.n_target = int(n_target)
        self.noisy = noisy
        self.strict_ancilla = strict_ancilla
        self.noise_model = (DeviceCardNoiseModel(card, virtual_z_family=virtual_z_family)
                            if noisy else None)

    def _state(self, genome: CircuitGenome,
               noise_model: Optional[DeviceCardNoiseModel]) -> np.ndarray:
        dev, qnode = genome.generate_pennylane_circuit(
            noise_model=noise_model, input_mode="basis", return_probs=False,
            measure_registers=False)
        zeros = torch.zeros(len(genome.input_indexes), dtype=torch.int64)
        out = qnode(zeros, None)
        return out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)

    def __call__(self, genome: CircuitGenome) -> None:
        n = len(genome.qubits)
        target = list(range(self.n_target))
        ideal = self._state(genome, None)
        F_ideal = ghz_fidelity_from_state(ideal, n, target, self.strict_ancilla)
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
            fitness["noisy_fidelity"] = ghz_fidelity_from_state(
                noisy, n, target, self.strict_ancilla)
            fitness["noisy_loss"] = 1.0 - fitness["noisy_fidelity"]
        genome.fitness = fitness
