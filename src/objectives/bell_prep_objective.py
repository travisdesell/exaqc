"""Bell-state preparation objective with hardware-aware co-objectives.

For a genome on two (or more) wires, evaluates

  task_loss      1 - <Phi+| rho_ideal |Phi+>   (noiseless, default.qubit)
  p_fail         tier-0 failure probability from the device card
                 (gates + idle; src.objectives.hardware_objectives)
  n_entanglers   number of two-qubit gates
  total_time_ns  ASAP-schedule makespan with the card's gate durations
  noisy_fidelity <Phi+| rho_noisy |Phi+> under DeviceCardNoiseModel
                 (tier-1; optional, costs a default.mixed simulation)

and writes them into ``genome.fitness``.  The Bell pair is taken on the
first two output wires; extra wires must be returned to |0> (they are
traced out, so junk on them shows up as lost fidelity only if it is
entangled with the pair -- ``strict_ancilla=True`` additionally penalises
any population outside |0> on the other wires).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.circuits.circuit import CircuitGenome
from src.evolution.objective import Objective
from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics


PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0)


def _bell_fidelity_from_state(state: np.ndarray, n_wires: int, pair=(0, 1),
                              strict_ancilla: bool = False) -> float:
    """state: pure vector (2^n,) or density matrix (2^n, 2^n).  Reduces to
    the pair by partial trace and returns <Phi+|rho|Phi+>."""
    state = np.asarray(state)
    if state.ndim == 1:
        rho = np.outer(state, state.conj())
    else:
        rho = state
    dims = [2] * n_wires
    R = rho.reshape(dims + dims)
    keep = list(pair)
    others = [w for w in range(n_wires) if w not in keep]
    if strict_ancilla and others:
        # population of any non-pair wire outside |0>
        probs = np.real(np.einsum("ii->i", rho.reshape(2 ** n_wires, 2 ** n_wires)))
        leak = 0.0
        for idx, p in enumerate(probs):
            bits = [(idx >> (n_wires - 1 - w)) & 1 for w in range(n_wires)]
            if any(bits[w] for w in others):
                leak += p
    else:
        leak = 0.0
    # Partial trace over the non-pair wires via einsum (wire 0 = most
    # significant bit in PennyLane's ordering, matching reshape order).
    letters = "abcdefghijklmnopqrstuvwxyz"
    row = [letters[w] for w in range(n_wires)]
    col = [letters[w] if w in others else letters[w].upper() for w in range(n_wires)]
    out = [letters[w] for w in keep] + [letters[w].upper() for w in keep]
    R = np.einsum("".join(row + col) + "->" + "".join(out), R)
    n_keep = len(keep)
    rho_pair = R.reshape(2 ** n_keep, 2 ** n_keep)
    F = float(np.real(PHI_PLUS.conj() @ rho_pair @ PHI_PLUS))
    return max(0.0, F - leak)


class BellPrepObjective(Objective):
    def __init__(self, card: DeviceCard, noisy: bool = True,
                 strict_ancilla: bool = True, virtual_z_family: bool = False):
        self.card = card
        self.noisy = noisy
        self.strict_ancilla = strict_ancilla
        self.noise_model = DeviceCardNoiseModel(card, virtual_z_family=virtual_z_family) if noisy else None

    def _state(self, genome: CircuitGenome, noise_model: Optional[DeviceCardNoiseModel]) -> np.ndarray:
        dev, qnode = genome.generate_pennylane_circuit(
            noise_model=noise_model, input_mode="basis", return_probs=False,
            measure_registers=False)
        zeros = torch.zeros(len(genome.input_indexes), dtype=torch.int64)
        out = qnode(zeros, None)
        return out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)

    def __call__(self, genome: CircuitGenome) -> None:
        n = len(genome.qubits)
        ideal = self._state(genome, None)
        F_ideal = _bell_fidelity_from_state(ideal, n, strict_ancilla=self.strict_ancilla)
        m = hardware_metrics(genome, self.card)
        fitness = {
            "task_loss": 1.0 - F_ideal,
            "p_fail": m["p_fail"],
            "n_entanglers": float(m["n_entanglers"]),
            "total_time_ns": m["total_time_ns"],
            "n_gates": float(m["n_gates"]),
        }
        if self.noisy:
            noisy = self._state(genome, self.noise_model)
            fitness["noisy_fidelity"] = _bell_fidelity_from_state(
                noisy, n, strict_ancilla=self.strict_ancilla)
            fitness["noisy_loss"] = 1.0 - fitness["noisy_fidelity"]
        genome.fitness = fitness
