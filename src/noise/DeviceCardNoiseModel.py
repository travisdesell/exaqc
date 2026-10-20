from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml

from src.noise.PennyLaneNoiseModel import PennyLaneNoiseModel
from src.objectives.hardware_objectives import DeviceCard, asap_schedule

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label: str) -> np.ndarray:
    M = np.array([[1.0 + 0j]])
    for c in label:
        M = np.kron(M, _PAULI[c])
    return M


def _kraus_from_pauli_probs(probs: Dict[str, float]) -> List[np.ndarray]:
    """Kraus operators sqrt(p_P) P of a Pauli channel.  Small negative
    twirl probabilities (non-Pauli-dominated channels) are clipped and the
    distribution renormalised so the channel is CPTP."""
    labels = list(probs)
    p = np.array([max(float(probs[l]), 0.0) for l in labels])
    s = p.sum()
    if s <= 0.0:
        return [_pauli_matrix("I" * len(labels[0]))]
    p = p / s
    return [np.sqrt(pi) * _pauli_matrix(l) for l, pi in zip(labels, p) if pi > 0.0]


class DeviceCardNoiseModel(PennyLaneNoiseModel):
    """Per-gate noise from an optimal-control *device card*.

    Instead of one depolarising / thermal-relaxation strength for all
    single-qubit gates and one for all two-qubit gates (the IBM path),
    every gate in the card carries its own Pauli-twirled error channel,
    obtained by optimising the gate's control pulse on a modelled
    architecture and Pauli-twirling the resulting error map.  Wires that
    idle while a neighbour is being driven receive the card's idle
    channel for the idle duration (ASAP schedule; see
    ``src.objectives.hardware_objectives``).

    The same evolved circuit can be evaluated under several cards (one
    per open-system model, or one per architecture) without touching the
    genome, which is the point of the architecture-aware comparison.

    Usage::

        nm = DeviceCardNoiseModel.from_json("device_card_transmon_mixed.json")
        dev, qnode = genome.generate_pennylane_circuit(noise_model=nm, ...)
    """

    def __init__(self, card: DeviceCard, *, apply_idle: bool = True,
                 virtual_z_family: bool = False, **kwargs):
        kwargs.setdefault("noise_type", "device_card")
        kwargs.setdefault("apply_after_gates", True)
        super().__init__(**kwargs)
        self.card = card
        self.apply_idle = apply_idle
        # Treat Z/S/T (and adjoints) as virtual frame updates: no error,
        # no duration.  The card's entries are for physically driven
        # phase gates; hardware usually does them virtually.
        self.virtual_z_family = virtual_z_family
        self._kraus_cache: Dict[str, List[np.ndarray]] = {}
        self._idle_kraus_cache: Dict[float, List[np.ndarray]] = {}
        self._schedule = None
        self._genome = None
        # Wall-clock time at which each wire last became free, used to charge
        # the idle interval that precedes each gate.
        self._free_at: Dict[int, float] = {}

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "DeviceCardNoiseModel":
        card = DeviceCard.load(path)
        if kwargs.pop("virtual_z_family", False):
            card = _with_virtual_z(card)
            kwargs["virtual_z_family"] = True
        return cls(card, **kwargs)

    def backend_name(self) -> str:
        return "pennylane"

    def device_name(self) -> str:
        return "default.mixed"

    def is_noisy(self) -> bool:
        return True

    # Hooks -----------------------------------------------------------------
    def begin_circuit(self, genome) -> None:
        self._genome = genome
        self._schedule = asap_schedule(genome, self.card) if self.apply_idle else None
        self._free_at = {w: 0.0 for w in range(len(genome.qubits))}

    def before_gate(self, gate_wires: Sequence[int], gate=None) -> None:
        """Charge each of this gate's wires for the interval it spent idle
        since it was last busy. Applying the channel here rather than at the
        end of the circuit is what makes the idle model correct: the state the
        channel acts on is the state the wire actually held while waiting."""
        if (self._schedule is None or gate is None
                or not getattr(gate, "enabled", True)):
            return
        wires = self._as_wires(gate_wires)
        if not wires:
            return
        start = self._schedule.start_ns.get(gate.innovation_number)
        if start is None:
            return
        for w in wires:
            gap = start - self._free_at.get(w, 0.0)
            if gap > 1e-9:
                kraus = self._idle_kraus(gap)
                if kraus is not None and len(kraus) > 1:
                    qml.QubitChannel(kraus, wires=[w])
            self._free_at[w] = start + self.card.duration(gate.method_name)

    def after_gate(self, gate_wires: Sequence[int], gate=None) -> None:
        if not self.apply_after_gates or gate is None or not getattr(gate, "enabled", True):
            return
        wires = self._as_wires(gate_wires)
        if not wires:            # disabled / empty gate: nothing was applied
            return
        kraus = self._kraus_for(gate.method_name)
        if kraus is not None and len(kraus) > 1:
            qml.QubitChannel(kraus, wires=wires)

    def before_measurement(self, measurement_wires: Sequence[int]) -> None:
        """Terminal idle only: the interval between a wire becoming free for
        the last time and the end of the circuit. Everything earlier was
        charged by before_gate at the point it occurred."""
        if self._schedule is None:
            return
        end = self._schedule.makespan_ns
        for w in sorted(self._free_at):
            gap = end - self._free_at.get(w, 0.0)
            if gap <= 1e-9:
                continue
            kraus = self._idle_kraus(gap)
            if kraus is not None and len(kraus) > 1:
                qml.QubitChannel(kraus, wires=[w])

    # Channels --------------------------------------------------------------
    def _kraus_for(self, method_name: str) -> Optional[List[np.ndarray]]:
        if method_name not in self._kraus_cache:
            if not self.card.has(method_name):
                raise KeyError(
                    f"device card has no channel for gate {method_name!r}; "
                    f"restrict the evolvable gate set to {sorted(self.card.gates)}")
            probs = self.card.entry(method_name)["pauli_error"]
            self._kraus_cache[method_name] = _kraus_from_pauli_probs(probs)
        return self._kraus_cache[method_name]

    def _idle_kraus(self, idle_ns: float) -> Optional[List[np.ndarray]]:
        key = round(float(idle_ns), 6)
        if key in self._idle_kraus_cache:
            return self._idle_kraus_cache[key]
        diag = self.card.idle_ptm_diagonal(idle_ns)
        if diag is None:
            return None
        # Invert the one-qubit twirl: p_P = (1/4) sum_Q chi(P,Q) R_QQ.
        labels = ["I", "X", "Y", "Z"]
        chi = np.array([[1, 1, 1, 1], [1, 1, -1, -1],
                        [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float)
        R = np.array([diag[l] for l in labels])
        p = chi @ R / 4.0
        out = _kraus_from_pauli_probs(dict(zip(labels, p)))
        self._idle_kraus_cache[key] = out
        return out

    def save_noise_summary(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"noise_type=device_card\n")
            f.write(f"architecture={self.card.architecture}\n")
            f.write(f"model={self.card.model}\n")
            f.write(f"virtual_z_family={self.virtual_z_family}\n")
            for g, e in sorted(self.card.gates.items()):
                f.write(f"{g}: T={e['T_ns']} ns  1-F={e['infidelity']:.3e}  "
                        f"p_err={1 - e['pauli_error']['I' * int(e.get('wires', 1))]:.3e}\n")


def _with_virtual_z(card: DeviceCard) -> DeviceCard:
    """Copy of the card in which z/s/t/sdg/tdg are error-free and instant."""
    import copy
    c = copy.deepcopy(card)
    for g in ("z", "s", "t", "sdg", "tdg"):
        if g in c.gates:
            e = c.gates[g]
            n = int(e.get("wires", 1))
            e["T_ns"] = 0.0
            e["infidelity"] = 0.0
            e["pauli_error"] = {k: (1.0 if set(k) == {"I"} else 0.0)
                                for k in e["pauli_error"]}
            e["virtual"] = True
    return c
