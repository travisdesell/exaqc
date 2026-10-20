"""Hardware-aware, simulation-free objectives computed from a device card.

A *device card* is a JSON file produced by the optimal-control side
(``Non-Markov/code/python/make_device_card.py``): for every gate in a
universal set it records the pulse duration ``T_ns`` and the Pauli-twirled
error probabilities of the gate's error channel under one open-system
model, plus the single-qubit idle channel at each gate duration.  This
module turns a ``CircuitGenome`` plus a card into three scalar objectives:

``p_fail``
    Probability that at least one Pauli error occurs anywhere in the
    circuit, gates and idle periods included (independent-error, tier-0
    estimate; the PennyLane simulation with ``DeviceCardNoiseModel`` is
    the tier-1 version).  ``1 / (1 - p_fail)`` is the expected number of
    shots per error-free execution.
``n_entanglers``
    Number of two-qubit gates (the native entangler count).
``total_time_ns``
    Makespan of an as-soon-as-possible schedule in which each gate occupies
    its wires for ``T_ns``; this is "depth" in physical units.

Nothing here touches PennyLane, so it is cheap enough to evaluate inside
the evolutionary loop for every genome.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger


@dataclass
class DeviceCard:
    """In-memory view of a device-card JSON."""

    architecture: str
    model: str
    gates: Dict[str, dict]
    idle: Dict[float, dict]
    raw: dict = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "DeviceCard":
        raw = json.load(open(path))
        idle = {float(k): v for k, v in raw.get("idle", {}).items()}
        return cls(
            architecture=raw.get("architecture", "?"),
            model=raw.get("model", "?"),
            gates=raw["gates"],
            idle=idle,
            raw=raw,
        )

    # Gate lookups ----------------------------------------------------------
    def has(self, method_name: str) -> bool:
        return method_name in self.gates

    def entry(self, method_name: str) -> dict:
        try:
            return self.gates[method_name]
        except KeyError:
            raise KeyError(
                f"device card {self.architecture}/{self.model} has no entry for "
                f"gate {method_name!r}; available: {sorted(self.gates)}"
            )

    def duration(self, method_name: str) -> float:
        return float(self.entry(method_name)["T_ns"])

    def error_probability(self, method_name: str) -> float:
        """1 - P(no Pauli error) for the gate."""
        pe = self.entry(method_name)["pauli_error"]
        identity = "I" * int(self.entry(method_name).get("wires", 1))
        return max(0.0, 1.0 - float(pe[identity]))

    # Idle channel ----------------------------------------------------------
    def idle_ptm_diagonal(self, duration_ns: float) -> Optional[Dict[str, float]]:
        """Pauli transfer diagonal of the idle channel over ``duration_ns``.

        Cards written after 2026-09-23 carry a dense idle grid (50 ns steps),
        so we interpolate within it: exactly at a grid point, log-linearly
        between two, and only outside the grid do we fall back on the
        multiplicative power law. The fallback matters because the power law is
        exact only for a Markovian Pauli channel; under the pseudomode models
        the idle decay is quadratic at short times, and extrapolating a single
        50 ns window to 100 ns understates the idle error by about 1.7x
        (1.24e-4 against a directly propagated 2.15e-4).
        """
        if duration_ns <= 0.0 or not self.idle:
            return None
        grid = sorted(self.idle)
        if duration_ns in self.idle:
            return {k: max(v, 0.0)
                    for k, v in self.idle[duration_ns]["ptm_error_diag"].items()}
        lo = [T for T in grid if T < duration_ns]
        hi = [T for T in grid if T > duration_ns]
        if lo and hi:
            a, b = max(lo), min(hi)
            da = self.idle[a]["ptm_error_diag"]
            db = self.idle[b]["ptm_error_diag"]
            w = (duration_ns - a) / (b - a)
            out = {}
            for k in da:
                va, vb = max(da[k], 1e-300), max(db[k], 1e-300)
                # PTM diagonals decay multiplicatively, so interpolate in log.
                out[k] = float(np.exp((1.0 - w) * np.log(va) + w * np.log(vb)))
            return out
        base_T = max(lo) if lo else min(hi)
        diag = self.idle[base_T]["ptm_error_diag"]
        power = duration_ns / base_T
        return {k: max(v, 0.0) ** power for k, v in diag.items()}

    def idle_error_probability(self, duration_ns: float) -> float:
        """1 - P(no error) for one wire idling ``duration_ns``."""
        diag = self.idle_ptm_diagonal(duration_ns)
        if diag is None:
            return 0.0
        # p_I = (1/4) sum_Q R_QQ for one qubit.
        p_id = sum(diag.values()) / len(diag)
        return max(0.0, 1.0 - p_id)


@dataclass
class Schedule:
    layers: List[List[object]]          # gates per layer (ASAP)
    start_ns: Dict[int, float]          # innovation_number -> start time
    makespan_ns: float
    busy_ns: Dict[int, float]           # wire -> total time occupied by gates
    idle_ns: Dict[int, float]           # wire -> makespan - busy


def asap_schedule(genome, card: DeviceCard) -> Schedule:
    """As-soon-as-possible schedule: a gate starts when all its wires are
    free.  Gate order is the genome's depth order, so the schedule is a
    deterministic function of the genome."""
    genome.sort_gates()
    wire_free_at: Dict[int, float] = {w: 0.0 for w in range(len(genome.qubits))}
    busy: Dict[int, float] = {w: 0.0 for w in wire_free_at}
    start: Dict[int, float] = {}
    layer_of: Dict[int, int] = {}
    layers: List[List[object]] = []
    for gate in genome.gates:
        if not gate.enabled:
            continue
        wires = gate.get_pennylane_wires(genome.qubits)
        T = card.duration(gate.method_name)
        t0 = max(wire_free_at[w] for w in wires) if wires else 0.0
        start[gate.innovation_number] = t0
        for w in wires:
            wire_free_at[w] = t0 + T
            busy[w] += T
        # Layer index = 1 + max layer of any predecessor on these wires.
        pred_layers = [layer_of.get(w, -1) for w in wires]
        li = max(pred_layers) + 1 if pred_layers else 0
        while len(layers) <= li:
            layers.append([])
        layers[li].append(gate)
        for w in wires:
            layer_of[w] = li
    makespan = max(wire_free_at.values()) if wire_free_at else 0.0
    idle = {w: makespan - busy[w] for w in busy}
    return Schedule(layers, start, makespan, busy, idle)


def hardware_metrics(genome, card: DeviceCard,
                     idle_wires: Optional[Sequence[int]] = None) -> dict:
    """Compute the three objectives (and a few diagnostics) for a genome.

    Args:
        genome: CircuitGenome.
        card: DeviceCard.
        idle_wires: wires that accumulate idle error.  Default: every wire
            that is touched by at least one gate or is an output wire.
    """
    sched = asap_schedule(genome, card)
    log_success = 0.0
    n_2q = 0
    n_gates = 0
    for gate in genome.gates:
        if not gate.enabled:
            continue
        n_gates += 1
        eps = card.error_probability(gate.method_name)
        if len(gate.get_pennylane_wires(genome.qubits)) >= 2:
            n_2q += 1
        log_success += _log1m(eps)
    if idle_wires is None:
        touched = {w for g in genome.gates if g.enabled
                   for w in g.get_pennylane_wires(genome.qubits)}
        idle_wires = sorted(touched | set(genome.output_indexes))
    idle_err_total = 0.0
    for w in idle_wires:
        eps_idle = card.idle_error_probability(sched.idle_ns.get(w, 0.0))
        idle_err_total += eps_idle
        log_success += _log1m(eps_idle)
    import math
    p_fail = 1.0 - math.exp(log_success)
    return {
        "p_fail": p_fail,
        "n_entanglers": n_2q,
        "total_time_ns": sched.makespan_ns,
        "n_gates": n_gates,
        "n_layers": len(sched.layers),
        "expected_shots_per_success": (1.0 / (1.0 - p_fail)) if p_fail < 1.0 else float("inf"),
        "idle_error_sum": idle_err_total,
    }


def _log1m(p: float) -> float:
    import math
    p = min(max(p, 0.0), 1.0 - 1e-16)
    return math.log1p(-p)


def attach_hardware_fitness(genome, card: DeviceCard, prefix: str = "") -> dict:
    """Compute the metrics and merge them into ``genome.fitness`` (creating
    it if absent) so they sit next to the task loss.  Returns the metrics."""
    m = hardware_metrics(genome, card)
    if getattr(genome, "fitness", None) is None:
        genome.fitness = {}
    for k in ("p_fail", "n_entanglers", "total_time_ns"):
        genome.fitness[prefix + k] = m[k]
    return m
