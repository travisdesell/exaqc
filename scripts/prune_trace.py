"""Apply the prune stage to every circuit in a trace, and re-evaluate.

The search accumulates single-qubit gates that lower p_fail without doing
anything to the state -- the screening objective is state-blind, so a Z-family
gate on a wire sitting in |0> looks like it removes charged idle time while
costing nothing. Step 5 of the procedure deletes them: greedily drop any gate
whose removal preserves exact preparation in the closed system.

This script runs that stage over an existing trace and writes a new trace with
both numbers per step, so the movie can show the pruned circuit and the
before/after gap is on record rather than lost.

    .venv_win/Scripts/python scripts/prune_trace.py \
        --trace results/trace_ghz4_mixed.json --out results/trace_ghz4_mixed_pruned.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics  # noqa: E402
from src.objectives.state_prep_objective import (  # noqa: E402
    TARGETS, StatePrepObjective, state_fidelity)
from scripts.ablate_drives import greedy_prune  # noqa: E402
from scripts.reference_circuits import genome  # noqa: E402

Gate = Tuple[str, List[int]]


def evaluate(gates: List[Gate], task: str, n: int, card: DeviceCard,
             nm: DeviceCardNoiseModel) -> Dict[str, float]:
    obj = StatePrepObjective(card, task, noisy=False)
    G = genome(list(gates), n)
    obj(G)
    hm = hardware_metrics(genome(list(gates), n), card)
    dev, qnode = genome(list(gates), n).generate_pennylane_circuit(
        noise_model=nm, input_mode="basis", return_probs=False,
        measure_registers=False)
    rho = qnode(torch.zeros(n, dtype=torch.int64), None)
    rho = rho.detach().cpu().numpy() if hasattr(rho, "detach") else np.asarray(rho)
    n_t = int(round(np.log2(len(TARGETS[task]))))
    F = state_fidelity(rho, TARGETS[task], n, list(range(n_t)), strict_ancilla=True)
    return {"infidelity": 1.0 - F, "p_fail": hm["p_fail"],
            "time_ns": hm["total_time_ns"], "n_gates": len(gates),
            "n_entanglers": hm["n_entanglers"],
            "task_loss": G.fitness["task_loss"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--card", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.trace))
    # Traces from evolve_ghz_pareto label the task "GHZ_4"; the target registry
    # keys are "ghz4". Normalise rather than depend on which driver wrote it.
    task = d["task"].replace("GHZ_", "ghz").strip()
    if task not in TARGETS:
        raise SystemExit(f"unknown task {d['task']!r} -> {task!r}; "
                         f"have {sorted(TARGETS)}")
    n = int(d["n_wires"])
    cardp = args.card or (
        Path("../../Arch_Aware_Circuit_Evolution/device_cards")
        / f"device_card_{d['architecture']}_{d['model']}.json")
    card = DeviceCard.load(cardp)
    nm = DeviceCardNoiseModel(card)

    print(f"{task} / {d['model']}: pruning {len(d['trace'])} traced circuits\n")
    print(f"{'it':>3s} {'raw gates':>9s} {'pruned':>7s} "
          f"{'1-F raw':>11s} {'1-F pruned':>11s} {'gain':>7s} {'T raw':>6s} {'T pr':>6s}")
    out = []
    for r in d["trace"]:
        raw = [(a, list(b)) for a, b in r["gates"]]
        pruned = greedy_prune(raw, n, card, only_1q=True, target=task)
        a = evaluate(raw, task, n, card, nm)
        b = evaluate(pruned, task, n, card, nm)
        assert b["task_loss"] < 1e-9, f"prune broke iteration {r['iteration']}"
        gain = a["infidelity"] / b["infidelity"] if b["infidelity"] > 0 else float("nan")
        print(f"{r['iteration']:3d} {a['n_gates']:9d} {b['n_gates']:7d} "
              f"{a['infidelity']:11.4e} {b['infidelity']:11.4e} {gain:7.3f} "
              f"{a['time_ns']:6.0f} {b['time_ns']:6.0f}")
        rec = dict(r)
        rec.update({"gates": [[x, list(y)] for x, y in pruned],
                    "gates_raw": [[x, list(y)] for x, y in raw],
                    "infidelity": b["infidelity"], "infidelity_raw": a["infidelity"],
                    "p_fail": b["p_fail"], "p_fail_raw": a["p_fail"],
                    "time_ns": b["time_ns"], "n_gates": b["n_gates"],
                    "n_entanglers": b["n_entanglers"],
                    "n_pruned": a["n_gates"] - b["n_gates"]})
        out.append(rec)

    # Deduplicate: pruning often maps several raw circuits onto the same
    # circuit, and a movie should not sit on repeated frames.
    dedup, seen = [], None
    for r in out:
        key = tuple((a, tuple(b)) for a, b in r["gates"])
        if key == seen:
            continue
        seen = key
        r = dict(r)
        r["iteration"] = len(dedup)
        dedup.append(r)
    print(f"\n{len(out)} steps -> {len(dedup)} distinct circuits after pruning")

    d2 = dict(d)
    d2["trace"] = dedup
    d2["pruned"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(d2, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
