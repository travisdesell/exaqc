"""Standard (closed-system) preparation circuits against learned circuits,
simulated under every open-system model.

The question: textbook state-preparation circuits are derived in a closed
system, where they are exact and minimal.  Run them under an open-system model
and they are neither.  A circuit learned against that model's device card is
usually longer -- more gates, sometimes more layers -- yet should prepare the
state better, because the extra structure buys something the closed-system
derivation had no reason to want.  This script measures that for every task we
have: the four Bell states and GHZ_3 / GHZ_4.

Standard circuits, all from |0...0>:
  Phi+   H_0 CX_{01}
  Phi-   H_0 Z_0 CX_{01}
  Psi+   X_1 H_0 CX_{01}
  Psi-   X_1 H_0 Z_0 CX_{01}
  GHZ_n  H_0 then a CNOT cascade, and the balanced fan-out variant

Every circuit is evaluated by a scheduled tier-1 density-matrix simulation under
each card: each gate's Pauli-twirled channel, each wire charged for the idle
intervals it actually experiences at the point they occur.

    .venv_win/Scripts/python scripts/standard_vs_learned.py \
        --family transmon_detuned --out ../../Arch_Aware_Circuit_Evolution/paper/generated/standard_vs_learned.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics  # noqa: E402
from src.objectives.state_prep_objective import (  # noqa: E402
    StatePrepObjective, TARGETS, TARGET_WIRES, state_fidelity)
from scripts.ablate_drives import greedy_prune  # noqa: E402
from scripts.reference_circuits import chain, tree, genome  # noqa: E402

MODELS = ["schrod", "lindblad_markov", "lindblad_nm", "mixed"]
OPEN_MODELS = ["lindblad_markov", "lindblad_nm", "mixed"]
SHORT = {"schrod": "Schrod", "lindblad_markov": "Lind-M",
         "lindblad_nm": "Lind-NM", "mixed": "Mixed"}
CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")
Gate = Tuple[str, List[int]]

# Task -> directory of search results, and the standard circuit(s).
BELL_STANDARD: Dict[str, List[Gate]] = {
    "phi_plus":  [("h", [0]), ("cx", [0, 1])],
    "phi_minus": [("h", [0]), ("z", [0]), ("cx", [0, 1])],
    "psi_plus":  [("x", [1]), ("h", [0]), ("cx", [0, 1])],
    "psi_minus": [("x", [1]), ("h", [0]), ("z", [0]), ("cx", [0, 1])],
}


def standard_circuits(task: str) -> Dict[str, List[Gate]]:
    if task in BELL_STANDARD:
        return {"standard": BELL_STANDARD[task]}
    n = TARGET_WIRES[task]
    return {"standard (cascade)": chain(n), "standard (fan-out)": tree(n)}


def results_dir(task: str) -> str:
    return "bell4" if task in BELL_STANDARD else task


def learned_for(task: str, model: str, family: str) -> Optional[List[Gate]]:
    """Best (lowest p_fail) circuit the search returned for this card."""
    d = results_dir(task)
    if task in BELL_STANDARD:
        pats = [f"results/{d}/{task}__{family}_{model}_s*/summary.json"]
    else:
        pats = [f"results/{d}/{family}_{model}_s*/summary.json",
                f"results/{d}/{model}_s*/summary.json"]
    rows = []
    for pat in pats:
        for p in glob.glob(pat):
            rows.extend(json.load(open(p))["front"])
        if rows:
            break
    if not rows:
        return None
    b = min(rows, key=lambda r: r["fitness"]["p_fail"])
    return [(g, list(w)) for g, w in b["gates"]]


def evaluate(gates: Sequence[Gate], task: str, card: DeviceCard,
             noise: DeviceCardNoiseModel) -> Dict[str, float]:
    n_wires = TARGET_WIRES[task]
    obj = StatePrepObjective(card, task, noisy=False)
    G = genome(list(gates), n_wires)
    obj(G)
    closed_loss = G.fitness["task_loss"]
    hm = hardware_metrics(genome(list(gates), n_wires), card)
    # tier-1: density matrix with every card channel, scheduled idle included
    dev, qnode = genome(list(gates), n_wires).generate_pennylane_circuit(
        noise_model=noise, input_mode="basis", return_probs=False,
        measure_registers=False)
    import torch
    rho = qnode(torch.zeros(n_wires, dtype=torch.int64), None)
    rho = rho.detach().cpu().numpy() if hasattr(rho, "detach") else np.asarray(rho)
    F = state_fidelity(rho, TARGETS[task], n_wires,
                       list(range(TARGET_WIRES[task])), strict_ancilla=True)
    return {"infidelity": 1.0 - F, "closed_task_loss": closed_loss,
            "n_gates": len(gates), "time_ns": hm["total_time_ns"],
            "n_entanglers": hm["n_entanglers"], "p_fail": hm["p_fail"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="transmon_detuned")
    ap.add_argument("--tasks", nargs="+",
                    default=["phi_plus", "phi_minus", "psi_plus", "psi_minus",
                             "ghz3", "ghz4"])
    ap.add_argument("--prune", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cards, noise = {}, {}
    for m in MODELS:
        p = CARDS / f"device_card_{args.family}_{m}.json"
        if p.exists():
            cards[m] = DeviceCard.load(p)
            noise[m] = DeviceCardNoiseModel(cards[m])
    print(f"card family {args.family}: {', '.join(cards)}\n")

    out: Dict[str, dict] = {}
    for task in args.tasks:
        if task not in TARGETS:
            print(f"[skip] unknown task {task}")
            continue
        circuits: Dict[str, List[Gate]] = dict(standard_circuits(task))
        for m in cards:
            g = learned_for(task, m, args.family)
            if g is None:
                continue
            circuits[f"learned@{SHORT[m]}"] = g
            if args.prune:
                circuits[f"learned@{SHORT[m]} pruned"] = greedy_prune(
                    g, TARGET_WIRES[task], cards[m], only_1q=True,
                    target=task)
        if not any(k.startswith("learned") for k in circuits):
            print(f"[skip] {task}: no learned circuits for family {args.family}")
            continue

        rows: Dict[str, Dict[str, dict]] = {}
        for name, gates in circuits.items():
            rows[name] = {}
            for m in cards:
                rows[name][m] = evaluate(gates, task, cards[m], noise[m])

        w = max(len(k) for k in rows) + 1
        print(f"=== {task}: state infidelity 1-F (scheduled tier-1), "
              f"columns = model simulated under")
        print(" " * w + "  ".join(f"{SHORT[m]:>12s}" for m in cards)
              + "   gates  T(ns)")
        for name in rows:
            r = rows[name]
            first = next(iter(cards))
            print(f"{name:{w}s}" + "  ".join(f"{r[m]['infidelity']:12.4e}" for m in cards)
                  + f"   {r[first]['n_gates']:5d}  {r[first]['time_ns']:5.0f}")
        # the comparison the user asked for
        std_key = min((k for k in rows if k.startswith("standard")),
                      key=lambda k: rows[k][OPEN_MODELS[0]]["infidelity"])
        print(f"  best standard circuit: {std_key}")
        for m in cards:
            if m == "schrod":
                continue
            best_learned = min((k for k in rows if k.startswith("learned")),
                               key=lambda k: rows[k][m]["infidelity"])
            a = rows[std_key][m]["infidelity"]
            b = rows[best_learned][m]["infidelity"]
            verdict = "learned better" if b < a else "standard better"
            print(f"   {SHORT[m]:8s} standard {a:.4e} | {best_learned} {b:.4e}"
                  f" | {a / b:6.3f}x  <- {verdict}")
        print()
        out[task] = {"circuits": {k: [[a, list(b)] for a, b in v]
                                  for k, v in circuits.items()},
                     "results": rows, "best_standard": std_key}

    if args.out:
        json.dump({"family": args.family, "models": list(cards), "tasks": out},
                  open(args.out, "w"), indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
