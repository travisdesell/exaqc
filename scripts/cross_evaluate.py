"""Circuit-level chain of degradation: learn under model k, run under model k'.

This is the circuit analogue of the gate-level chain-of-degradation table in the
companion control study.  For each task we take

  - the circuit the search returned for each device card (i.e. learned under
    model k, including k = schrod, the closed system),
  - the textbook reference circuits,
  - each learned circuit after feasibility-preserving pruning of its
    individually-deletable single-qubit gates,

and evaluate every one of them under every card (model k').  The diagonal is
"learned and run under the same model"; the row k = schrod is "learned in the
closed system and run on an open device", which is what a search with no device
physics in the loop produces.

Every entry is a scheduled tier-1 density-matrix state infidelity: each gate's
Pauli-twirled channel from the card, each wire charged for the idle intervals it
actually experiences, at the point in the circuit where they occur.

    .venv_win/Scripts/python scripts/cross_evaluate.py --task ghz3 \
        --out ../../Arch_Aware_Circuit_Evolution/paper/generated/cross_ghz3.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics  # noqa: E402
from scripts.ablate_drives import greedy_prune, feasible  # noqa: E402
from scripts.compare_distributions import distributions  # noqa: E402
from scripts.reference_circuits import chain, tree, with_drive, genome  # noqa: E402

MODELS = ["schrod", "lindblad_markov", "lindblad_nm", "mixed"]
SHORT = {"schrod": "Schrod", "lindblad_markov": "Lind-M",
         "lindblad_nm": "Lind-NM", "mixed": "Mixed"}
CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")
# Card family: "transmon_detuned" is the valid architecture; plain
# "transmon" is the resonant one kept only as a counterexample.
FAMILY = "transmon_detuned"
TASKS = {"bell": (2, "bell_pareto/final"), "ghz3": (3, "ghz3"), "ghz4": (4, "ghz4")}
Gate = Tuple[str, List[int]]


def learned(task_dir: str, model: str) -> Optional[List[Gate]]:
    rows = []
    for p in glob.glob(f"results/{task_dir}/{FAMILY}_{model}_s*/summary.json"):
        rows.extend(json.load(open(p))["front"])
    if not rows:
        return None
    best = min(rows, key=lambda r: r["fitness"]["p_fail"])
    return [(g, list(w)) for g, w in best["gates"]]


def main() -> int:
    global FAMILY
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ghz3")
    ap.add_argument("--prune", action="store_true",
                    help="also evaluate each learned circuit after pruning")
    ap.add_argument("--out", default=None)
    ap.add_argument("--family", default=FAMILY,
                    help="device-card family "
                         "(transmon_detuned = valid; transmon = resonant)")
    args = ap.parse_args()
    FAMILY = args.family
    print(f"card family: {FAMILY}")
    n, task_dir = TASKS[args.task]

    cards = {m: DeviceCard.load(CARDS / f"device_card_{FAMILY}_{m}.json")
             for m in MODELS if (CARDS / f"device_card_{FAMILY}_{m}.json").exists()}
    noise = {m: DeviceCardNoiseModel(c) for m, c in cards.items()}

    circuits: Dict[str, List[Gate]] = {}
    for m in MODELS:
        g = learned(task_dir, m)
        if g is not None:
            circuits[f"learned@{SHORT[m]}"] = g
            if args.prune:
                circuits[f"learned@{SHORT[m]} pruned"] = greedy_prune(
                    g, n, cards[m], only_1q=True, target=args.task)
    circuits["textbook cascade"] = chain(n)
    circuits["textbook fan-out"] = tree(n)

    # Evaluate everything under every card.
    table: Dict[str, Dict[str, float]] = {}
    extra: Dict[str, dict] = {}
    for name, gates in circuits.items():
        row = {}
        for m, card in cards.items():
            if not feasible(gates, n, card):
                row[m] = float("nan")
                continue
            d = distributions(gates, n, noise[m])
            row[m] = 1.0 - d["fidelity"]
        table[name] = row
        hm = hardware_metrics(genome(list(gates), n), list(cards.values())[0])
        extra[name] = {"n_gates": len(gates), "time_ns": hm["total_time_ns"],
                       "n_entanglers": hm["n_entanglers"],
                       "gates": [[a, list(b)] for a, b in gates]}

    order = [k for k in circuits]
    w = max(len(k) for k in order) + 1
    print(f"\n=== {args.task}: state infidelity 1-F, rows = circuit, columns = "
          f"model it is RUN under")
    print(" " * w + "  ".join(f"{SHORT[m]:>12s}" for m in cards)
          + "   gates  T(ns)")
    for k in order:
        print(f"{k:{w}s}" + "  ".join(f"{table[k][m]:12.4e}" for m in cards)
              + f"   {extra[k]['n_gates']:5d}  {extra[k]['time_ns']:5.0f}")

    # The comparison the table exists to make: for each open-system model,
    # how much worse is the closed-system-learned circuit than the circuit
    # learned under that model itself?
    print("\n--- closed-system-learned vs model-learned, evaluated under each model")
    for m in cards:
        if m == "schrod":
            continue
        a = table.get(f"learned@{SHORT['schrod']}", {}).get(m)
        b = table.get(f"learned@{SHORT[m]}", {}).get(m)
        cas = table["textbook cascade"][m]
        if a is None or b is None:
            continue
        print(f"  run under {SHORT[m]:8s}: learned@Schrod {a:.4e} | "
              f"learned@{SHORT[m]} {b:.4e} | ratio {a / b:6.3f}x | "
              f"textbook cascade {cas:.4e} ({cas / b:.3f}x)")

    if args.out:
        json.dump({"task": args.task, "n": n, "models": list(cards),
                   "infidelity": table, "circuits": extra},
                  open(args.out, "w"), indent=2)
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
