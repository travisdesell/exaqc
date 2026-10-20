"""Feasibility-preserving ablation: does the learned circuit's advantage come
from its inert single-qubit drives, or from something else?

For a learned circuit we build a ladder of variants, each of which must still
prepare the target state exactly:

  full          the learned circuit as evolved
  no-drive      every single-qubit gate that acts trivially on the state it
                sees (i.e. can be deleted without changing the prepared state)
                removed, checked by re-evaluating the noiseless fidelity
  entanglers    only the two-qubit gates plus the minimum single-qubit gates
                needed for feasibility (greedy deletion to a local minimum)

and, separately, each variant is evaluated under EVERY card, so the
model-dependence of one fixed circuit is isolated from the model-dependence of
what the search happened to return.

    .venv_win/Scripts/python scripts/ablate_drives.py --task ghz3 --model lindblad_nm
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.state_prep_objective import (  # noqa: E402
    StatePrepObjective, TARGET_WIRES)
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics  # noqa: E402
from scripts.reference_circuits import genome  # noqa: E402
from scripts.compare_distributions import distributions  # noqa: E402

MODELS = ["schrod", "lindblad_markov", "lindblad_nm", "mixed"]
CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")
Gate = Tuple[str, List[int]]


def feasible(gates: Sequence[Gate], n: int, card: DeviceCard, tol: float = 1e-9,
             target: str = None) -> bool:
    """Does this gate list still prepare the target exactly (closed system)?

    `target` names the state; it defaults to GHZ_n for backwards compatibility,
    but callers working on the Bell states MUST pass it -- GHZ_2 is Phi+, so the
    default silently validates Phi-/Psi+/Psi- against the wrong state.
    """
    obj = StatePrepObjective(card, target or f"ghz{n}", noisy=False)
    G = genome(list(gates), n)
    obj(G)
    return G.fitness["task_loss"] <= tol


def greedy_prune(gates: Sequence[Gate], n: int, card: DeviceCard,
                 only_1q: bool = True, target: str = None) -> List[Gate]:
    """Delete gates one at a time, keeping every deletion that preserves exact
    preparation of `target`. Returns a local minimum under single deletions."""
    cur = list(gates)
    changed = True
    while changed:
        changed = False
        for i in range(len(cur)):
            if only_1q and len(cur[i][1]) > 1:
                continue
            trial = cur[:i] + cur[i + 1:]
            if feasible(trial, n, card, target=target):
                cur = trial
                changed = True
                break
    return cur


def best_learned(task_dir: str, model: str) -> List[Gate] | None:
    rows = []
    for p in glob.glob(f"results/{task_dir}/{model}_s*/summary.json"):
        rows.extend(json.load(open(p))["front"])
    if not rows:
        return None
    best = min(rows, key=lambda r: r["fitness"]["p_fail"])
    return [(g, list(w)) for g, w in best["gates"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ghz3")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--model", default="lindblad_nm",
                    help="card whose learned circuit is ablated")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    task_dir = {"bell": "bell_pareto/final", "ghz3": "ghz3", "ghz4": "ghz4"}[args.task]
    n = args.n or {"bell": 2, "ghz3": 3, "ghz4": 4}[args.task]

    home = DeviceCard.load(CARDS / f"device_card_transmon_{args.model}.json")
    learned = best_learned(task_dir, args.model)
    if learned is None:
        print("no learned circuit found")
        return 1
    pruned = greedy_prune(learned, n, home, only_1q=True)
    variants: Dict[str, List[Gate]] = {"full": list(learned), "pruned-1q": pruned}
    n_removed = len(learned) - len(pruned)

    print(f"task {args.task}  learned-on {args.model}")
    print(f"  full      {len(learned):2d} gates: {learned}")
    print(f"  pruned-1q {len(pruned):2d} gates ({n_removed} inert 1q gates removed): {pruned}")
    print()
    out: Dict[str, dict] = {}
    hdr = f"{'card':16s} " + "  ".join(f"{k:>12s}" for k in variants) + "   ratio(full/pruned)"
    print(hdr)
    for m in MODELS:
        cp = CARDS / f"device_card_transmon_{m}.json"
        if not cp.exists():
            continue
        card = DeviceCard.load(cp)
        nm = DeviceCardNoiseModel(card)
        row = {}
        for name, gates in variants.items():
            assert feasible(gates, n, card), f"{name} infeasible under {m}"
            d = distributions(gates, n, nm)
            hm = hardware_metrics(genome(list(gates), n), card)
            row[name] = {"infidelity": 1.0 - d["fidelity"],
                         "p_fail": hm["p_fail"],
                         "time_ns": hm["total_time_ns"],
                         "n_gates": len(gates)}
        ratio = row["full"]["infidelity"] / row["pruned-1q"]["infidelity"]
        print(f"{m:16s} " + "  ".join(f"{row[k]['infidelity']:12.4e}" for k in variants)
              + f"   {ratio:.3f}"
              + ("  <- drives help" if ratio < 0.98 else
                 "  <- drives hurt" if ratio > 1.02 else "  <- no effect"))
        out[m] = row
    if args.out:
        json.dump({"task": args.task, "learned_on": args.model,
                   "variants": {k: [[a, list(b)] for a, b in v] for k, v in variants.items()},
                   "results": out}, open(args.out, "w"), indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
