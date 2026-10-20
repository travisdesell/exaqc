"""Evaluate reference (hand-built) preparation circuits under every device card
and compare them to the evolved optima.

Four references per task where they differ:

  chain       H then a linear cascade of CNOTs (what a naive compiler emits)
  tree        H then a balanced binary fan-out: same entangler count, fewer
              layers, so a shorter circuit in nanoseconds
  chain+drive chain with every idle spectator window filled by T pulses
  tree+drive  tree with the same filling -- the combination of the textbook
              depth optimum and the spectator-drive effect the search found

T is diagonal, so T|0> = |0> up to phase and a wire still in |0> can absorb
any number of them without changing the prepared state; the script asserts
this by checking the task loss of every reference.

    .venv_win/Scripts/python scripts/reference_circuits.py \
        --out ../../Arch_Aware_Circuit_Evolution/paper/generated/textbook_pfail.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.circuits.registers import expand_registers  # noqa: E402
from src.objectives.ghz_prep_objective import GHZPrepObjective  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard, asap_schedule  # noqa: E402

MODELS = ["schrod", "lindblad_markov", "lindblad_nm", "mixed"]
CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")
Gate = Tuple[str, List[int]]


def chain(n: int) -> List[Gate]:
    return [("h", [0])] + [("cx", [i, i + 1]) for i in range(n - 1)]


def tree(n: int) -> List[Gate]:
    """Balanced fan-out: at each round every wire already in the state copies
    itself onto a fresh wire, so the entangler count is n-1 as for the chain
    but the layer count is ceil(log2 n)."""
    gates: List[Gate] = [("h", [0])]
    have = [0]
    nxt = 1
    while nxt < n:
        new = []
        for src in have:
            if nxt >= n:
                break
            gates.append(("cx", [src, nxt]))
            new.append(nxt)
            nxt += 1
        have = have + new
    return gates


def genome(gates: Sequence[Gate], n_wires: int) -> CircuitGenome:
    inp = expand_registers({"q": n_wires})
    g = CircuitGenome(genome_number=0, target="pennylane",
                      input_qubits=inp.copy(), output_qubits=inp.copy())
    for i, (name, wires) in enumerate(gates):
        g.add_gate(depth=0.02 + 0.96 * i / max(1, len(gates)), method_name=name,
                   qubits=[("q", w) for w in wires])
    return g


def with_drive(gates: Sequence[Gate], n_wires: int, card: DeviceCard) -> List[Gate]:
    """Fill each wire's pre-entanglement idle window with T pulses.

    A wire is safe to drive only while it is still in |0>, i.e. before the
    first two-qubit gate that touches it; driving it later would change the
    prepared state.  The number of pulses is that wire's idle time before its
    first entangler, divided by the T duration.
    """
    G = genome(gates, n_wires)          # one genome: innovation numbers must match
    sched = asap_schedule(G, card)
    T_t = card.duration("t")
    first_cx_start: Dict[int, float] = {}
    for g in G.gates:
        if not g.enabled:
            continue
        wires = g.get_pennylane_wires(G.qubits)
        if len(wires) < 2:
            continue
        t0 = sched.start_ns[g.innovation_number]
        for w in wires:
            first_cx_start.setdefault(w, t0)
            first_cx_start[w] = min(first_cx_start[w], t0)
    fill: List[Gate] = []
    for w in range(n_wires):
        window = first_cx_start.get(w, 0.0)
        # wire 0 is busy with the H; every other wire idles from t=0
        busy = card.duration("h") if w == 0 else 0.0
        n_pulses = int(max(0.0, window - busy) // T_t)
        fill += [("t", [w])] * n_pulses
    # Drives first: they must be scheduled into the idle window, and the ASAP
    # scheduler respects the genome's depth order.
    return [gates[0]] + fill + list(gates[1:])


def best_evolved(task_dir: str, model: str):
    rows = []
    for p in glob.glob(f"results/{task_dir}/{model}_s*/summary.json"):
        rows.extend(json.load(open(p))["front"])
    return min(rows, key=lambda r: r["fitness"]["p_fail"]) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="assert every reference prepares the state this well")
    args = ap.parse_args()

    tasks = {"bell": (2, "bell_pareto/final"), "ghz3": (3, "ghz3"), "ghz4": (4, "ghz4")}
    out: Dict[str, dict] = {}
    for tag, (n, task_dir) in tasks.items():
        per_model = {}
        for m in MODELS:
            cp = CARDS / f"device_card_transmon_{m}.json"
            if not cp.exists():
                continue
            card = DeviceCard.load(cp)
            obj = GHZPrepObjective(card, n_target=n, noisy=False)
            refs = {"chain": chain(n), "tree": tree(n)}
            refs["chain+drive"] = with_drive(refs["chain"], n, card)
            refs["tree+drive"] = with_drive(refs["tree"], n, card)
            got = {}
            for name, gates in refs.items():
                G = genome(gates, n)
                obj(G)
                assert G.fitness["task_loss"] <= args.tol, \
                    f"{tag}/{m}/{name} does not prepare the state: {G.fitness['task_loss']}"
                got[name] = {"p_fail": G.fitness["p_fail"],
                             "time_ns": G.fitness["total_time_ns"],
                             "n_entanglers": G.fitness["n_entanglers"],
                             "n_gates": len(gates),
                             "gates": [[a, b] for a, b in gates]}
            ev = best_evolved(task_dir, m)
            if ev is None:
                continue
            best_ref_name = min(got, key=lambda k: got[k]["p_fail"])
            per_model[m] = {
                "textbook": got["chain"]["p_fail"],     # consumed by the report table
                "evolved": ev["fitness"]["p_fail"],
                "evolved_time_ns": ev["fitness"]["total_time_ns"],
                "evolved_n_gates": len(ev["gates"]),
                "evolved_gates": ev["gates"],
                "references": got,
                "best_reference": best_ref_name,
            }
            print(f"{tag:5s} {m:16s} " + "  ".join(
                f"{k}={got[k]['p_fail']:.4e}@{got[k]['time_ns']:.0f}ns" for k in
                ("chain", "tree", "chain+drive", "tree+drive"))
                  + f"  || evolved={ev['fitness']['p_fail']:.4e}"
                    f"@{ev['fitness']['total_time_ns']:.0f}ns"
                    f"  best_ref/evolved="
                    f"{ev['fitness']['p_fail'] / got[best_ref_name]['p_fail']:.2f}x")
        if per_model:
            out[tag] = per_model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
