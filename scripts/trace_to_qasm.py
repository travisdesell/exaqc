"""Emit OpenQASM for the circuits in a search trace.

By default writes the initial guess (iteration 0 -- the textbook circuit the
search was seeded from) and the final circuit, plus a barrier-annotated listing
of the ASAP schedule so the idle gaps the device card charges for are visible in
the QASM comments. --all writes every improving circuit.

QASM has no notion of pulse duration or idle cost, which is the whole point of
the device card, so the timing is carried in comments rather than silently lost.

    .venv_win/Scripts/python scripts/trace_to_qasm.py \
        --trace results/trace_ghz4_mixed.json --which 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Gate = Tuple[str, List[int]]

# exaqc gate name -> QASM 2.0 gate
QASM = {"h": "h", "x": "x", "z": "z", "s": "s", "t": "t",
        "sdg": "sdg", "tdg": "tdg", "cx": "cx"}


def schedule(gates: Sequence[Gate], n: int, dur: Dict[str, float]):
    free = [0.0] * n
    out = []
    for name, wires in gates:
        t0 = max(free[w] for w in wires)
        out.append((t0, name, list(wires)))
        for w in wires:
            free[w] = t0 + dur[name]
    return out, (max(free) if free else 0.0)


def to_qasm(gates: Sequence[Gate], n: int, dur: Dict[str, float],
            header: Sequence[str]) -> str:
    lines = [f"// {h}" for h in header]
    lines += ["OPENQASM 2.0;", 'include "qelib1.inc";', "",
              f"qreg q[{n}];", f"creg c[{n}];", ""]
    placed, total = schedule(gates, n, dur)
    # Sort by start time: the genome order is a valid topological order but is
    # not monotonic in time, so the emitted comments looked out of sequence.
    # Sorting preserves per-wire order (gates sharing a wire have distinct
    # start times) and only permutes gates that commute, so the circuit is
    # unchanged -- checked against the simulated state.
    placed = sorted(placed, key=lambda x: (x[0], min(x[2])))
    lines.append(f"// ASAP schedule, total {total:.0f} ns "
                 f"(times in ns; QASM cannot express them)")
    lines.append("// barrier marks an ASAP layer boundary")
    busy = [0.0] * n
    prev_start = [None]
    for t0, name, wires in placed:
        if prev_start[0] is not None and t0 != prev_start[0]:
            lines.append("barrier q;")
        prev_start[0] = t0
        for w in wires:
            gap = t0 - busy[w]
            if gap > 1e-9:
                lines.append(f"//   q[{w}] idles {gap:.0f} ns before this gate")
        args = ",".join(f"q[{w}]" for w in wires)
        # QASM's cx argument order IS control,target, but spelling it out costs
        # nothing and removes the commonest source of transcription errors.
        role = (f"control q[{wires[0]}] -> target q[{wires[1]}], "
                if name == "cx" else "")
        lines.append(f"{QASM[name]} {args};"
                     f"   // {role}t = {t0:.0f}-{t0 + dur[name]:.0f} ns")
        for w in wires:
            busy[w] = t0 + dur[name]
    for w in range(n):
        gap = total - busy[w]
        if gap > 1e-9:
            lines.append(f"//   q[{w}] idles {gap:.0f} ns to the end")
    lines += ["", f"measure q -> c;"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--card", default=None)
    ap.add_argument("--which", default="0,last",
                    help="comma-separated iteration indices, or 'all'; "
                         "'last' means the final circuit")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    d = json.load(open(args.trace))
    tr = d["trace"]
    n = int(d["n_wires"])
    cardp = args.card or (
        Path("../../Arch_Aware_Circuit_Evolution/device_cards")
        / f"device_card_{d['architecture']}_{d['model']}.json")
    card = json.load(open(cardp))
    dur = {g: float(e["T_ns"]) for g, e in card["gates"].items()}

    if args.which == "all":
        picks = list(range(len(tr)))
    else:
        picks = []
        for tok in args.which.split(","):
            picks.append(len(tr) - 1 if tok.strip() == "last" else int(tok))

    for i in picks:
        r = tr[i]
        gates = [(a, list(b)) for a, b in r["gates"]]
        header = [
            f"{d['task']} on {d['architecture']} / {d['model']}",
            f"iteration {r['iteration']} ({r['tag']})",
            f"gates {r['n_gates']}, CNOTs {r['n_entanglers']}, "
            f"duration {r['time_ns']:.0f} ns",
            f"p_fail (card, tier 0) = {r['p_fail']:.6e}",
            f"state infidelity (tier 1, density matrix) = {r['infidelity']:.6e}",
            f"closed-system task loss = {r['task_loss']:.2e} "
            f"(0 means it prepares the target exactly)",
        ]
        q = to_qasm(gates, n, dur, header)
        if args.out_dir:
            p = Path(args.out_dir) / f"{d['task']}_{d['model']}_it{r['iteration']:03d}.qasm"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(q, encoding="utf-8")
            print(f"wrote {p}")
        else:
            print(q)
            print("-" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
