"""Merge the Pareto fronts of several restarts into one non-dominated set.

    .venv_win/Scripts/python scripts/merge_pareto.py results/ghz3/mixed_s*/summary.json \
        --out results/ghz3/mixed_union.json

Prints the union front sorted by p_fail and writes it as JSON.  Restarts are
necessary because the mutation-only search reaches feasible local optima that
no single gate deletion escapes; the union is the honest front for a card.
"""
from __future__ import annotations

import argparse
import glob
import json
from typing import List, Sequence

KEYS = ["p_fail", "n_entanglers", "total_time_ns"]


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = [p for pat in args.summaries for p in glob.glob(pat)]
    rows: List[dict] = []
    meta = {}
    for p in paths:
        d = json.load(open(p))
        meta = {k: d.get(k) for k in ("task", "model", "architecture", "n_target",
                                      "wires", "gates", "task_tol", "virtual_z")}
        for r in d["front"]:
            rows.append({"fitness": r["fitness"], "gates": r["gates"], "source": p,
                         "seed": d.get("seed")})
    pts = [[float(r["fitness"][k]) for k in KEYS] for r in rows]
    front = [i for i in range(len(rows))
             if not any(dominates(pts[j], pts[i]) for j in range(len(rows)) if j != i)]
    # De-duplicate identical objective vectors, keeping the fewest-gate circuit.
    best = {}
    for i in front:
        key = tuple(pts[i])
        if key not in best or len(rows[i]["gates"]) < len(rows[best[key]]["gates"]):
            best[key] = i
    keep = sorted(best.values(), key=lambda i: pts[i])

    print(f"{len(paths)} runs, {len(rows)} front points in, {len(keep)} non-dominated out")
    for i in keep:
        f = rows[i]["fitness"]
        print(f"  p_fail={f['p_fail']:.4e}  cx={int(f['n_entanglers'])}  "
              f"T={f['total_time_ns']:.0f} ns  layers={int(f.get('n_layers', 0))}  "
              f"noisyF={f.get('noisy_fidelity', float('nan')):.5f}  "
              f"seed={rows[i]['seed']}  gates={rows[i]['gates']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "n_runs": len(paths),
                       "front": [rows[i] for i in keep]}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
