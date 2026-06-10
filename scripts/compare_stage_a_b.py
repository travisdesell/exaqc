"""Compare Stage A vs Stage B results cell-by-cell.

Stage A: src/Ryan_cookin/results/stage_a.csv (or stage_a_trained.csv)
Stage B: src/Ryan_cookin/results/stage_b.csv

Both have one row per (dataset, encoder, n_qubits, seed). For each cell
we produce a side-by-side row with test_acc / test_loss for A and B,
the delta (B - A), and a "winner" column.

Usage:
    python scripts/compare_stage_a_b.py
    python scripts/compare_stage_a_b.py --a stage_a_trained.csv --b stage_b.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_csv(path: str) -> dict[tuple, dict]:
    """Index rows by (dataset, encoder, n_qubits, seed)."""
    out: dict[tuple, dict] = {}
    if not os.path.exists(path):
        print(f"  (missing) {path}")
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["dataset"],
                row["encoder"],
                int(row["n_qubits"]),
                int(row["seed"]),
            )
            out[key] = row
    return out


def fmt_delta(d: float) -> str:
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.3f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="src/Ryan_cookin/results/stage_a.csv")
    p.add_argument("--b", default="src/Ryan_cookin/results/stage_b.csv")
    p.add_argument("--out", default="src/Ryan_cookin/results/stage_ab_compare.csv")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    a_rows = load_csv(args.a)
    b_rows = load_csv(args.b)

    keys = sorted(set(a_rows) | set(b_rows))

    out_rows = []
    a_wins = 0
    b_wins = 0
    ties = 0

    by_dataset: dict[str, list[float]] = defaultdict(list)  # acc deltas per dataset

    print(
        f"{'dataset':14s} {'encoder':16s} {'N':>2s} | "
        f"{'A_acc':>6s} {'B_acc':>6s} {'dacc':>7s} | "
        f"{'A_loss':>7s} {'B_loss':>7s} {'dloss':>7s} | "
        f"{'B_gates':>7s} winner"
    )
    print("-" * 110)

    for key in keys:
        dataset, encoder, n, seed = key
        a = a_rows.get(key)
        b = b_rows.get(key)

        if a is None or b is None:
            status = "A missing" if a is None else "B missing"
            print(
                f"{dataset:14s} {encoder:16s} {n:2d} | "
                f"{status}"
            )
            continue

        a_acc = float(a["test_acc"])
        b_acc = float(b["test_acc"])
        a_loss = float(a["test_loss"])
        b_loss = float(b["test_loss"])
        b_gates = b.get("best_n_gates", "?")

        d_acc = b_acc - a_acc
        d_loss = b_loss - a_loss

        if d_acc > 1e-6:
            winner = "B"
            b_wins += 1
        elif d_acc < -1e-6:
            winner = "A"
            a_wins += 1
        else:
            winner = "tie"
            ties += 1

        by_dataset[dataset].append(d_acc)

        print(
            f"{dataset:14s} {encoder:16s} {n:2d} | "
            f"{a_acc:6.3f} {b_acc:6.3f} {fmt_delta(d_acc):>7s} | "
            f"{a_loss:7.3f} {b_loss:7.3f} {fmt_delta(d_loss):>7s} | "
            f"{str(b_gates):>7s} {winner}"
        )

        out_rows.append({
            "dataset": dataset, "encoder": encoder, "n_qubits": n, "seed": seed,
            "a_test_acc": a_acc, "b_test_acc": b_acc, "delta_test_acc": d_acc,
            "a_test_loss": a_loss, "b_test_loss": b_loss, "delta_test_loss": d_loss,
            "b_best_n_gates": b_gates,
            "winner": winner,
        })

    print("-" * 110)
    n_compared = a_wins + b_wins + ties
    print(
        f"Totals: A wins {a_wins} | B wins {b_wins} | ties {ties} "
        f"(of {n_compared} comparable cells)"
    )

    print("\nMean dacc by dataset (B - A, positive = B better):")
    for ds in sorted(by_dataset):
        deltas = by_dataset[ds]
        mean = sum(deltas) / len(deltas)
        print(f"  {ds:14s}  n={len(deltas):2d}  mean dacc = {fmt_delta(mean)}")

    if out_rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {len(out_rows)} comparison rows to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
