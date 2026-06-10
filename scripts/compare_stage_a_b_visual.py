"""Produce a markdown report + a PNG heatmap comparing Stage A and Stage B.

Reads:
    src/Ryan_cookin/results/stage_a_trained.csv
    src/Ryan_cookin/results/stage_b.csv
    src/Ryan_cookin/results/stage_ab_compare.csv

Writes:
    src/Ryan_cookin/results/stage_ab_compare.md      per-dataset tables
    src/Ryan_cookin/results/stage_ab_compare.png     heatmap grid
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["iris", "wine", "seeds", "breast_cancer"]
ENCODERS = ["fixed_angle", "fixed_amplitude", "fixed_basis", "learned"]
N_VALUES = [4, 6, 8, 10]


def load_csv(path: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["dataset"], row["encoder"],
                int(row["n_qubits"]), int(row["seed"]),
            )
            out[key] = row
    return out


def _fmt_acc(v: float) -> str:
    return f"{v:.3f}"


def _fmt_delta(d: float) -> str:
    if abs(d) < 1e-6:
        return " 0.000"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.3f}"


def _winner_marker(winner: str) -> str:
    return {"A": "A", "B": "B*", "tie": "="}.get(winner, "?")


def build_markdown(compare: dict[tuple, dict]) -> str:
    out = []
    out.append("# Stage A vs Stage B comparison\n")
    out.append(
        "Cells show `A_acc / B_acc (Δ)` where Δ = B − A. "
        "Bold means Stage B wins; italic means Stage A wins; plain means tie "
        "(|Δ| < 1e-6). `gates` is the enabled gate count in Stage B's best "
        "evolved genome — for reference, Stage A's hand-built ansatz always "
        "has `2N` rotation gates + `2(N-1)` CNOTs.\n"
    )

    # Per-dataset table: rows = encoder, columns = N
    for ds in DATASETS:
        out.append(f"\n## {ds}\n")
        out.append(
            "| encoder | "
            + " | ".join(f"N={n}" for n in N_VALUES)
            + " |"
        )
        out.append(
            "|---|" + "|".join(["---"] * len(N_VALUES)) + "|"
        )
        for enc in ENCODERS:
            cells = [f"`{enc}`"]
            for n in N_VALUES:
                key = (ds, enc, n, 0)
                row = compare.get(key)
                if row is None:
                    cells.append("—")
                    continue
                a = float(row["a_test_acc"])
                b = float(row["b_test_acc"])
                d = float(row["delta_test_acc"])
                gates = row["b_best_n_gates"]
                winner = row["winner"]
                base = f"{_fmt_acc(a)} / {_fmt_acc(b)} ({_fmt_delta(d)}) g={gates}"
                if winner == "B":
                    base = f"**{base}**"
                elif winner == "A":
                    base = f"_{base}_"
                cells.append(base)
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    # Summary
    out.append("\n## Summary\n")
    wins = {"A": 0, "B": 0, "tie": 0}
    by_dataset_delta: dict[str, list[float]] = defaultdict(list)
    by_encoder_delta: dict[str, list[float]] = defaultdict(list)
    by_encoder_wins: dict[str, dict[str, int]] = defaultdict(
        lambda: {"A": 0, "B": 0, "tie": 0}
    )
    for row in compare.values():
        wins[row["winner"]] += 1
        by_dataset_delta[row["dataset"]].append(float(row["delta_test_acc"]))
        by_encoder_delta[row["encoder"]].append(float(row["delta_test_acc"]))
        by_encoder_wins[row["encoder"]][row["winner"]] += 1

    total = sum(wins.values())
    out.append(
        f"- Win counts (of {total} cells): "
        f"**B wins {wins['B']}**, A wins {wins['A']}, ties {wins['tie']}"
    )

    out.append("\n### Mean Δacc by dataset (B − A, positive = B better)\n")
    out.append("| dataset | n | mean Δacc |")
    out.append("|---|---|---|")
    for ds in DATASETS:
        deltas = by_dataset_delta.get(ds, [])
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        out.append(f"| {ds} | {len(deltas)} | {_fmt_delta(mean)} |")

    out.append("\n### Mean Δacc by encoder\n")
    out.append("| encoder | n | mean Δacc |")
    out.append("|---|---|---|")
    for enc in ENCODERS:
        deltas = by_encoder_delta.get(enc, [])
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        out.append(f"| {enc} | {len(deltas)} | {_fmt_delta(mean)} |")

    out.append("\n### Win counts by encoder (B vs A)\n")
    out.append(
        "How often Stage B's evolved ansatz beats Stage A's fixed RY-CNOT "
        "chain, holding the encoder fixed. The `learned` row answers "
        "\"does learn + evolution beat learn + standard ansatz?\".\n"
    )
    out.append("| encoder | B wins | A wins | ties | n |")
    out.append("|---|---|---|---|---|")
    for enc in ENCODERS:
        w = by_encoder_wins.get(enc)
        if not w:
            continue
        n = w["A"] + w["B"] + w["tie"]
        out.append(
            f"| {enc} | **{w['B']}** | {w['A']} | {w['tie']} | {n} |"
        )

    # Best per dataset
    out.append("\n### Best cell per dataset\n")
    out.append(
        "| dataset | Stage A best (cell, acc) | Stage B best (cell, acc, gates) |"
    )
    out.append("|---|---|---|")
    best_a: dict[str, tuple] = {}
    best_b: dict[str, tuple] = {}
    for row in compare.values():
        ds = row["dataset"]
        a_acc = float(row["a_test_acc"])
        b_acc = float(row["b_test_acc"])
        if ds not in best_a or a_acc > best_a[ds][0]:
            best_a[ds] = (a_acc, row["encoder"], int(row["n_qubits"]))
        if ds not in best_b or b_acc > best_b[ds][0]:
            best_b[ds] = (b_acc, row["encoder"], int(row["n_qubits"]),
                          row["b_best_n_gates"])
    for ds in DATASETS:
        a = best_a.get(ds)
        b = best_b.get(ds)
        if a is None or b is None:
            continue
        out.append(
            f"| {ds} | "
            f"`{a[1]}` N={a[2]}, acc={a[0]:.3f} | "
            f"`{b[1]}` N={b[2]}, acc={b[0]:.3f}, gates={b[3]} |"
        )

    return "\n".join(out) + "\n"


def build_heatmap(compare: dict[tuple, dict], out_path: str) -> None:
    """Grid: 4 columns (datasets) x 3 rows (A_acc, B_acc, delta_acc).

    Each subplot heatmap rows = encoders, columns = N values.
    """
    # 4 encoders now, so make the rows a bit taller.
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))

    # Build 3D arrays: axis 0 = datasets, axis 1 = encoders, axis 2 = N
    n_ds = len(DATASETS)
    n_enc = len(ENCODERS)
    n_N = len(N_VALUES)

    a_acc = np.full((n_ds, n_enc, n_N), np.nan)
    b_acc = np.full((n_ds, n_enc, n_N), np.nan)
    d_acc = np.full((n_ds, n_enc, n_N), np.nan)
    b_gates = np.full((n_ds, n_enc, n_N), np.nan)

    for row in compare.values():
        di = DATASETS.index(row["dataset"])
        ei = ENCODERS.index(row["encoder"])
        ni = N_VALUES.index(int(row["n_qubits"]))
        a_acc[di, ei, ni] = float(row["a_test_acc"])
        b_acc[di, ei, ni] = float(row["b_test_acc"])
        d_acc[di, ei, ni] = float(row["delta_test_acc"])
        try:
            b_gates[di, ei, ni] = float(row["b_best_n_gates"])
        except (ValueError, TypeError):
            pass

    row_labels = ["Stage A test_acc", "Stage B test_acc", "Δacc = B − A"]
    data_per_row = [a_acc, b_acc, d_acc]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    # Symmetric color scale for delta around 0; viridis fixed 0..1 for acc.
    vmins = [0.0, 0.0, None]
    vmaxs = [1.0, 1.0, None]

    for ri in range(3):
        for di, ds in enumerate(DATASETS):
            ax = axes[ri][di]
            data = data_per_row[ri][di]  # shape (n_enc, n_N)
            if cmaps[ri] == "RdBu_r":
                vmax = max(0.05, float(np.nanmax(np.abs(d_acc))))
                im = ax.imshow(data, cmap=cmaps[ri], vmin=-vmax, vmax=vmax,
                               aspect="auto")
            else:
                im = ax.imshow(data, cmap=cmaps[ri],
                               vmin=vmins[ri], vmax=vmaxs[ri], aspect="auto")

            # Annotate cells
            for ei in range(n_enc):
                for ni in range(n_N):
                    v = data[ei, ni]
                    if np.isnan(v):
                        continue
                    text = (f"{v:+.2f}" if ri == 2 else f"{v:.2f}")
                    # Show gate count on Stage B row for context
                    if ri == 1 and not np.isnan(b_gates[di, ei, ni]):
                        text = f"{v:.2f}\ng={int(b_gates[di, ei, ni])}"
                    color = "white" if abs(v) > 0.5 and ri != 2 else "black"
                    if ri == 2:
                        # white text on dark blues/reds, black otherwise
                        color = "white" if abs(v) > 0.15 else "black"
                    ax.text(ni, ei, text, ha="center", va="center",
                            fontsize=8, color=color)

            ax.set_xticks(range(n_N))
            ax.set_xticklabels([f"N={n}" for n in N_VALUES], fontsize=8)
            ax.set_yticks(range(n_enc))
            ax.set_yticklabels(ENCODERS, fontsize=8)
            if ri == 0:
                ax.set_title(ds, fontsize=11, fontweight="bold")
            if di == 0:
                ax.set_ylabel(row_labels[ri], fontsize=10)

            # Lightweight colorbar on the rightmost column of each row
            if di == n_ds - 1:
                fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)

    fig.suptitle(
        "Stage A (fixed RY-CNOT-chain ansatz) vs Stage B (evolved ansatz)\n"
        "Columns = datasets, rows in each subplot = encoders, "
        "x-axis = n_qubits. Stage B cells also show gate count `g=`.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--compare",
                   default="src/Ryan_cookin/results/stage_ab_compare.csv")
    p.add_argument("--out_md",
                   default="src/Ryan_cookin/results/stage_ab_compare.md")
    p.add_argument("--out_png",
                   default="src/Ryan_cookin/results/stage_ab_compare.png")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    compare = load_csv(args.compare)
    if not compare:
        print(f"No data in {args.compare}", file=sys.stderr)
        return 1

    md = build_markdown(compare)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {args.out_md}")

    build_heatmap(compare, args.out_png)
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
