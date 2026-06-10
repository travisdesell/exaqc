"""Comprehensive comparison across Stages A, B, C, E.

Produces a single markdown report + matplotlib figure that summarizes
the full research progression:

    Stage A:  trained encoder + FIXED ansatz, N swept manually
    Stage B:  trained encoder + EVOLVED ansatz, N swept manually
    Stage C:  trained encoder + EVOLVED ansatz, EVOLVED N
    Stage E:  EVOLVED encoder (feature-dependent gates) + EVOLVED ansatz + EVOLVED N

Per-encoder Stage A/B/C breakdown answers: "for a given encoder family,
does evolving more parts of the pipeline help?"

Stage E sits apart in the breakdown because it has no encoder dimension
— the encoder is part of the genome.

Stage D's two new encoders (linear_proj, reupload_euler) appear inside
Stage A/B/C rows alongside the original four.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


DATASETS = ["iris", "wine", "seeds", "breast_cancer"]
ENCODERS = [
    "fixed_basis", "fixed_angle", "fixed_amplitude",
    "learned", "linear_proj", "reupload_euler",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def best_per_group(rows: list[dict], group_key, score_key="test_acc",
                   higher_better=True) -> dict:
    out: dict = {}
    for row in rows:
        gk = group_key(row)
        score = float(row[score_key])
        cur = out.get(gk)
        if cur is None or (
            (higher_better and score > float(cur[score_key])) or
            (not higher_better and score < float(cur[score_key]))
        ):
            out[gk] = row
    return out


def _fmt_acc(v: float) -> str:
    return f"{v:.3f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage_a", default="src/Ryan_cookin/results/stage_a_trained.csv")
    p.add_argument("--stage_b", default="src/Ryan_cookin/results/stage_b.csv")
    p.add_argument("--stage_c", default="src/Ryan_cookin/results/stage_c.csv")
    p.add_argument("--stage_e", default="src/Ryan_cookin/results/stage_e.csv")
    p.add_argument("--out_md", default="src/Ryan_cookin/results/all_stages_compare.md")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    a_rows = load_csv(args.stage_a)
    b_rows = load_csv(args.stage_b)
    c_rows = load_csv(args.stage_c)
    e_rows = load_csv(args.stage_e)

    print(f"Loaded: A={len(a_rows)} B={len(b_rows)} C={len(c_rows)} E={len(e_rows)}")

    md = []
    md.append("# All-stages comparison (A vs B vs C vs E)\n")
    md.append(
        "Progression of what gets *evolved* (instead of fixed/swept) "
        "across stages:\n"
    )
    md.append("| stage | encoder | ansatz structure | N (qubits) |")
    md.append("|---|---|---|---|")
    md.append("| A | trained params | fixed (RY-CNOT chain) | swept |")
    md.append("| B | trained params | evolved | swept |")
    md.append("| C | trained params | evolved | **evolved** |")
    md.append("| E | **evolved** (feature-dep gates) | evolved | evolved |")
    md.append("")

    # Best per dataset across every stage
    md.append("## Best test_acc per dataset, across all stages\n")
    md.append(
        "| dataset | Stage A | Stage B | Stage C | Stage E |"
    )
    md.append("|---|---|---|---|---|")

    for ds in DATASETS:
        a_in = [r for r in a_rows if r["dataset"] == ds]
        b_in = [r for r in b_rows if r["dataset"] == ds]
        c_in = [r for r in c_rows if r["dataset"] == ds]
        e_in = [r for r in e_rows if r["dataset"] == ds]

        def fmt_a(r):
            if r is None:
                return "—"
            return (f"**{_fmt_acc(float(r['test_acc']))}** "
                    f"({r['encoder']}, N={r['n_qubits']})")

        def fmt_b(r):
            if r is None:
                return "—"
            return (f"**{_fmt_acc(float(r['test_acc']))}** "
                    f"({r['encoder']}, N={r['n_qubits']}, "
                    f"g={r['best_n_gates']})")

        def fmt_c(r):
            if r is None:
                return "—"
            return (f"**{_fmt_acc(float(r['test_acc']))}** "
                    f"({r['encoder']}, N={r['best_n_qubits']}, "
                    f"g={r['best_n_gates']})")

        def fmt_e(r):
            if r is None:
                return "—"
            return (f"**{_fmt_acc(float(r['test_acc']))}** "
                    f"(N={r['best_n_qubits']}, "
                    f"enc={r['n_enc_gates']}, ans={r['n_ansatz_gates']})")

        best_a = max(a_in, key=lambda r: float(r["test_acc"])) if a_in else None
        best_b = max(b_in, key=lambda r: float(r["test_acc"])) if b_in else None
        best_c = max(c_in, key=lambda r: float(r["test_acc"])) if c_in else None
        best_e = max(e_in, key=lambda r: float(r["test_acc"])) if e_in else None

        md.append(
            f"| {ds} | {fmt_a(best_a)} | {fmt_b(best_b)} | "
            f"{fmt_c(best_c)} | {fmt_e(best_e)} |"
        )
    md.append("")

    # Per-encoder summary: mean test_acc across datasets/N for each stage
    md.append("## Mean test_acc by encoder, by stage (averaged over datasets and N)\n")
    md.append("| encoder | Stage A (mean acc) | Stage B (mean acc) | Stage C (mean acc) |")
    md.append("|---|---|---|---|")

    def mean_acc_by_encoder(rows: list[dict], enc: str) -> float | None:
        accs = [float(r["test_acc"]) for r in rows if r["encoder"] == enc]
        if not accs:
            return None
        return sum(accs) / len(accs)

    for enc in ENCODERS:
        a_m = mean_acc_by_encoder(a_rows, enc)
        b_m = mean_acc_by_encoder(b_rows, enc)
        c_m = mean_acc_by_encoder(c_rows, enc)
        cells = [enc]
        for m in (a_m, b_m, c_m):
            cells.append("—" if m is None else f"{m:.3f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Stage E summary (no encoder breakdown)
    md.append("## Stage E summary (encoder is part of the genome)\n")
    md.append("| dataset | test_acc | test_loss | best N | enc gates | ansatz gates |")
    md.append("|---|---|---|---|---|---|")
    for ds in DATASETS:
        e_in = [r for r in e_rows if r["dataset"] == ds]
        if not e_in:
            md.append(f"| {ds} | — | — | — | — | — |")
            continue
        best_e = max(e_in, key=lambda r: float(r["test_acc"]))
        md.append(
            f"| {ds} | {_fmt_acc(float(best_e['test_acc']))} | "
            f"{float(best_e['test_loss']):.4f} | "
            f"{best_e['best_n_qubits']} | {best_e['n_enc_gates']} | "
            f"{best_e['n_ansatz_gates']} |"
        )
    md.append("")

    # Evolved N statistics for Stage C
    if c_rows:
        md.append("## Stage C: how N evolved (best genomes)\n")
        md.append(
            "Reveals whether evolution preferred bigger or smaller registers. "
            "Stage C starts at N=4; values below 4 mean evolution shrunk the "
            "register, above 4 mean it grew.\n"
        )
        md.append("| dataset/encoder | initial N | best N | delta |")
        md.append("|---|---|---|---|")
        for r in c_rows:
            ini = int(r["initial_n_qubits"])
            bst = int(r["best_n_qubits"])
            md.append(
                f"| {r['dataset']}/{r['encoder']} | {ini} | "
                f"{bst} | {bst - ini:+d} |"
            )
        md.append("")

        # Histogram of best_n_qubits across all Stage C cells
        from collections import Counter
        hist = Counter(int(r["best_n_qubits"]) for r in c_rows)
        md.append("### Stage C evolved-N histogram (across all cells)\n")
        md.append("| N | count |")
        md.append("|---|---|")
        for n in sorted(hist.keys()):
            md.append(f"| {n} | {hist[n]} |")
        md.append("")

    # Stage E evolved-N (one row per dataset, but worth showing)
    if e_rows:
        md.append("## Stage E: evolved register sizes\n")
        md.append("| dataset | best N | enc gates | ansatz gates |")
        md.append("|---|---|---|---|")
        for r in e_rows:
            md.append(
                f"| {r['dataset']} | {r['best_n_qubits']} | "
                f"{r['n_enc_gates']} | {r['n_ansatz_gates']} |"
            )
        md.append("")

    # Champion per dataset: best test_acc across all stages
    md.append("## Champion per dataset (best test_acc across A, B, C, E)\n")
    md.append("| dataset | stage | encoder | N | test_acc | test_loss | gates |")
    md.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        contenders = []
        for r in a_rows:
            if r["dataset"] == ds:
                contenders.append(("A", r["encoder"], r.get("n_qubits", "—"),
                                   float(r["test_acc"]), float(r["test_loss"]),
                                   "—"))
        for r in b_rows:
            if r["dataset"] == ds:
                contenders.append(("B", r["encoder"], r.get("n_qubits", "—"),
                                   float(r["test_acc"]), float(r["test_loss"]),
                                   r.get("best_n_gates", "—")))
        for r in c_rows:
            if r["dataset"] == ds:
                contenders.append(("C", r["encoder"], r.get("best_n_qubits", "—"),
                                   float(r["test_acc"]), float(r["test_loss"]),
                                   r.get("best_n_gates", "—")))
        for r in e_rows:
            if r["dataset"] == ds:
                contenders.append(("E", "—evolved—", r.get("best_n_qubits", "—"),
                                   float(r["test_acc"]), float(r["test_loss"]),
                                   f"{r.get('n_enc_gates','?')}+{r.get('n_ansatz_gates','?')}"))
        if not contenders:
            md.append(f"| {ds} | — | — | — | — | — | — |")
            continue
        best = max(contenders, key=lambda c: c[3])
        stage, enc, N, acc, loss, g = best
        md.append(f"| {ds} | {stage} | {enc} | {N} | {acc:.3f} | {loss:.4f} | {g} |")
    md.append("")

    # Stage-level summary: mean / best test_acc per stage
    md.append("## Stage-level summary (mean / best test_acc across all cells)\n")
    md.append("| stage | n cells | mean acc | best acc | worst acc |")
    md.append("|---|---|---|---|---|")
    for name, rows in (("A", a_rows), ("B", b_rows), ("C", c_rows), ("E", e_rows)):
        if not rows:
            md.append(f"| {name} | 0 | — | — | — |")
            continue
        accs = [float(r["test_acc"]) for r in rows]
        md.append(
            f"| {name} | {len(rows)} | {sum(accs)/len(accs):.3f} | "
            f"{max(accs):.3f} | {min(accs):.3f} |"
        )
    md.append("")

    # A → B → C progression on the `learned` encoder (best per dataset)
    md.append("## A → B → C progression on `learned` encoder (best cell per dataset)\n")
    md.append(
        "Tracks the same encoder family across increasing evolution scope: "
        "Stage A (fixed ansatz, N swept) → Stage B (evolved ansatz, N swept) "
        "→ Stage C (evolved ansatz AND N).\n"
    )
    md.append("| dataset | A best (N) | B best (N) | C best (evolved N) | A→B Δ | B→C Δ |")
    md.append("|---|---|---|---|---|---|")
    for ds in DATASETS:
        a_l = [r for r in a_rows if r["dataset"] == ds and r["encoder"] == "learned"]
        b_l = [r for r in b_rows if r["dataset"] == ds and r["encoder"] == "learned"]
        c_l = [r for r in c_rows if r["dataset"] == ds and r["encoder"] == "learned"]
        a_best = max((float(r["test_acc"]) for r in a_l), default=None)
        b_best = max((float(r["test_acc"]) for r in b_l), default=None)
        c_best = max((float(r["test_acc"]) for r in c_l), default=None)
        a_N = max(a_l, key=lambda r: float(r["test_acc"]))["n_qubits"] if a_l else "—"
        b_N = max(b_l, key=lambda r: float(r["test_acc"]))["n_qubits"] if b_l else "—"
        c_N = max(c_l, key=lambda r: float(r["test_acc"]))["best_n_qubits"] if c_l else "—"
        ab = f"{b_best - a_best:+.3f}" if (a_best is not None and b_best is not None) else "—"
        bc = f"{c_best - b_best:+.3f}" if (b_best is not None and c_best is not None) else "—"
        md.append(
            f"| {ds} | {_fmt_acc(a_best) if a_best is not None else '—'} (N={a_N}) | "
            f"{_fmt_acc(b_best) if b_best is not None else '—'} (N={b_N}) | "
            f"{_fmt_acc(c_best) if c_best is not None else '—'} (N={c_N}) | "
            f"{ab} | {bc} |"
        )
    md.append("")

    # Stage E vs the best of A/B/C per dataset
    if e_rows:
        md.append("## Stage E vs best of staged (A/B/C) per dataset\n")
        md.append(
            "Does fully co-evolving the encoder structure (Stage E) beat "
            "the best staged formulation? Δ > 0 = E wins.\n"
        )
        md.append("| dataset | best of A/B/C | Stage E | Δ (E − best) |")
        md.append("|---|---|---|---|")
        for ds in DATASETS:
            best_staged = None
            for r in a_rows + b_rows + c_rows:
                if r["dataset"] == ds:
                    acc = float(r["test_acc"])
                    if best_staged is None or acc > best_staged:
                        best_staged = acc
            e_in = [r for r in e_rows if r["dataset"] == ds]
            if not e_in:
                md.append(f"| {ds} | {_fmt_acc(best_staged) if best_staged else '—'} | — | — |")
                continue
            e_acc = max(float(r["test_acc"]) for r in e_in)
            delta = (e_acc - best_staged) if best_staged is not None else None
            md.append(
                f"| {ds} | "
                f"{_fmt_acc(best_staged) if best_staged is not None else '—'} | "
                f"{_fmt_acc(e_acc)} | "
                f"{('—' if delta is None else f'{delta:+.3f}')} |"
            )
        md.append("")

    out_path = args.out_md
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
