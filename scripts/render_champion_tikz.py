"""Emit Quantikz (TikZ-based) LaTeX for the champion circuit per (stage, dataset).

Companion to scripts/render_champion_circuits.py. That one uses
qml.draw_mpl, which is a tracer -- gate angles get evaluated at a
placeholder x, so data-dependent enc_* gates show up as plain numbers
and the data dependence is lost.

This script writes LaTeX that keeps the SYMBOLIC FORM of every
data-dependent gate:

  enc_ry on q0 reading x[f]
    ->  \gate{R_y(a_{n} x_{f} + b_{n})}

  enc_rot on q0 reading x[f]
    ->  \gate{R\!\left(\substack{a^\alpha_n x_f + b^\alpha_n \\
                                  a^\beta_n  x_f + b^\beta_n  \\
                                  a^\gamma_n x_f + b^\gamma_n}\right)}

  enc_xx on (q0, q1) reading x[f_0], x[f_1]
    ->  \gate[2]{R_{XX}(a^0_{n} x_{f_0} + a^1_{n} x_{f_1} + b_{n})}

Trained-scalar ansatz gates (ry, rx, rz, rxx, ryy, rzz) show their
converged numeric value to two decimal places. cx, cz use the usual
\ctrl / \targ / \control quantikz idioms.

Stage A's ansatz is hand-built (per-qubit RY + linear CNOT chain at
depth=2); we synthesize the column list directly from that structure.
Stages B and C have external encoders (FixedAngle, ReuploadEuler,
etc.) that apply many gates before the genome runs; rather than
expand those into individual columns we render the encoder as one
labelled block so the TikZ stays readable. Stages E and F have all
gates inside the genome, so the rendering is complete.

Run:
    PYTHONPATH=. python scripts/render_champion_tikz.py

Output:
    src/Ryan_cookin/results/champion_circuits_tikz/<stage>/<dataset>.tex
    src/Ryan_cookin/results/champion_circuits_tikz/README.txt

Compile a single file with:
    pdflatex -output-directory=<dir> <file>.tex
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import torch

from src.circuits.circuit import CircuitGenome


DATASET_REGISTRY = {
    "iris":          (4,  3),
    "wine":         (13,  3),
    "seeds":         (7,  3),
    "breast_cancer":(30,  2),
}
DATASETS = list(DATASET_REGISTRY.keys())

RESULTS_DIR = "src/Ryan_cookin/results"
DEFAULT_OUT_DIR = f"{RESULTS_DIR}/champion_circuits_tikz"


# ---------- Stage spec (mirrors render_champion_circuits.py) ---------------

@dataclass
class StageSpec:
    name: str
    csv_path: str
    weights_dir: str
    fname_template: str
    kind: str   # 'A', 'B_or_C', 'E', 'F'

    def champion_row(self, dataset: str) -> dict | None:
        if not os.path.exists(self.csv_path):
            return None
        with open(self.csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset]
        if not rows:
            return None
        return sorted(rows, key=lambda r: (-float(r["test_acc"]), float(r["test_loss"])))[0]


STAGES: list[StageSpec] = [
    StageSpec("A",                f"{RESULTS_DIR}/stage_a_trained.csv",     f"{RESULTS_DIR}/weights",                       "{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt", "A"),
    StageSpec("B",                f"{RESULTS_DIR}/stage_b.csv",             f"{RESULTS_DIR}/weights_stage_b",               "{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt", "B_or_C"),
    StageSpec("B_multiseed",      f"{RESULTS_DIR}/stage_b_multiseed.csv",   f"{RESULTS_DIR}/weights_stage_b_multiseed",     "{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt", "B_or_C"),
    StageSpec("C",                f"{RESULTS_DIR}/stage_c.csv",             f"{RESULTS_DIR}/weights_stage_c",               "{dataset}_{encoder}_seed{seed}.pt",             "B_or_C"),
    StageSpec("C_multiseed",      f"{RESULTS_DIR}/stage_c_multiseed.csv",   f"{RESULTS_DIR}/weights_stage_c_multiseed",     "{dataset}_{encoder}_seed{seed}.pt",             "B_or_C"),
    StageSpec("E_v1",             f"{RESULTS_DIR}/stage_e.csv",             f"{RESULTS_DIR}/weights_stage_e",               "{dataset}_seed{seed}.pt",                       "E"),
    StageSpec("E_v2",             f"{RESULTS_DIR}/stage_e_v2.csv",          f"{RESULTS_DIR}/weights_stage_e_v2",            "{dataset}_seed{seed}.pt",                       "E"),
    StageSpec("F_v1",             f"{RESULTS_DIR}/stage_f_v1.csv",          f"{RESULTS_DIR}/weights_stage_f_v1",            "{dataset}_seed{seed}.pt",                       "F"),
    StageSpec("F_v2",             f"{RESULTS_DIR}/stage_f_v2.csv",          f"{RESULTS_DIR}/weights_stage_f_v2",            "{dataset}_seed{seed}.pt",                       "F"),
    StageSpec("F_v2_multiseed",   f"{RESULTS_DIR}/stage_f_v2_multiseed.csv",f"{RESULTS_DIR}/weights_stage_f_v2_multiseed",  "{dataset}_seed{seed}.pt",                       "F"),
    StageSpec("F_v2_big",         f"{RESULTS_DIR}/stage_f_v2_big.csv",      f"{RESULTS_DIR}/weights_stage_f_v2_big",        "{dataset}_seed{seed}.pt",                       "F"),
]


# ---------- Gate -> LaTeX label --------------------------------------------

def _label_enc_ry(inn, fi): return rf"R_y(a_{{{inn}}}\, x_{{{fi}}} + b_{{{inn}}})"
def _label_enc_rx(inn, fi): return rf"R_x(a_{{{inn}}}\, x_{{{fi}}} + b_{{{inn}}})"
def _label_enc_rz(inn, fi): return rf"R_z(a_{{{inn}}}\, x_{{{fi}}} + b_{{{inn}}})"

def _label_enc_rot(inn, fi):
    # Three Euler angles, each a learned linear function of x[fi].
    return (
        rf"R\!\left(\substack{{"
        rf"a^{{\alpha}}_{{{inn}}} x_{{{fi}}} + b^{{\alpha}}_{{{inn}}} \\ "
        rf"a^{{\beta}}_{{{inn}}}  x_{{{fi}}} + b^{{\beta}}_{{{inn}}}  \\ "
        rf"a^{{\gamma}}_{{{inn}}} x_{{{fi}}} + b^{{\gamma}}_{{{inn}}}"
        rf"}}\right)"
    )

def _label_enc_2q(axis, inn, fi0, fi1):
    return (
        rf"R_{{{axis}}}\!\left("
        rf"a^{{0}}_{{{inn}}} x_{{{fi0}}} + "
        rf"a^{{1}}_{{{inn}}} x_{{{fi1}}} + "
        rf"b_{{{inn}}}\right)"
    )

def _label_constant_1q(axis, val):
    return rf"R_{axis}({val:+.2f})"

def _label_constant_2q(axis, val):
    return rf"R_{{{axis}}}({val:+.2f})"


def _wire_idx(qubit_tuple) -> int:
    return qubit_tuple[1]


def gate_to_column(gate, n_qubits: int, n_features: int) -> list[str]:
    """Return a list of length n_qubits with one cell per wire for this gate.

    Cells that are not affected by the gate are ``\qw``. Cells covered by a
    multi-wire gate block are empty strings (Quantikz reads those as
    "already covered by the gate to the north").
    """
    method = gate.method_name
    inn = gate.innovation_number
    col = [r"\qw"] * n_qubits

    if method == "cx":
        c, t = _wire_idx(gate.qubits[0]), _wire_idx(gate.qubits[1])
        col[c] = rf"\ctrl{{{t - c}}}"
        col[t] = r"\targ{}"
        return col

    if method == "cz":
        a, b = _wire_idx(gate.qubits[0]), _wire_idx(gate.qubits[1])
        if a > b:
            a, b = b, a
        col[a] = rf"\ctrl{{{b - a}}}"
        col[b] = r"\control{}"
        return col

    if method in ("enc_ry", "enc_rx", "enc_rz"):
        w = _wire_idx(gate.qubits[0])
        fi = w % n_features
        labels = {"enc_ry": _label_enc_ry, "enc_rx": _label_enc_rx, "enc_rz": _label_enc_rz}
        col[w] = rf"\gate{{{labels[method](inn, fi)}}}"
        return col

    if method == "enc_rot":
        w = _wire_idx(gate.qubits[0])
        fi = w % n_features
        col[w] = rf"\gate{{{_label_enc_rot(inn, fi)}}}"
        return col

    if method in ("enc_xx", "enc_yy", "enc_zz"):
        w0 = _wire_idx(gate.qubits[0])
        w1 = _wire_idx(gate.qubits[1])
        lo, hi = sorted((w0, w1))
        axis = method[-2:].upper()
        fi0, fi1 = w0 % n_features, w1 % n_features
        span = hi - lo + 1
        col[lo] = rf"\gate[{span}]{{{_label_enc_2q(axis, inn, fi0, fi1)}}}"
        for w in range(lo + 1, hi + 1):
            col[w] = ""
        return col

    if method in ("ry", "rx", "rz"):
        w = _wire_idx(gate.qubits[0])
        val = float(next(iter(gate.parameters.values())))
        col[w] = rf"\gate{{{_label_constant_1q(method[-1], val)}}}"
        return col

    if method in ("rxx", "ryy", "rzz"):
        w0 = _wire_idx(gate.qubits[0])
        w1 = _wire_idx(gate.qubits[1])
        lo, hi = sorted((w0, w1))
        axis = method.upper()[-2:]
        val = float(next(iter(gate.parameters.values())))
        span = hi - lo + 1
        col[lo] = rf"\gate[{span}]{{{_label_constant_2q(axis, val)}}}"
        for w in range(lo + 1, hi + 1):
            col[w] = ""
        return col

    # Fallback for any gate we haven't explicitly handled: a labelled
    # single-qubit-style box on the first wire it touches.
    w = _wire_idx(gate.qubits[0])
    col[w] = rf"\gate{{\mathrm{{{method.replace('_', r'\_')}}}}}"
    return col


def genome_to_columns(genome, n_features: int) -> tuple[list[list[str]], int]:
    """Sort enabled gates by depth and turn each into a column."""
    genome.sort_gates()
    n_qubits = len(genome.qubits)
    gates = [g for g in genome.gates if g.enabled]
    cols = [gate_to_column(g, n_qubits, n_features) for g in gates]
    return cols, n_qubits


def columns_to_body(columns: list[list[str]], n_qubits: int) -> str:
    """Lay columns out into rows; one row per wire, columns separated by &."""
    rows = []
    for w in range(n_qubits):
        cells = [r"\lstick{$q_{" + str(w) + r"}$}"]
        for col in columns:
            cells.append(col[w])
        cells.append(r"\qw")
        rows.append(" & ".join(cells))
    return " \\\\\n".join(rows)


# ---------- Stage A: hand-built RY-CNOT chain --------------------------------

def stage_a_columns(n_qubits: int, ansatz_depth: int) -> list[list[str]]:
    """Stage A's ansatz is fixed: per-qubit RY then a linear CNOT chain,
    repeated `ansatz_depth` times. The encoder is rendered separately as an
    annotation rather than expanded into columns."""
    cols: list[list[str]] = []
    for _ in range(ansatz_depth):
        for q in range(n_qubits):
            col = [r"\qw"] * n_qubits
            col[q] = r"\gate{R_y(\phi)}"
            cols.append(col)
        for q in range(n_qubits - 1):
            col = [r"\qw"] * n_qubits
            col[q] = rf"\ctrl{{1}}"
            col[q + 1] = r"\targ{}"
            cols.append(col)
    return cols


# ---------- Document wrappers ---------------------------------------------

PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage{quantikz}
\begin{document}
\begin{quantikz}[row sep={1cm,between origins}, column sep=0.6cm]
"""

POSTAMBLE = r"""
\end{quantikz}
\end{document}
"""


def wrap_standalone(body: str, header_comment: str = "") -> str:
    head = ""
    if header_comment:
        head = "% " + header_comment.replace("\n", "\n% ") + "\n"
    return head + PREAMBLE + body + POSTAMBLE


# ---------- Per-stage rendering --------------------------------------------

def render_stage_A(meta, n_features, n_classes) -> tuple[str, str]:
    encoder_name = meta["encoder"]
    n_qubits = int(meta["n_qubits"])
    ansatz_depth = int(meta.get("ansatz_depth", 2))
    cols = stage_a_columns(n_qubits, ansatz_depth)
    body = columns_to_body(cols, n_qubits)
    note = (
        f"Encoder '{encoder_name}' (D={n_features}, N={n_qubits}) applies first; "
        "not expanded here. Ansatz: per-qubit R_y + linear CNOT chain, "
        f"depth={ansatz_depth}. test_acc={meta['test_acc']:.3f}."
    )
    return body, note


def render_stage_B_or_C(meta, n_features, n_classes) -> tuple[str, str]:
    encoder_name = meta["encoder"]
    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    cols, n_qubits = genome_to_columns(genome, n_features)
    body = columns_to_body(cols, n_qubits)
    note = (
        f"Encoder '{encoder_name}' (D={n_features}, N={n_qubits}) applies first; "
        "not expanded here. What follows is the evolved ansatz portion of "
        f"the qnode. Enabled gates: {sum(1 for g in genome.gates if g.enabled)}, "
        f"test_acc={meta['test_acc']:.3f}."
    )
    return body, note


def render_stage_E(meta, n_features, n_classes) -> tuple[str, str]:
    import src.Ryan_cookin.stage_e as _s
    _s._register_encoder_gate_specs()
    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    cols, n_qubits = genome_to_columns(genome, n_features)
    body = columns_to_body(cols, n_qubits)
    note = (
        f"D={n_features} features, N={n_qubits} qubits, K={n_classes} classes. "
        f"enc_gates={meta.get('n_enc_gates','?')}, "
        f"ansatz_gates={meta.get('n_ansatz_gates','?')}, "
        f"test_acc={meta['test_acc']:.3f}. "
        "enc_ry/rx/rz angles are learned linear functions of one feature "
        "(slope a, bias b per gate)."
    )
    return body, note


def render_stage_F(meta, n_features, n_classes) -> tuple[str, str]:
    import src.Ryan_cookin.stage_f as _s
    _s._register_stage_f_specs()
    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    cols, n_qubits = genome_to_columns(genome, n_features)
    body = columns_to_body(cols, n_qubits)
    note = (
        f"D={n_features} features, N={n_qubits} qubits, K={n_classes} classes. "
        f"enc_gates={meta.get('n_enc_gates','?')}, "
        f"ansatz_gates={meta.get('n_ansatz_gates','?')}, "
        f"test_acc={meta['test_acc']:.3f}. "
        "enc_rot is a universal 1-qubit data-dependent gate (3 Euler angles, "
        "each its own affine function of one feature). enc_xx/yy/zz are "
        "2-qubit Ising rotations whose angle is a linear combination of two "
        "feature components."
    )
    return body, note


KIND_DISPATCH = {
    "A": render_stage_A,
    "B_or_C": render_stage_B_or_C,
    "E": render_stage_E,
    "F": render_stage_F,
}


def _build_filename(stage: StageSpec, row: dict) -> str:
    n_qubits = row.get("n_qubits") or row.get("best_n_qubits") or "?"
    return stage.fname_template.format(
        dataset=row["dataset"],
        encoder=row.get("encoder", ""),
        n_qubits=n_qubits,
        seed=row.get("seed", 0),
    )


def render_one(stage: StageSpec, row: dict, out_dir: str) -> bool:
    dataset = row["dataset"]
    n_features, n_classes = DATASET_REGISTRY[dataset]
    fname = _build_filename(stage, row)
    weights_path = os.path.join(stage.weights_dir, fname)
    if not os.path.exists(weights_path):
        print(f"  MISSING  stage {stage.name:18s} {dataset:14s}  -> {weights_path}")
        return False

    meta = torch.load(weights_path, map_location="cpu", weights_only=False)
    body, note = KIND_DISPATCH[stage.kind](meta, n_features, n_classes)
    header = (
        f"stage {stage.name}, dataset {dataset}, seed {row.get('seed', 0)}\n"
        f"{note}\n"
        "Compile with: pdflatex <this file>"
    )
    tex = wrap_standalone(body, header_comment=header)

    stage_dir = os.path.join(out_dir, stage.name)
    os.makedirs(stage_dir, exist_ok=True)
    out_path = os.path.join(stage_dir, f"{dataset}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  wrote stage {stage.name:18s} {dataset:14s}  -> {out_path}")
    return True


README = """Champion circuits as Quantikz/LaTeX

Each .tex file is a `standalone` document with a single quantikz
diagram. Encoder gates (enc_*) render symbolically, e.g.

  enc_ry on q0 reading x[0]
    -> R_y(a_n x_0 + b_n)

  enc_rot
    -> Rot(a^alpha_n x_f + b^alpha_n,
           a^beta_n  x_f + b^beta_n,
           a^gamma_n x_f + b^gamma_n)

  enc_xx
    -> R_{XX}(a^0_n x_{f_0} + a^1_n x_{f_1} + b_n)

so the data dependence is explicit rather than evaluated to a number
at some placeholder x (which is what the matplotlib PNGs in
champion_circuits/ show).

Trained-scalar ansatz gates (ry, rx, rz, rxx, ryy, rzz) DO show their
converged numeric value, since for those it's not a function of x.

Stage A renders the hand-built RY-CNOT chain with R_y(phi) placeholders.
Stages B and C render only the genome (ansatz) part; the external
encoder is summarised in the header comment because expanding its
gates inline would blow the diagram up to dozens of columns.

Compile with:
    pdflatex -output-directory <stage_dir> <stage_dir>/<dataset>.tex

You'll need the `quantikz` LaTeX package installed (it ships with
recent TeX Live distributions; otherwise `tlmgr install quantikz`).
"""


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"out_dir: {args.out_dir}\n")

    n_written = 0
    n_missing = 0
    for stage in STAGES:
        for dataset in DATASETS:
            row = stage.champion_row(dataset)
            if row is None:
                print(f"  NO ROW   stage {stage.name:18s} {dataset:14s}")
                n_missing += 1
                continue
            ok = render_one(stage, row, args.out_dir)
            if ok:
                n_written += 1
            else:
                n_missing += 1

    readme_path = os.path.join(args.out_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README)
    print(f"\nDone. {n_written} .tex files in {args.out_dir}/  "
          f"({n_missing} missing).  README at {readme_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
