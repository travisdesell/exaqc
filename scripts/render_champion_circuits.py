"""Render the champion circuit (best test_acc) per (stage, dataset).

For every stage that has a results CSV + per-cell .pt weights:
  1. Find the highest-test_acc row per dataset (tie-break by lowest
     test_loss).
  2. Load the .pt for that row.
  3. Rebuild the qnode using the same logic the stage driver used,
     applied with the trained parameter values.
  4. Save a PNG at champion_circuits/{stage}_{dataset}.png with a
     title strip describing the cell.

Run:
    PYTHONPATH=. python scripts/render_champion_circuits.py

Output:
    src/Ryan_cookin/results/champion_circuits/<stage>_<dataset>.png
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

from src.circuits.circuit import CircuitGenome
from src.Ryan_cookin.encoders import make_encoder


DATASET_REGISTRY = {
    "iris":          (4,  3),
    "wine":         (13,  3),
    "seeds":         (7,  3),
    "breast_cancer":(30,  2),
}
DATASETS = list(DATASET_REGISTRY.keys())

RESULTS_DIR = "src/Ryan_cookin/results"
DEFAULT_OUT_DIR = f"{RESULTS_DIR}/champion_circuits"


# ---------- Stage-specific qnode builders ---------------------------------

def _output_indexes(n_qubits: int, n_classes: int) -> list[int]:
    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    return list(range(n_output))


def _params_dict_from_genome(genome: CircuitGenome) -> dict:
    """Pull trained parameter values out of the genome into a flat dict."""
    out = {}
    for gate in genome.gates:
        if not gate.enabled:
            continue
        for name, value in gate.parameters.items():
            out[f"{gate.innovation_number}:{name}"] = torch.tensor(
                float(value), dtype=torch.float64
            )
    return out


def build_qnode_stage_a(meta: dict, n_features: int, n_classes: int):
    """Stage A: encoder + hand-built RY-then-CNOT chain (depth=2)."""
    encoder_name = meta["encoder"]
    n_qubits = int(meta["n_qubits"])
    ansatz_depth = int(meta.get("ansatz_depth", 2))
    encoder = make_encoder(encoder_name, n_qubits=n_qubits, n_features=n_features)

    theta_enc = meta["theta_enc"].to(torch.float64)
    theta_ansatz = meta["theta_ansatz"].to(torch.float64)

    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_indexes(n_qubits, n_classes)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(x, theta_enc, theta_ansatz):
        encoder.apply(x, theta_enc)
        idx = 0
        for _ in range(ansatz_depth):
            for q in range(n_qubits):
                qml.RY(theta_ansatz[idx], wires=q)
                idx += 1
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.probs(wires=output_wires)

    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    return lambda: circ(x, theta_enc, theta_ansatz), \
        (lambda: qml.draw_mpl(circ, decimals=2)(x, theta_enc, theta_ansatz))


def build_qnode_encoder_plus_genome(meta: dict, n_features: int, n_classes: int):
    """Stage B / Stage C: external encoder + evolved genome gates."""
    encoder_name = meta["encoder"]
    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    n_qubits = len(genome.qubits)
    encoder = make_encoder(encoder_name, n_qubits=n_qubits, n_features=n_features)

    theta_enc = meta.get("theta_enc")
    if theta_enc is None:
        theta_enc = torch.zeros(0, dtype=torch.float64)
    else:
        theta_enc = theta_enc.to(torch.float64)

    params_dict = _params_dict_from_genome(genome)
    genome.sort_gates()
    gates_snapshot = list(genome.gates)
    qubits = genome.qubits

    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_indexes(n_qubits, n_classes)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(x, theta_enc, params_dict):
        encoder.apply(x, theta_enc)
        for gate in gates_snapshot:
            if gate.enabled:
                gate.add_to_pennylane_circuit(qubits, params=params_dict)
        return qml.probs(wires=output_wires)

    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    return lambda: circ(x, theta_enc, params_dict), \
        (lambda: qml.draw_mpl(circ, decimals=2)(x, theta_enc, params_dict))


_ENC_AXIS_OP_E = {"enc_ry": qml.RY, "enc_rx": qml.RX, "enc_rz": qml.RZ}


def build_qnode_stage_e(meta: dict, n_features: int, n_classes: int):
    """Stage E: enc_ry/rx/rz feature-dependent gates + plain ansatz gates."""
    # Make sure the encoder gate specs are registered for from_dict to work.
    import src.Ryan_cookin.stage_e as _stage_e
    _stage_e._register_encoder_gate_specs()

    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    n_qubits = len(genome.qubits)
    genome.sort_gates()
    gates_snapshot = list(genome.gates)
    qubits = genome.qubits
    params_dict = _params_dict_from_genome(genome)

    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_indexes(n_qubits, n_classes)

    enc_meta: dict[int, tuple[int, int, str]] = {}
    for gate in gates_snapshot:
        if gate.method_name in _ENC_AXIS_OP_E:
            wire = qubits.index(gate.qubits[0])
            feat_idx = wire % n_features
            enc_meta[id(gate)] = (wire, feat_idx, gate.method_name)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(x, params_dict):
        for gate in gates_snapshot:
            if not gate.enabled:
                continue
            m = enc_meta.get(id(gate))
            if m is not None:
                wire, fi, mname = m
                a = params_dict[f"{gate.innovation_number}:a"]
                b = params_dict[f"{gate.innovation_number}:b"]
                _ENC_AXIS_OP_E[mname](a * x[fi] + b, wires=wire)
            else:
                gate.add_to_pennylane_circuit(qubits, params=params_dict)
        return qml.probs(wires=output_wires)

    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    return lambda: circ(x, params_dict), \
        (lambda: qml.draw_mpl(circ, decimals=2)(x, params_dict))


_ENC_2Q_F = {"enc_xx": qml.IsingXX, "enc_yy": qml.IsingYY, "enc_zz": qml.IsingZZ}


def build_qnode_stage_f(meta: dict, n_features: int, n_classes: int):
    """Stage F (v1 or v2): enc_rot + optionally enc_xx/yy/zz + ansatz gates."""
    import src.Ryan_cookin.stage_f as _stage_f
    _stage_f._register_stage_f_specs()

    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    n_qubits = len(genome.qubits)
    genome.sort_gates()
    gates_snapshot = list(genome.gates)
    qubits = genome.qubits
    params_dict = _params_dict_from_genome(genome)

    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_indexes(n_qubits, n_classes)

    enc1_meta: dict[int, tuple[int, int]] = {}
    enc2_meta: dict[int, tuple[int, int, int, int, Callable]] = {}
    for gate in gates_snapshot:
        if gate.method_name == "enc_rot":
            w0 = qubits.index(gate.qubits[0])
            enc1_meta[id(gate)] = (w0, w0 % n_features)
        elif gate.method_name in _ENC_2Q_F:
            w0 = qubits.index(gate.qubits[0])
            w1 = qubits.index(gate.qubits[1])
            enc2_meta[id(gate)] = (
                w0, w1, w0 % n_features, w1 % n_features,
                _ENC_2Q_F[gate.method_name],
            )

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(x, params_dict):
        for gate in gates_snapshot:
            if not gate.enabled:
                continue
            inn = gate.innovation_number
            m1 = enc1_meta.get(id(gate))
            if m1 is not None:
                wire, fi = m1
                a_a = params_dict[f"{inn}:a_alpha"]
                b_a = params_dict[f"{inn}:b_alpha"]
                a_b = params_dict[f"{inn}:a_beta"]
                b_b = params_dict[f"{inn}:b_beta"]
                a_g = params_dict[f"{inn}:a_gamma"]
                b_g = params_dict[f"{inn}:b_gamma"]
                xi = x[fi]
                qml.Rot(a_a*xi + b_a, a_b*xi + b_b, a_g*xi + b_g, wires=wire)
                continue
            m2 = enc2_meta.get(id(gate))
            if m2 is not None:
                w0, w1, f0, f1, axis_op = m2
                a0 = params_dict[f"{inn}:a_q0"]
                a1 = params_dict[f"{inn}:a_q1"]
                b  = params_dict[f"{inn}:b"]
                axis_op(a0 * x[f0] + a1 * x[f1] + b, wires=[w0, w1])
                continue
            gate.add_to_pennylane_circuit(qubits, params=params_dict)
        return qml.probs(wires=output_wires)

    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    return lambda: circ(x, params_dict), \
        (lambda: qml.draw_mpl(circ, decimals=2)(x, params_dict))


# ---------- Stage specs ---------------------------------------------------

@dataclass
class StageSpec:
    name: str                 # short label used in filename / title
    csv_path: str
    weights_dir: str
    fname_template: str       # uses {dataset}, {encoder}, {n_qubits}, {seed}
    qnode_builder: Callable
    encoder_required: bool    # True for A/B/C (encoder in CSV row), False for E/F

    def champion_row(self, dataset: str) -> dict | None:
        if not os.path.exists(self.csv_path):
            return None
        with open(self.csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset]
        if not rows:
            return None
        def key(r):
            return (-float(r["test_acc"]), float(r["test_loss"]))
        return sorted(rows, key=key)[0]


STAGES: list[StageSpec] = [
    StageSpec(
        name="A",
        csv_path=f"{RESULTS_DIR}/stage_a_trained.csv",
        weights_dir=f"{RESULTS_DIR}/weights",
        fname_template="{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt",
        qnode_builder=build_qnode_stage_a,
        encoder_required=True,
    ),
    StageSpec(
        name="B",
        csv_path=f"{RESULTS_DIR}/stage_b.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_b",
        fname_template="{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt",
        qnode_builder=build_qnode_encoder_plus_genome,
        encoder_required=True,
    ),
    StageSpec(
        name="B_multiseed",
        csv_path=f"{RESULTS_DIR}/stage_b_multiseed.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_b_multiseed",
        fname_template="{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt",
        qnode_builder=build_qnode_encoder_plus_genome,
        encoder_required=True,
    ),
    StageSpec(
        name="C",
        csv_path=f"{RESULTS_DIR}/stage_c.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_c",
        fname_template="{dataset}_{encoder}_seed{seed}.pt",
        qnode_builder=build_qnode_encoder_plus_genome,
        encoder_required=True,
    ),
    StageSpec(
        name="C_multiseed",
        csv_path=f"{RESULTS_DIR}/stage_c_multiseed.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_c_multiseed",
        fname_template="{dataset}_{encoder}_seed{seed}.pt",
        qnode_builder=build_qnode_encoder_plus_genome,
        encoder_required=True,
    ),
    StageSpec(
        name="E_v1",
        csv_path=f"{RESULTS_DIR}/stage_e.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_e",
        fname_template="{dataset}_seed{seed}.pt",
        qnode_builder=build_qnode_stage_e,
        encoder_required=False,
    ),
    StageSpec(
        name="E_v2",
        csv_path=f"{RESULTS_DIR}/stage_e_v2.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_e_v2",
        fname_template="{dataset}_seed{seed}.pt",
        qnode_builder=build_qnode_stage_e,
        encoder_required=False,
    ),
    StageSpec(
        name="F_v1",
        csv_path=f"{RESULTS_DIR}/stage_f_v1.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_f_v1",
        fname_template="{dataset}_seed{seed}.pt",
        qnode_builder=build_qnode_stage_f,
        encoder_required=False,
    ),
    StageSpec(
        name="F_v2",
        csv_path=f"{RESULTS_DIR}/stage_f_v2.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_f_v2",
        fname_template="{dataset}_seed{seed}.pt",
        qnode_builder=build_qnode_stage_f,
        encoder_required=False,
    ),
    StageSpec(
        name="F_v2_multiseed",
        csv_path=f"{RESULTS_DIR}/stage_f_v2_multiseed.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_f_v2_multiseed",
        fname_template="{dataset}_seed{seed}.pt",
        qnode_builder=build_qnode_stage_f,
        encoder_required=False,
    ),
    StageSpec(
        name="F_v2_big",
        csv_path=f"{RESULTS_DIR}/stage_f_v2_big.csv",
        weights_dir=f"{RESULTS_DIR}/weights_stage_f_v2_big",
        fname_template="{dataset}_seed{seed}.pt",
        qnode_builder=build_qnode_stage_f,
        encoder_required=False,
    ),
]


def _build_filename(stage: StageSpec, row: dict) -> str:
    """Substitute the dataset / encoder / n_qubits / seed fields into the
    stage's filename template. n_qubits comes from CSV column n_qubits for
    A/B (manual N sweep) or best_n_qubits for C/E/F (evolved N)."""
    n_qubits = row.get("n_qubits") or row.get("best_n_qubits") or "?"
    return stage.fname_template.format(
        dataset=row["dataset"],
        encoder=row.get("encoder", ""),
        n_qubits=n_qubits,
        seed=row.get("seed", 0),
    )


def _title(stage: StageSpec, row: dict, n_features: int, n_classes: int,
           extra_gate_counts: dict) -> list[str]:
    encoder = row.get("encoder", "(evolved)")
    n_qubits = row.get("n_qubits") or row.get("best_n_qubits") or "?"
    label = f"stage {stage.name} -- {row['dataset']}"
    sub = (
        f"encoder={encoder}, N={n_qubits}, D={n_features}, K={n_classes}, "
        f"seed={row.get('seed', 0)}"
    )
    gate_pieces = []
    for k in ("best_n_gates", "n_enc_gates", "n_ansatz_gates"):
        if k in row and row[k] != "":
            gate_pieces.append(f"{k}={row[k]}")
    trained = (
        f"trained: test_acc={float(row['test_acc']):.3f}, "
        f"test_loss={float(row['test_loss']):.4f}"
    )
    return [label, sub, ", ".join(gate_pieces) if gate_pieces else "", trained]


def render_one(stage: StageSpec, row: dict, out_dir: str) -> bool:
    dataset = row["dataset"]
    n_features, n_classes = DATASET_REGISTRY[dataset]
    fname = _build_filename(stage, row)
    weights_path = os.path.join(stage.weights_dir, fname)
    if not os.path.exists(weights_path):
        print(f"  MISSING  stage {stage.name:18s} {dataset:14s}  -> {weights_path}")
        return False

    meta = torch.load(weights_path, map_location="cpu", weights_only=False)
    _, drawer = stage.qnode_builder(meta, n_features, n_classes)
    fig, ax = drawer()

    title = "\n".join(t for t in _title(stage, row, n_features, n_classes, {}) if t)
    fig.suptitle(title, fontsize=9)

    out_path = os.path.join(out_dir, f"{stage.name}_{dataset}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  rendered stage {stage.name:18s} {dataset:14s}  "
        f"test_acc={float(row['test_acc']):.3f}"
    )
    return True


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"out_dir: {args.out_dir}\n")

    rendered = 0
    missing = 0
    for stage in STAGES:
        for dataset in DATASETS:
            row = stage.champion_row(dataset)
            if row is None:
                print(f"  NO ROW   stage {stage.name:18s} {dataset:14s}")
                missing += 1
                continue
            ok = render_one(stage, row, args.out_dir)
            if ok:
                rendered += 1
            else:
                missing += 1
    print(f"\nDone. {rendered} PNGs in {args.out_dir}/  "
          f"({missing} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
