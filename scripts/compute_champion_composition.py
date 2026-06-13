"""Compute (n_qubits, depth, n_1q, n_2q) for the champion circuit per
(stage, dataset).

For each champion .pt we rebuild the same qnode the stage driver ran
in production, execute it once with a placeholder x to populate the
tape, then walk tape.operations:

  - n_qubits  : tape.num_wires (which matches the device wire count;
                for evolved-N stages this is the genome's final N)
  - n_1q      : count of ops with len(op.wires) == 1
  - n_2q      : count of ops with len(op.wires) == 2
  - n_multi   : count of ops with len(op.wires) >= 3 (e.g. AmplitudeEmbedding
                stays as a single multi-wire op unless we expand it)
  - depth     : greedy layer-assignment depth (parallelism across
                disjoint-wire ops within a layer)

Output:
  - src/Ryan_cookin/results/champion_composition.csv
  - a text table printed to stdout, ready to paste into tables.txt
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import pennylane as qml
import torch

from src.circuits.circuit import CircuitGenome
from src.Ryan_cookin.encoders import make_encoder


RESULTS_DIR = "src/Ryan_cookin/results"

DATASET_REGISTRY = {
    "iris":          (4,  3),
    "wine":         (13,  3),
    "seeds":         (7,  3),
    "breast_cancer":(30,  2),
}
DATASETS = list(DATASET_REGISTRY.keys())


# ---------- Stage spec (mirrors render_champion_*.py) ---------------------

class StageSpec:
    def __init__(self, name, csv_path, weights_dir, fname_template, kind):
        self.name = name
        self.csv_path = csv_path
        self.weights_dir = weights_dir
        self.fname_template = fname_template
        self.kind = kind

    def champion_row(self, dataset):
        if not os.path.exists(self.csv_path):
            return None
        with open(self.csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset]
        if not rows:
            return None
        return sorted(rows, key=lambda r: (-float(r["test_acc"]), float(r["test_loss"])))[0]


STAGES = [
    StageSpec("A",              f"{RESULTS_DIR}/stage_a_trained.csv",      f"{RESULTS_DIR}/weights",                       "{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt", "A"),
    StageSpec("B",              f"{RESULTS_DIR}/stage_b.csv",              f"{RESULTS_DIR}/weights_stage_b",               "{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt", "B_or_C"),
    StageSpec("B_multiseed",    f"{RESULTS_DIR}/stage_b_multiseed.csv",    f"{RESULTS_DIR}/weights_stage_b_multiseed",     "{dataset}_N{n_qubits}_{encoder}_seed{seed}.pt", "B_or_C"),
    StageSpec("C",              f"{RESULTS_DIR}/stage_c.csv",              f"{RESULTS_DIR}/weights_stage_c",               "{dataset}_{encoder}_seed{seed}.pt",             "B_or_C"),
    StageSpec("C_multiseed",    f"{RESULTS_DIR}/stage_c_multiseed.csv",    f"{RESULTS_DIR}/weights_stage_c_multiseed",     "{dataset}_{encoder}_seed{seed}.pt",             "B_or_C"),
    StageSpec("E_v1",           f"{RESULTS_DIR}/stage_e.csv",              f"{RESULTS_DIR}/weights_stage_e",               "{dataset}_seed{seed}.pt",                       "E"),
    StageSpec("E_v2",           f"{RESULTS_DIR}/stage_e_v2.csv",           f"{RESULTS_DIR}/weights_stage_e_v2",            "{dataset}_seed{seed}.pt",                       "E"),
    StageSpec("F_v1",           f"{RESULTS_DIR}/stage_f_v1.csv",           f"{RESULTS_DIR}/weights_stage_f_v1",            "{dataset}_seed{seed}.pt",                       "F"),
    StageSpec("F_v2",           f"{RESULTS_DIR}/stage_f_v2.csv",           f"{RESULTS_DIR}/weights_stage_f_v2",            "{dataset}_seed{seed}.pt",                       "F"),
    StageSpec("F_v2_multiseed", f"{RESULTS_DIR}/stage_f_v2_multiseed.csv", f"{RESULTS_DIR}/weights_stage_f_v2_multiseed",  "{dataset}_seed{seed}.pt",                       "F"),
    StageSpec("F_v2_big",       f"{RESULTS_DIR}/stage_f_v2_big.csv",       f"{RESULTS_DIR}/weights_stage_f_v2_big",        "{dataset}_seed{seed}.pt",                       "F"),
]


def _build_filename(stage, row):
    n_qubits = row.get("n_qubits") or row.get("best_n_qubits") or "?"
    return stage.fname_template.format(
        dataset=row["dataset"],
        encoder=row.get("encoder", ""),
        n_qubits=n_qubits,
        seed=row.get("seed", 0),
    )


def _output_wires(n_qubits, n_classes):
    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    return list(range(n_output))


def _params_from_genome(genome):
    out = {}
    for g in genome.gates:
        if not g.enabled:
            continue
        for name, v in g.parameters.items():
            out[f"{g.innovation_number}:{name}"] = torch.tensor(float(v), dtype=torch.float64)
    return out


# Per-stage qnode builders. Each returns a callable that returns the qnode
# (so we can call qnode(x, ...) and inspect qnode.qtape afterward).

def build_qnode_A(meta, n_features, n_classes):
    encoder_name = meta["encoder"]
    n_qubits = int(meta["n_qubits"])
    ansatz_depth = int(meta.get("ansatz_depth", 2))
    encoder = make_encoder(encoder_name, n_qubits=n_qubits, n_features=n_features)
    theta_enc = meta["theta_enc"].to(torch.float64)
    theta_ansatz = meta["theta_ansatz"].to(torch.float64)
    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_wires(n_qubits, n_classes)

    @qml.qnode(dev, interface="torch")
    def circ(x):
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
    return circ, x


def build_qnode_BC(meta, n_features, n_classes):
    encoder_name = meta["encoder"]
    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    n_qubits = len(genome.qubits)
    encoder = make_encoder(encoder_name, n_qubits=n_qubits, n_features=n_features)
    theta_enc = meta.get("theta_enc")
    theta_enc = theta_enc.to(torch.float64) if theta_enc is not None else torch.zeros(0, dtype=torch.float64)
    params_dict = _params_from_genome(genome)
    genome.sort_gates()
    gates_snapshot = list(genome.gates)
    qubits = genome.qubits
    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_wires(n_qubits, n_classes)

    @qml.qnode(dev, interface="torch")
    def circ(x):
        encoder.apply(x, theta_enc)
        for g in gates_snapshot:
            if g.enabled:
                g.add_to_pennylane_circuit(qubits, params=params_dict)
        return qml.probs(wires=output_wires)

    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    return circ, x


_ENC_AXIS_E = {"enc_ry": qml.RY, "enc_rx": qml.RX, "enc_rz": qml.RZ}


def build_qnode_E(meta, n_features, n_classes):
    import src.Ryan_cookin.stage_e as _s
    _s._register_encoder_gate_specs()
    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    n_qubits = len(genome.qubits)
    genome.sort_gates()
    gates_snapshot = list(genome.gates)
    qubits = genome.qubits
    params_dict = _params_from_genome(genome)
    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_wires(n_qubits, n_classes)
    enc_meta = {}
    for g in gates_snapshot:
        if g.method_name in _ENC_AXIS_E:
            wire = qubits.index(g.qubits[0])
            enc_meta[id(g)] = (wire, wire % n_features, g.method_name)

    @qml.qnode(dev, interface="torch")
    def circ(x):
        for g in gates_snapshot:
            if not g.enabled:
                continue
            m = enc_meta.get(id(g))
            if m is not None:
                w, fi, mname = m
                a = params_dict[f"{g.innovation_number}:a"]
                b = params_dict[f"{g.innovation_number}:b"]
                _ENC_AXIS_E[mname](a * x[fi] + b, wires=w)
            else:
                g.add_to_pennylane_circuit(qubits, params=params_dict)
        return qml.probs(wires=output_wires)

    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    return circ, x


_ENC_2Q_F = {"enc_xx": qml.IsingXX, "enc_yy": qml.IsingYY, "enc_zz": qml.IsingZZ}


def build_qnode_F(meta, n_features, n_classes):
    import src.Ryan_cookin.stage_f as _s
    _s._register_stage_f_specs()
    genome = CircuitGenome.from_dict(meta["best_genome_dict"])
    n_qubits = len(genome.qubits)
    genome.sort_gates()
    gates_snapshot = list(genome.gates)
    qubits = genome.qubits
    params_dict = _params_from_genome(genome)
    dev = qml.device("default.qubit", wires=n_qubits)
    output_wires = _output_wires(n_qubits, n_classes)
    enc1_meta = {}
    enc2_meta = {}
    for g in gates_snapshot:
        if g.method_name == "enc_rot":
            w0 = qubits.index(g.qubits[0])
            enc1_meta[id(g)] = (w0, w0 % n_features)
        elif g.method_name in _ENC_2Q_F:
            w0 = qubits.index(g.qubits[0])
            w1 = qubits.index(g.qubits[1])
            enc2_meta[id(g)] = (w0, w1, w0 % n_features, w1 % n_features, _ENC_2Q_F[g.method_name])

    @qml.qnode(dev, interface="torch")
    def circ(x):
        for g in gates_snapshot:
            if not g.enabled:
                continue
            inn = g.innovation_number
            m1 = enc1_meta.get(id(g))
            if m1 is not None:
                wire, fi = m1
                aa = params_dict[f"{inn}:a_alpha"]; ba = params_dict[f"{inn}:b_alpha"]
                ab = params_dict[f"{inn}:a_beta"];  bb = params_dict[f"{inn}:b_beta"]
                ag = params_dict[f"{inn}:a_gamma"]; bg = params_dict[f"{inn}:b_gamma"]
                xi = x[fi]
                qml.Rot(aa*xi + ba, ab*xi + bb, ag*xi + bg, wires=wire)
                continue
            m2 = enc2_meta.get(id(g))
            if m2 is not None:
                w0, w1, f0, f1, axis = m2
                a0 = params_dict[f"{inn}:a_q0"]
                a1 = params_dict[f"{inn}:a_q1"]
                b  = params_dict[f"{inn}:b"]
                axis(a0*x[f0] + a1*x[f1] + b, wires=[w0, w1])
                continue
            g.add_to_pennylane_circuit(qubits, params=params_dict)
        return qml.probs(wires=output_wires)

    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    return circ, x


BUILDERS = {
    "A":      build_qnode_A,
    "B_or_C": build_qnode_BC,
    "E":      build_qnode_E,
    "F":      build_qnode_F,
}


def composition_from_qnode(qnode, x, register_size: int):
    """Build a tape from the qnode's underlying function so we can walk ops.

    PennyLane's qnode.qtape was removed in recent releases; instead we use
    qml.tape.make_qscript on the wrapped function, which traces it once
    under a queueing context and returns a QuantumScript with .operations.

    register_size is passed in because qscript.wires only counts wires the
    circuit actually touches; for Stage E/F evolved circuits the genome
    can have N=6 qubits but only place gates on a subset of them. We want
    the genome's full N for the "number of qubits" column.
    """
    # qnode.func is the undecorated body
    qscript = qml.tape.make_qscript(qnode.func)(x)
    ops = qscript.operations
    n_qubits = register_size
    n_1q = sum(1 for op in ops if len(op.wires) == 1)
    n_2q = sum(1 for op in ops if len(op.wires) == 2)
    n_multi = sum(1 for op in ops if len(op.wires) >= 3)
    # Greedy depth: each op gets layer = max(used[w] for w in op.wires) + 1
    used = {}
    depth = 0
    for op in ops:
        ws = list(op.wires)
        layer = max((used.get(w, 0) for w in ws), default=0) + 1
        for w in ws:
            used[w] = layer
        depth = max(depth, layer)
    return {"n_qubits": n_qubits, "depth": depth, "n_1q": n_1q,
            "n_2q": n_2q, "n_multi": n_multi}


def compute_one(stage, row):
    dataset = row["dataset"]
    n_features, n_classes = DATASET_REGISTRY[dataset]
    weights_path = os.path.join(stage.weights_dir, _build_filename(stage, row))
    if not os.path.exists(weights_path):
        return None
    meta = torch.load(weights_path, map_location="cpu", weights_only=False)
    builder = BUILDERS[stage.kind]
    qnode, x = builder(meta, n_features, n_classes)
    # Register size from the genome (Stage E/F) or from the meta dict.
    if "best_genome_dict" in meta:
        register_size = len(meta["best_genome_dict"]["input_qubits"])
    else:
        register_size = int(meta["n_qubits"])
    stats = composition_from_qnode(qnode, x, register_size)
    stats.update({
        "stage":    stage.name,
        "dataset":  dataset,
        "test_acc": round(float(row["test_acc"]), 4),
    })
    return stats


def format_table(rows):
    headers = ["stage", "dataset", "N", "depth", "n_1q", "n_2q", "n_multi", "test_acc"]
    widths = [max(len(h), max(len(str(r[h.replace('N','n_qubits')])) for r in rows)) for h in headers]
    line_fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [line_fmt.format(*headers),
             line_fmt.format(*["-"*w for w in widths])]
    for r in rows:
        lines.append(line_fmt.format(
            r["stage"], r["dataset"], r["n_qubits"], r["depth"],
            r["n_1q"], r["n_2q"], r["n_multi"], f"{r['test_acc']:.3f}",
        ))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default=f"{RESULTS_DIR}/champion_composition.csv")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rows = []
    for stage in STAGES:
        for ds in DATASETS:
            row = stage.champion_row(ds)
            if row is None:
                continue
            stats = compute_one(stage, row)
            if stats is None:
                print(f"  MISSING  stage {stage.name:18s} {ds}")
                continue
            rows.append(stats)
            print(f"  stage {stage.name:18s} {ds:14s} "
                  f"N={stats['n_qubits']:2d} depth={stats['depth']:3d} "
                  f"1q={stats['n_1q']:3d} 2q={stats['n_2q']:3d} "
                  f"multi={stats['n_multi']:2d}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage","dataset","n_qubits","depth","n_1q","n_2q","n_multi","test_acc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {args.out_csv}\n")

    print("=" * 70)
    print("Paste this into src/Ryan_cookin/tables.txt:\n")
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
