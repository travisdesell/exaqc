"""Render PNGs of the best-of-run evolved circuits from the Stage B sweep.

Each .pt file in weights_stage_b/ contains:
  - best_genome_dict: serialized CircuitGenome (from genome.to_dict())
  - theta_enc:        trained encoder params (or None)
  - fitness metrics for the title strip

We rebuild the genome via CircuitGenome.from_dict, build a qnode that
mirrors Stage B's EncoderAnsatzObjective (encoder.apply -> evolved
gates in depth order -> qml.probs on output wires), and draw it with
qml.draw_mpl using the trained angles. So unlike the structure-only
Stage A draw mode, every PNG here shows the actual converged solution.

Usage:
    python scripts/render_stage_b_circuits.py
    python scripts/render_stage_b_circuits.py \
        --weights_dir src/Ryan_cookin/results/weights_stage_b \
        --out_dir src/Ryan_cookin/results/circuits_stage_b
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

from src.circuits.circuit import CircuitGenome
from src.Ryan_cookin.encoders import make_encoder


DEFAULT_WEIGHTS_DIR = "src/Ryan_cookin/results/weights_stage_b"
DEFAULT_OUT_DIR = "src/Ryan_cookin/results/circuits_stage_b"

DATASETS = [
    ("iris",          4, 3),
    ("wine",         13, 3),
    ("seeds",         7, 3),
    ("breast_cancer",30, 2),
]
ENCODERS = ["fixed_angle", "fixed_amplitude", "learned"]
N_VALUES = [4, 6, 8, 10]


def _build_combos():
    out = []
    for ds_name, n_features, n_classes in DATASETS:
        for n in N_VALUES:
            for enc in ENCODERS:
                label = f"{ds_name}_N{n}_{enc}"
                out.append((label, ds_name, enc, n, n_features, n_classes))
    return out


COMBOS = _build_combos()


def _draw_one(weights_path: str, n_features: int, n_classes: int) -> dict:
    """Load a .pt and produce the draw artefacts (no rendering yet)."""
    meta = torch.load(weights_path, map_location="cpu", weights_only=False)

    genome_dict = meta["best_genome_dict"]
    genome = CircuitGenome.from_dict(genome_dict)
    n_qubits = meta["n_qubits"]
    encoder_name = meta["encoder"]

    encoder = make_encoder(encoder_name, n_qubits=n_qubits, n_features=n_features)
    theta_enc = meta.get("theta_enc")
    if theta_enc is None:
        theta_enc = torch.zeros(0, dtype=torch.float64)

    # Build the trained-params dict directly from the genome's stored
    # parameter values (objective.train_genome_objective wrote them back
    # via torch_params_to_genome at the end of training).
    params_dict = {}
    for gate in genome.gates:
        if gate.enabled:
            for name, value in gate.parameters.items():
                key = f"{gate.innovation_number}:{name}"
                params_dict[key] = torch.tensor(float(value), dtype=torch.float64)

    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    output_indexes = list(range(n_output))

    dev = qml.device("default.qubit", wires=n_qubits)
    genome.sort_gates()
    gates_snapshot = list(genome.gates)
    qubits = genome.qubits

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(x, theta_enc, params_dict):
        encoder.apply(x, theta_enc)
        for gate in gates_snapshot:
            if gate.enabled:
                gate.add_to_pennylane_circuit(qubits, params=params_dict)
        return qml.probs(wires=output_indexes)

    # Draw with concrete inputs. Use the same linspace pattern as the
    # Stage A renderer for visual consistency.
    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    fig, ax = qml.draw_mpl(circ, decimals=2, style="default")(x, theta_enc, params_dict)
    return {
        "fig": fig,
        "ax": ax,
        "meta": meta,
        "n_enabled_gates": sum(1 for g in gates_snapshot if g.enabled),
        "n_genome_params": len(params_dict),
    }


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--weights_dir", type=str, default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"weights_dir: {args.weights_dir}")
    print(f"out_dir:     {args.out_dir}\n")

    n_rendered = 0
    n_missing = 0
    for label, ds_name, encoder_name, n_qubits, n_features, n_classes in COMBOS:
        weights_path = os.path.join(
            args.weights_dir,
            f"{ds_name}_N{n_qubits}_{encoder_name}_seed{args.seed}.pt",
        )
        if not os.path.exists(weights_path):
            print(f"  MISSING  {label}")
            n_missing += 1
            continue

        out = _draw_one(weights_path, n_features, n_classes)
        meta = out["meta"]
        fig = out["fig"]

        title_lines = [
            label,
            f"encoder={encoder_name}, N={n_qubits}, D={n_features}, K={n_classes}, "
            f"gates={out['n_enabled_gates']}, ansatz_params={out['n_genome_params']}",
            f"trained: n_genomes={meta['n_genomes']}, pop={meta['pop_size']}, "
            f"epochs_inner={meta['epochs_inner']}, "
            f"train_acc={meta['train_acc']:.3f}, "
            f"test_acc={meta['test_acc']:.3f}, "
            f"test_loss={meta['test_loss']:.4f}",
        ]
        fig.suptitle("\n".join(title_lines), fontsize=9)

        out_path = os.path.join(args.out_dir, f"{label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(
            f"  rendered {label:34s} gates={out['n_enabled_gates']:3d} "
            f"test_acc={meta['test_acc']:.3f}"
        )
        n_rendered += 1

    print(
        f"\nDone. {n_rendered} PNGs in {args.out_dir}/  "
        f"({n_missing} missing weights)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
