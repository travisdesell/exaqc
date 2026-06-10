"""Render PNGs of the encoder + ansatz combos used in Stage A.

These are NOT evolved circuits -- Stage A's ansatz is hand-built and
identical for every encoder choice. The point of these images is to
show what the actual circuit being trained looks like for a given
(encoder, N, dataset) cell so reviewers can see the structure.

Two modes:

    # Default: structure-only PNGs with placeholder rotation values.
    python scripts/render_stage_a_circuits.py

    # Trained-weights mode: load per-cell trained theta_enc / theta_ansatz
    # from .pt files written by stage_a.py --weights_dir, render circuits
    # whose printed rotation angles are the ones the optimizer converged to.
    python scripts/render_stage_a_circuits.py \
        --weights_dir src/Ryan_cookin/results/weights \
        --out_dir src/Ryan_cookin/results/circuits_trained
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

from src.Ryan_cookin.encoders import make_encoder, initial_encoder_params


DEFAULT_OUT_DIR = "src/Ryan_cookin/results/circuits"

# (dataset_name, n_features, n_classes) — must match Stage A's DATASET_REGISTRY.
DATASETS = [
    ("iris",          4, 3),
    ("wine",         13, 3),
    ("seeds",         7, 3),
    ("breast_cancer",30, 2),
]
ENCODERS = ["fixed_angle", "fixed_amplitude", "learned"]
N_VALUES = [4, 6, 8, 10]


def _build_combos():
    """All 48 (dataset, N, encoder) cells of the Stage A sweep."""
    out = []
    for ds_name, n_features, n_classes in DATASETS:
        for n in N_VALUES:
            for enc in ENCODERS:
                label = f"{ds_name}_N{n}_{enc}"
                out.append((label, enc, n, n_features, n_classes))
    return out


COMBOS = _build_combos()


def build_qnode(
    encoder_name: str,
    n_qubits: int,
    n_features: int,
    n_classes: int,
    *,
    weights_dir: str | None = None,
    dataset_name: str | None = None,
    seed: int = 0,
):
    encoder = make_encoder(encoder_name, n_qubits=n_qubits, n_features=n_features)
    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    output_wires = list(range(n_output))

    dev = qml.device("default.qubit", wires=n_qubits)
    ansatz_depth = 2
    n_ansatz_params = n_qubits * ansatz_depth

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

    # Concrete inputs for drawing. Try to load trained weights from the
    # weights_dir if one was provided; fall back to placeholder values
    # so the structure-only render still works.
    x = torch.tensor(np.linspace(0.1, 0.9, n_features), dtype=torch.float64)
    trained_meta = None
    if weights_dir is not None and dataset_name is not None:
        weights_path = os.path.join(
            weights_dir,
            f"{dataset_name}_N{n_qubits}_{encoder_name}_seed{seed}.pt",
        )
        if os.path.exists(weights_path):
            trained_meta = torch.load(weights_path, map_location="cpu", weights_only=False)
            theta_enc = trained_meta["theta_enc"].to(torch.float64)
            theta_ansatz = trained_meta["theta_ansatz"].to(torch.float64)
        else:
            theta_enc = initial_encoder_params(encoder.n_params, seed=seed)
            theta_ansatz = torch.tensor(
                np.linspace(-0.5, 0.5, n_ansatz_params), dtype=torch.float64,
            )
    else:
        theta_enc = initial_encoder_params(encoder.n_params, seed=seed)
        theta_ansatz = torch.tensor(
            np.linspace(-0.5, 0.5, n_ansatz_params), dtype=torch.float64,
        )
    return circ, x, theta_enc, theta_ansatz, encoder.n_params, n_ansatz_params, trained_meta


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--weights_dir", type=str, default=None,
                   help="If set, load trained theta_enc/theta_ansatz per cell "
                        "from .pt files in this directory and render with them.")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"out_dir: {args.out_dir}")
    print(f"weights_dir: {args.weights_dir or '(none -- placeholder rotations)'}\n")

    for label, encoder_name, n_qubits, n_features, n_classes in COMBOS:
        # Recover dataset name from label (everything before "_N").
        dataset_name = label.split("_N")[0]

        circ, x, te, ta, n_enc, n_ans, trained_meta = build_qnode(
            encoder_name, n_qubits, n_features, n_classes,
            weights_dir=args.weights_dir,
            dataset_name=dataset_name,
            seed=args.seed,
        )

        weights_status = "TRAINED" if trained_meta is not None else "placeholder"
        print(f"  rendering {label}  [{weights_status}]")

        fig, ax = qml.draw_mpl(circ, decimals=2, style="default")(x, te, ta)

        title_lines = [
            label,
            f"encoder={encoder_name}, N={n_qubits}, D={n_features}, K={n_classes}, "
            f"encoder_params={n_enc}, ansatz_params={n_ans}",
        ]
        if trained_meta is not None:
            title_lines.append(
                f"trained: epochs={trained_meta['epochs']}, "
                f"train_acc={trained_meta['train_acc']:.3f}, "
                f"test_acc={trained_meta['test_acc']:.3f}, "
                f"test_loss={trained_meta['test_loss']:.4f}"
            )
        else:
            title_lines.append("(structure only -- rotation values are placeholders, NOT trained)")
        fig.suptitle("\n".join(title_lines), fontsize=9)

        out_path = os.path.join(args.out_dir, f"{label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nDone. {len(COMBOS)} PNGs in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
