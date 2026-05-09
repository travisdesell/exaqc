"""Stage A: trainable encoder + fixed ansatz, swept over N and datasets.

Run:
    PYTHONPATH=. python -m src.Ryan_cookin.stage_a \
        --datasets iris wine \
        --encoders fixed_angle fixed_amplitude learned \
        --n_qubits 4 6 8 \
        --epochs 30 \
        --batch_size 8

Writes one CSV row per (dataset, encoder, n_qubits, seed) cell to
`src/Ryan_cookin/results/stage_a.csv`. Re-running with different
combinations *appends* to the same file so partial sweeps accumulate.

The downstream ansatz is deliberately simple and identical for every
encoder choice: `depth_ansatz` layers of (per-qubit RY + linear CNOT
chain). The point of Stage A is to isolate the encoder's contribution.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Iterable

import numpy as np
import pennylane as qml
import torch

from src.datasets.classification import (
    BreastCancerDataset,
    IrisDataset,
    SeedsDataset,
    WineDataset,
)
from src.Ryan_cookin.encoders import (
    ENCODERS,
    initial_encoder_params,
    make_encoder,
)


DATASET_REGISTRY = {
    "iris":          (IrisDataset,         4,  3),
    "wine":          (WineDataset,        13,  3),
    "seeds":         (SeedsDataset,        7,  3),
    "breast_cancer": (BreastCancerDataset, 30, 2),
}


# ---------- ansatz ----------------------------------------------------------

def ansatz_n_params(n_qubits: int, depth: int) -> int:
    return n_qubits * depth


def apply_ansatz(theta_ansatz: torch.Tensor, *, n_qubits: int, depth: int) -> None:
    """Fixed-structure VQC: per-qubit RY then linear CNOT chain, repeated."""
    idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qml.RY(theta_ansatz[idx], wires=q)
            idx += 1
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])


# ---------- training core ---------------------------------------------------

def build_qnode(
    encoder,
    *,
    n_qubits: int,
    ansatz_depth: int,
    n_classes: int,
):
    """Compose encoder + ansatz inside a single torch-interfaced QNode.

    Output qubits for measurement are the first ceil(log2(n_classes))
    qubits of the register, matching the convention used by the rest of
    the codebase.
    """
    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    output_wires = list(range(n_output))

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode_fn(x: torch.Tensor, theta_enc: torch.Tensor, theta_ansatz: torch.Tensor):
        encoder.apply(x, theta_enc)
        apply_ansatz(theta_ansatz, n_qubits=n_qubits, depth=ansatz_depth)
        return qml.probs(wires=output_wires)

    return qnode_fn, n_output


def train_one(
    *,
    dataset_name: str,
    encoder_name: str,
    n_qubits: int,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    ansatz_depth: int,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_cls, n_features, n_classes = DATASET_REGISTRY[dataset_name]
    train_ds = dataset_cls(split="train")
    test_ds = dataset_cls(split="test")

    encoder = make_encoder(encoder_name, n_qubits=n_qubits, n_features=n_features)

    qnode, n_output_qubits = build_qnode(
        encoder,
        n_qubits=n_qubits,
        ansatz_depth=ansatz_depth,
        n_classes=n_classes,
    )

    n_enc = encoder.n_params
    n_ans = ansatz_n_params(n_qubits, ansatz_depth)

    theta_enc = torch.nn.Parameter(initial_encoder_params(n_enc, seed=seed))
    theta_ansatz = torch.nn.Parameter(
        0.1 * torch.randn(n_ans, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(seed + 1))
    )

    trainable = [theta_ansatz]
    if n_enc > 0:
        trainable.insert(0, theta_enc)

    opt = torch.optim.Adam(trainable, lr=lr)

    eps = 1e-12

    def forward_probs(x: torch.Tensor) -> torch.Tensor:
        probs = qnode(x.to(torch.float64), theta_enc, theta_ansatz)
        probs = torch.as_tensor(probs, dtype=torch.float32)
        probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
        probs = probs[:n_classes]
        probs = probs / (probs.sum() + 1e-12)
        return probs

    def cross_entropy(probs: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return -(y_onehot * torch.log(probs.clamp_min(eps))).sum()

    @torch.no_grad()
    def evaluate(ds) -> tuple[float, float]:
        losses = []
        correct = 0
        total = 0
        for x, y, _ in ds:
            p = forward_probs(x)
            losses.append(cross_entropy(p, y))
            correct += int(torch.argmax(p).item() == int(torch.argmax(y).item()))
            total += 1
        avg = float(torch.stack(losses).mean().item()) if losses else 0.0
        acc = float(correct / max(total, 1))
        return avg, acc

    # Mini-batch training. Plain shuffled SGD; good enough for this comparison.
    indices = list(range(len(train_ds)))
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for _epoch in range(epochs):
        rng.shuffle(indices)
        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            losses = []
            for i in batch_idx:
                x, y, _ = train_ds[i]
                p = forward_probs(x)
                losses.append(cross_entropy(p, y))
            loss = torch.stack(losses).mean()

            opt.zero_grad()
            loss.backward()

            # NaN-grad sanitizer (cheap insurance even noise-free).
            for tp in trainable:
                if tp.grad is not None and not torch.isfinite(tp.grad).all():
                    tp.grad = torch.where(
                        torch.isfinite(tp.grad),
                        tp.grad,
                        torch.zeros_like(tp.grad),
                    )
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()
    elapsed = time.time() - t0

    train_loss, train_acc = evaluate(train_ds)
    test_loss, test_acc = evaluate(test_ds)

    return {
        "dataset": dataset_name,
        "encoder": encoder_name,
        "n_qubits": n_qubits,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "n_encoder_params": n_enc,
        "n_ansatz_params": n_ans,
        "elapsed_s": elapsed,
    }


# ---------- driver ----------------------------------------------------------

CSV_COLUMNS = [
    "dataset", "encoder", "n_qubits", "seed", "epochs", "lr",
    "train_loss", "train_acc", "test_loss", "test_acc",
    "n_encoder_params", "n_ansatz_params", "elapsed_s",
]


def append_csv(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   choices=sorted(DATASET_REGISTRY.keys()),
                   default=["iris", "wine", "seeds", "breast_cancer"])
    p.add_argument("--encoders", nargs="+",
                   choices=sorted(ENCODERS.keys()),
                   default=["fixed_angle", "fixed_amplitude", "learned"])
    p.add_argument("--n_qubits", nargs="+", type=int, default=[4, 6, 8, 10])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ansatz_depth", type=int, default=2)
    p.add_argument("--out", type=str,
                   default="src/Ryan_cookin/results/stage_a.csv")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    n_runs = (
        len(args.datasets) * len(args.encoders) *
        len(args.n_qubits) * len(args.seeds)
    )
    print(f"Stage A: {n_runs} runs "
          f"(datasets={args.datasets}, encoders={args.encoders}, "
          f"n_qubits={args.n_qubits}, seeds={args.seeds}, epochs={args.epochs})")

    rows = []
    for dataset in args.datasets:
        for n in args.n_qubits:
            for encoder in args.encoders:
                for seed in args.seeds:
                    print(f"  -> {dataset:14s} N={n:2d} {encoder:16s} seed={seed} ", end="", flush=True)
                    row = train_one(
                        dataset_name=dataset,
                        encoder_name=encoder,
                        n_qubits=n,
                        seed=seed,
                        epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        ansatz_depth=args.ansatz_depth,
                    )
                    print(
                        f"test_acc={row['test_acc']:.3f} "
                        f"test_loss={row['test_loss']:.4f} "
                        f"({row['elapsed_s']:.1f}s)"
                    )
                    rows.append(row)
                    # Flush after every run so a long sweep doesn't lose
                    # progress on Ctrl-C.
                    append_csv(args.out, [row])

    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
