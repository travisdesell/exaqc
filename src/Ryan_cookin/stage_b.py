"""Stage B: trainable encoder + EVOLVED ansatz, swept over N and datasets.

Stage A used a hand-built post-encoder ansatz (per-qubit RY + linear CNOT
chain, depth=2, identical for every cell). Stage B replaces that fixed
ansatz with an evolved CircuitGenome searched by the repo's EXAQC
machinery. Same encoders, same datasets, same N sweep, same CSV schema
plus a few evolution-specific columns so Stage A and Stage B numbers
sit side by side.

Run:
    PYTHONPATH=. python -m src.Ryan_cookin.stage_b \
        --datasets iris wine \
        --encoders fixed_angle fixed_amplitude learned \
        --n_qubits 4 6 8 \
        --n_genomes 50 --pop_size 20 --epochs_inner 5

Output:
    src/Ryan_cookin/results/stage_b.csv          one row per cell
    <weights_dir>/<cell>.pt                      best genome + thetas

Sanity: this isolates "structure search" as the only difference from
Stage A. Encoder + inner-loop training are unchanged in form.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import time
from typing import Iterable

import numpy as np
import pennylane as qml
import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.datasets.classification import (
    BreastCancerDataset,
    IrisDataset,
    SeedsDataset,
    WineDataset,
)
from src.evolution.exaqc import EXAQC
from src.evolution.objective import Objective
from src.evolution.steady_state_population import SteadyStatePopulation
from src.utils.helpers import genome_to_torch_params, torch_params_to_genome
from src.utils.profiler import EXAQCProfiler
from src.Ryan_cookin.encoders import (
    ENCODERS,
    fit_basis_thresholds,
    initial_encoder_params,
    make_encoder,
)


DATASET_REGISTRY = {
    "iris":          (IrisDataset,         4,  3),
    "wine":          (WineDataset,        13,  3),
    "seeds":         (SeedsDataset,        7,  3),
    "breast_cancer": (BreastCancerDataset, 30, 2),
}


# ---------- objective ------------------------------------------------------

def _compare_by_test_loss(a: CircuitGenome, b: CircuitGenome) -> float:
    """SteadyStatePopulation sorts genomes via this — lower test_loss is better."""
    return a.fitness["test_loss"] - b.fitness["test_loss"]


def _build_genome_qubits(n_qubits: int, n_classes: int):
    """Single register "q" with N wires. Output = first ceil(log2(K)) wires.

    Matches Stage A's geometry exactly: encoder + ansatz both touch all N
    wires, measurement on the first ceil(log2(K)).
    """
    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    input_qubits = [("q", i) for i in range(n_qubits)]
    output_qubits = [("q", i) for i in range(n_output)]
    return input_qubits, output_qubits, n_output


class EncoderAnsatzObjective(Objective):
    """Fitness = train the (encoder + evolved-ansatz) joint circuit, eval test loss/acc.

    Each call:
      1. Builds a fresh QNode that applies the fixed-shape encoder, then the
         genome's enabled gates in depth order, then qml.probs on output wires.
      2. Adam-trains encoder params (if any) and genome params for `epochs`
         epochs, mini-batched.
      3. Evaluates train + test, sets `genome.fitness` dict.

    The genome's *structure* is what evolves; its gate parameter VALUES are
    re-initialized from the gates' stored values and then trained inside the
    objective. After training, the trained values are written back to the
    genome (via torch_params_to_genome) so that mutation/crossover children
    inherit them as a warm start.
    """

    def __init__(
        self,
        *,
        train_data,
        test_data,
        n_features: int,
        n_classes: int,
        n_qubits: int,
        encoder_name: str,
        epochs: int,
        lr: float,
        batch_size: int,
        seed: int,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.encoder_name = encoder_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.target = "pennylane"

        # The fixed encoder shape stays constant across all genomes in a cell.
        # Basis encoding needs per-feature thresholds fitted on the train split
        # before the encoder is built; for other encoders this is a no-op.
        encoder_extras: dict = {}
        if encoder_name == "fixed_basis":
            encoder_extras["thresholds"] = fit_basis_thresholds(
                train_data, n_qubits=n_qubits, n_features=n_features,
            )
        self.encoder = make_encoder(
            encoder_name,
            n_qubits=n_qubits,
            n_features=n_features,
            **encoder_extras,
        )
        self.n_encoder_params = self.encoder.n_params

        # n_output is a property of the cell, not the genome.
        self.n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
        self.output_indexes = list(range(self.n_output))

    # ---- qnode wrapping ----

    def _make_qnode(self, genome: CircuitGenome):
        """Build a torch-interfaced qnode for a specific genome.

        The qnode closes over (encoder, genome, output_indexes). Inputs:
            x:           feature vector
            theta_enc:   encoder params (empty tensor if encoder has none)
            params_dict: dict[str, tensor] keyed by "{inn_num}:{param_name}"
        """
        dev = qml.device("default.qubit", wires=self.n_qubits)
        genome.sort_gates()
        gates_snapshot = list(genome.gates)  # frozen for this evaluation
        qubits = genome.qubits
        output_indexes = self.output_indexes

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(x, theta_enc, params_dict):
            self.encoder.apply(x, theta_enc)
            for gate in gates_snapshot:
                if gate.enabled:
                    gate.add_to_pennylane_circuit(qubits, params=params_dict)
            return qml.probs(wires=output_indexes)

        return circ

    # ---- helpers ----

    def _forward(self, circ, x, theta_enc, params_dict, eps: float = 1e-12) -> torch.Tensor:
        probs = circ(x.to(torch.float64), theta_enc, params_dict)
        probs = torch.as_tensor(probs, dtype=torch.float32)
        probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
        probs = probs[: self.n_classes]
        probs = probs / (probs.sum() + 1e-12)
        return probs

    @staticmethod
    def _cross_entropy(probs: torch.Tensor, y_onehot: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return -(y_onehot * torch.log(probs.clamp_min(eps))).sum()

    def _evaluate(self, circ, ds, theta_enc, params_dict) -> tuple[float, float]:
        with torch.no_grad():
            losses = []
            correct = 0
            total = 0
            for x, y, _ in ds:
                p = self._forward(circ, x, theta_enc, params_dict)
                losses.append(self._cross_entropy(p, y))
                correct += int(torch.argmax(p).item() == int(torch.argmax(y).item()))
                total += 1
            avg = float(torch.stack(losses).mean().item()) if losses else 0.0
            acc = float(correct / max(total, 1))
        return avg, acc

    # ---- the actual fitness call ----

    def __call__(self, genome: CircuitGenome) -> None:
        # Empty / unreachable genomes can happen during evolution. Validity is
        # checked by EXAQC.generate_genome, but be defensive.
        if not genome.is_valid():
            genome.fitness = {
                "train_loss": float("inf"),
                "train_acc": 0.0,
                "test_loss": float("inf"),
                "test_acc": 0.0,
            }
            return

        # Build qnode. theta_enc + genome params are the trainables.
        circ = self._make_qnode(genome)

        # Encoder params
        if self.n_encoder_params > 0:
            theta_enc = torch.nn.Parameter(
                initial_encoder_params(self.encoder, seed=self.seed)
            )
        else:
            theta_enc = torch.zeros(0, dtype=torch.float64)

        # Genome params (keyed dict of torch.nn.Parameter)
        params_dict = genome_to_torch_params(genome)
        # Trainables to feed Adam
        trainable: list[torch.nn.Parameter] = []
        if self.n_encoder_params > 0:
            trainable.append(theta_enc)
        trainable.extend(params_dict.values())

        # Edge case: genome may have zero trainable params (e.g., only CNOTs).
        # Skip optimizer in that case but still need to evaluate the fixed
        # encoder + structure on the test set.
        if len(trainable) > 0:
            opt = torch.optim.Adam(trainable, lr=self.lr)
            indices = list(range(len(self.train_data)))
            rng = np.random.default_rng(self.seed + genome.genome_number)

            for _epoch in range(self.epochs):
                rng.shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    batch_idx = indices[start : start + self.batch_size]
                    losses = []
                    for i in batch_idx:
                        x, y, _ = self.train_data[i]
                        p = self._forward(circ, x, theta_enc, params_dict)
                        losses.append(self._cross_entropy(p, y))
                    loss = torch.stack(losses).mean()

                    opt.zero_grad()
                    loss.backward()

                    # NaN sanitizer; same defensive pattern as stage_a.py.
                    for tp in trainable:
                        if tp.grad is not None and not torch.isfinite(tp.grad).all():
                            tp.grad = torch.where(
                                torch.isfinite(tp.grad),
                                tp.grad,
                                torch.zeros_like(tp.grad),
                            )
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    opt.step()

            # Write trained genome params back to the genome so children inherit.
            torch_params_to_genome(genome, params_dict)

        train_loss, train_acc = self._evaluate(circ, self.train_data, theta_enc, params_dict)
        test_loss, test_acc = self._evaluate(circ, self.test_data, theta_enc, params_dict)

        genome.fitness = {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        # Stash the trained encoder params on the genome's metadata so we can
        # recover them when we save the best-of-run weights.
        genome.metadata["theta_enc"] = theta_enc.detach().clone() if self.n_encoder_params > 0 else None

        logger.info(
            f"[{genome.genome_number:04d}] gates={len(genome.gates):3d} "
            f"params={len(params_dict):3d} "
            f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
            f"test_loss={test_loss:.4f}"
        )


# ---------- per-cell driver ------------------------------------------------

def train_one(
    *,
    dataset_name: str,
    encoder_name: str,
    n_qubits: int,
    seed: int,
    epochs_inner: int,
    lr: float,
    batch_size: int,
    n_genomes: int,
    pop_size: int,
    weights_dir: str | None = None,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random as pyrand
    pyrand.seed(seed)

    dataset_cls, n_features, n_classes = DATASET_REGISTRY[dataset_name]
    train_ds = dataset_cls(split="train")
    test_ds = dataset_cls(split="test")

    objective = EncoderAnsatzObjective(
        train_data=train_ds,
        test_data=test_ds,
        n_features=n_features,
        n_classes=n_classes,
        n_qubits=n_qubits,
        encoder_name=encoder_name,
        epochs=epochs_inner,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )

    input_qubits, output_qubits, n_output = _build_genome_qubits(n_qubits, n_classes)

    # out_dir=None on the population disables the per-genome JSON/PNG dumps
    # (skipped by gating in SteadyStatePopulation.insert_genome). The profiler
    # however unconditionally needs a real path, so give it a scratch dir.
    profiler_scratch = tempfile.mkdtemp(
        prefix=f"exaqc_b_{dataset_name}_N{n_qubits}_{encoder_name}_"
    )
    profiler = EXAQCProfiler(out_dir=profiler_scratch)
    population = SteadyStatePopulation(
        max_population_size=pop_size,
        compare=_compare_by_test_loss,
        out_dir=None,
        profiler=profiler,
    )

    exaqc = EXAQC(
        gate_specifications=pennylane_gate_specifications,
        population=population,
        objective=objective,
        hyperparameters={
            "epochs": epochs_inner,
            "learning_rate": lr,
            "log_every": 1000,
            "batch_size": batch_size,
            "encoding": "angle",
        },
        input_qubits=input_qubits,
        output_qubits=output_qubits,
        target="pennylane",
    )

    t0 = time.time()
    exaqc.run_for(n_genomes)
    elapsed = time.time() - t0

    best = population.get_best_genome()
    if best is None:
        raise RuntimeError("EXAQC produced no genomes for this cell")

    # Save best genome + its trained encoder params, for rendering / reproducibility.
    if weights_dir is not None:
        os.makedirs(weights_dir, exist_ok=True)
        out_path = os.path.join(
            weights_dir,
            f"{dataset_name}_N{n_qubits}_{encoder_name}_seed{seed}.pt",
        )
        torch.save(
            {
                "dataset": dataset_name,
                "encoder": encoder_name,
                "n_qubits": n_qubits,
                "n_features": n_features,
                "n_classes": n_classes,
                "seed": seed,
                "n_genomes": n_genomes,
                "pop_size": pop_size,
                "epochs_inner": epochs_inner,
                "theta_enc": best.metadata.get("theta_enc"),
                "best_genome_dict": best.to_dict(),
                "best_genome_number": best.genome_number,
                "train_loss": best.fitness["train_loss"],
                "train_acc": best.fitness["train_acc"],
                "test_loss": best.fitness["test_loss"],
                "test_acc": best.fitness["test_acc"],
            },
            out_path,
        )

    n_enabled_gates = sum(1 for g in best.gates if g.enabled)
    n_genome_params = sum(
        len(g.parameters) for g in best.gates if g.enabled
    )

    return {
        "dataset": dataset_name,
        "encoder": encoder_name,
        "n_qubits": n_qubits,
        "seed": seed,
        "epochs_inner": epochs_inner,
        "lr": lr,
        "n_genomes": n_genomes,
        "pop_size": pop_size,
        "train_loss": best.fitness["train_loss"],
        "train_acc": best.fitness["train_acc"],
        "test_loss": best.fitness["test_loss"],
        "test_acc": best.fitness["test_acc"],
        "n_encoder_params": objective.n_encoder_params,
        "best_n_gates": n_enabled_gates,
        "best_n_genome_params": n_genome_params,
        "best_genome_number": best.genome_number,
        "elapsed_s": elapsed,
    }


# ---------- CSV / CLI ------------------------------------------------------

CSV_COLUMNS = [
    "dataset", "encoder", "n_qubits", "seed",
    "epochs_inner", "lr", "n_genomes", "pop_size",
    "train_loss", "train_acc", "test_loss", "test_acc",
    "n_encoder_params", "best_n_gates", "best_n_genome_params",
    "best_genome_number", "elapsed_s",
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
    p.add_argument("--epochs_inner", type=int, default=5,
                   help="Adam epochs per genome inside the objective. Lower than "
                        "Stage A's 30 because we run many genomes per cell.")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_genomes", type=int, default=50,
                   help="Number of evolved genomes to evaluate per cell.")
    p.add_argument("--pop_size", type=int, default=20,
                   help="Steady-state population size per cell.")
    p.add_argument("--out", type=str,
                   default="src/Ryan_cookin/results/stage_b.csv")
    p.add_argument("--weights_dir", type=str,
                   default="src/Ryan_cookin/results/weights_stage_b",
                   help="Per-cell best genome + theta_enc are saved here.")
    p.add_argument("--logging_level", type=str, default="WARNING")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    n_runs = (
        len(args.datasets) * len(args.encoders) *
        len(args.n_qubits) * len(args.seeds)
    )
    print(f"Stage B: {n_runs} runs "
          f"(datasets={args.datasets}, encoders={args.encoders}, "
          f"n_qubits={args.n_qubits}, seeds={args.seeds}, "
          f"n_genomes={args.n_genomes}, pop_size={args.pop_size}, "
          f"epochs_inner={args.epochs_inner})")

    rows = []
    for dataset in args.datasets:
        for n in args.n_qubits:
            for encoder in args.encoders:
                for seed in args.seeds:
                    print(
                        f"  -> {dataset:14s} N={n:2d} {encoder:16s} "
                        f"seed={seed} ", end="", flush=True,
                    )
                    row = train_one(
                        dataset_name=dataset,
                        encoder_name=encoder,
                        n_qubits=n,
                        seed=seed,
                        epochs_inner=args.epochs_inner,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        n_genomes=args.n_genomes,
                        pop_size=args.pop_size,
                        weights_dir=args.weights_dir,
                    )
                    print(
                        f"test_acc={row['test_acc']:.3f} "
                        f"test_loss={row['test_loss']:.4f} "
                        f"gates={row['best_n_gates']} "
                        f"({row['elapsed_s']:.1f}s)"
                    )
                    rows.append(row)
                    append_csv(args.out, [row])

    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
