"""Stage C: trainable encoder + EVOLVED ansatz + EVOLVED qubit count N.

Stage B fixed N per cell and evolved the ansatz structure. Stage C drops
the N sweep dimension and lets evolution decide how many qubits the
circuit should have. Each genome carries its own `qubits` / `input_qubits`
state, and two new mutation operators in src/evolution/mutation.py
(`grow_register`, `shrink_register`) make that part of the search.

Sweep dimensions: (dataset, encoder, seed). NO n_qubits axis. The
initial population starts from a small register (default N=4) and
evolution explores from there. The output CSV records each best
genome's *evolved* N.

Run:
    PYTHONPATH=. python -m src.Ryan_cookin.stage_c \
        --datasets iris wine seeds breast_cancer \
        --encoders fixed_angle fixed_amplitude fixed_basis learned \
        --initial_n_qubits 4 \
        --n_genomes 80 --pop_size 20 --epochs_inner 5

Output:
    src/Ryan_cookin/results/stage_c.csv
    src/Ryan_cookin/results/weights_stage_c/<cell>.pt

Implementation notes worth reading before extending:

* The encoder is *not* a fixed object held by the objective the way Stage
  B's was. We build it lazily per genome, using the genome's current
  `n_qubits = len(genome.qubits)`. This applies to every encoder we
  support: angle/amplitude/learned just re-instantiate; basis re-fits
  thresholds from the training set at the new N (cheap — fit_basis_thresholds
  just takes a median per feature).

* `theta_enc` is fresh per genome (no warm-start). When N changes, its
  shape changes, so carrying values across is awkward. Re-initializing
  is simpler and the inner training loop converges quickly anyway.

* Trained gate-parameter values on the genome ARE carried across via
  `torch_params_to_genome` (same as Stage B). Mutations that don't
  change gate identity (enable/disable, reorder, qubit_swap) keep
  warm-start values; grow/shrink only add or disable gates and don't
  reshape existing ones, so warm-start works there too.
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


def _compare_by_test_loss(a: CircuitGenome, b: CircuitGenome) -> float:
    return a.fitness["test_loss"] - b.fitness["test_loss"]


def _build_initial_qubits(initial_n_qubits: int, n_classes: int):
    """Same register layout as Stage B: single register "q", output is the
    first ceil(log2 K) wires. Differs only in that N here is the *starting*
    register size — evolution will grow/shrink from there.
    """
    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    if initial_n_qubits < n_output:
        raise ValueError(
            f"initial_n_qubits={initial_n_qubits} is less than n_output={n_output} "
            f"required for {n_classes} classes."
        )
    input_qubits = [("q", i) for i in range(initial_n_qubits)]
    output_qubits = [("q", i) for i in range(n_output)]
    return input_qubits, output_qubits, n_output


class EncoderAnsatzObjective(Objective):
    """Same shape as Stage B's objective, but the encoder is rebuilt per
    genome using whatever N the genome currently has. See module docstring
    for why theta_enc is fresh per genome.
    """

    def __init__(
        self,
        *,
        train_data,
        test_data,
        n_features: int,
        n_classes: int,
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
        self.encoder_name = encoder_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.target = "pennylane"

        self.n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))

    def _build_encoder(self, n_qubits: int):
        """Build a fresh encoder sized for the current N.

        Basis encoder needs thresholds fit on the training set at this N.
        Other encoders just take (n_qubits, n_features).
        """
        extras: dict = {}
        if self.encoder_name == "fixed_basis":
            extras["thresholds"] = fit_basis_thresholds(
                self.train_data, n_qubits=n_qubits, n_features=self.n_features,
            )
        return make_encoder(
            self.encoder_name,
            n_qubits=n_qubits,
            n_features=self.n_features,
            **extras,
        )

    def _make_qnode(self, genome: CircuitGenome, encoder, n_qubits: int):
        dev = qml.device("default.qubit", wires=n_qubits)
        genome.sort_gates()
        gates_snapshot = list(genome.gates)
        qubits = genome.qubits
        output_indexes = list(genome.output_indexes)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(x, theta_enc, params_dict):
            encoder.apply(x, theta_enc)
            for gate in gates_snapshot:
                if gate.enabled:
                    gate.add_to_pennylane_circuit(qubits, params=params_dict)
            return qml.probs(wires=output_indexes)

        return circ

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

    def __call__(self, genome: CircuitGenome) -> None:
        if not genome.is_valid():
            genome.fitness = {
                "train_loss": float("inf"),
                "train_acc": 0.0,
                "test_loss": float("inf"),
                "test_acc": 0.0,
            }
            return

        n_qubits = len(genome.qubits)
        encoder = self._build_encoder(n_qubits)
        circ = self._make_qnode(genome, encoder, n_qubits)

        if encoder.n_params > 0:
            theta_enc = torch.nn.Parameter(
                initial_encoder_params(encoder, seed=self.seed)
            )
        else:
            theta_enc = torch.zeros(0, dtype=torch.float64)

        params_dict = genome_to_torch_params(genome)
        trainable: list[torch.nn.Parameter] = []
        if encoder.n_params > 0:
            trainable.append(theta_enc)
        trainable.extend(params_dict.values())

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
                    for tp in trainable:
                        if tp.grad is not None and not torch.isfinite(tp.grad).all():
                            tp.grad = torch.where(
                                torch.isfinite(tp.grad),
                                tp.grad,
                                torch.zeros_like(tp.grad),
                            )
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    opt.step()

            torch_params_to_genome(genome, params_dict)

        train_loss, train_acc = self._evaluate(circ, self.train_data, theta_enc, params_dict)
        test_loss, test_acc = self._evaluate(circ, self.test_data, theta_enc, params_dict)

        genome.fitness = {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        genome.metadata["theta_enc"] = (
            theta_enc.detach().clone() if encoder.n_params > 0 else None
        )
        genome.metadata["n_qubits"] = n_qubits

        logger.info(
            f"[{genome.genome_number:04d}] N={n_qubits:2d} "
            f"gates={len(genome.gates):3d} params={len(params_dict):3d} "
            f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
            f"test_loss={test_loss:.4f}"
        )


# ---------- per-cell driver ------------------------------------------------

def train_one(
    *,
    dataset_name: str,
    encoder_name: str,
    seed: int,
    initial_n_qubits: int,
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
        encoder_name=encoder_name,
        epochs=epochs_inner,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )

    input_qubits, output_qubits, n_output = _build_initial_qubits(
        initial_n_qubits, n_classes
    )

    profiler_scratch = tempfile.mkdtemp(
        prefix=f"exaqc_c_{dataset_name}_{encoder_name}_"
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
        allow_register_evolution=True,
    )

    t0 = time.time()
    exaqc.run_for(n_genomes)
    elapsed = time.time() - t0

    best = population.get_best_genome()
    if best is None:
        raise RuntimeError("EXAQC produced no genomes for this cell")

    best_n_qubits = best.metadata.get("n_qubits", len(best.qubits))

    if weights_dir is not None:
        os.makedirs(weights_dir, exist_ok=True)
        out_path = os.path.join(
            weights_dir,
            f"{dataset_name}_{encoder_name}_seed{seed}.pt",
        )
        torch.save(
            {
                "dataset": dataset_name,
                "encoder": encoder_name,
                "initial_n_qubits": initial_n_qubits,
                "best_n_qubits": best_n_qubits,
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
    n_genome_params = sum(len(g.parameters) for g in best.gates if g.enabled)

    return {
        "dataset": dataset_name,
        "encoder": encoder_name,
        "seed": seed,
        "initial_n_qubits": initial_n_qubits,
        "best_n_qubits": best_n_qubits,
        "epochs_inner": epochs_inner,
        "lr": lr,
        "n_genomes": n_genomes,
        "pop_size": pop_size,
        "train_loss": best.fitness["train_loss"],
        "train_acc": best.fitness["train_acc"],
        "test_loss": best.fitness["test_loss"],
        "test_acc": best.fitness["test_acc"],
        "best_n_gates": n_enabled_gates,
        "best_n_genome_params": n_genome_params,
        "best_genome_number": best.genome_number,
        "elapsed_s": elapsed,
    }


# ---------- CSV / CLI ------------------------------------------------------

CSV_COLUMNS = [
    "dataset", "encoder", "seed",
    "initial_n_qubits", "best_n_qubits",
    "epochs_inner", "lr", "n_genomes", "pop_size",
    "train_loss", "train_acc", "test_loss", "test_acc",
    "best_n_gates", "best_n_genome_params",
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
                   default=["fixed_angle", "fixed_amplitude", "fixed_basis", "learned"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--initial_n_qubits", type=int, default=4,
                   help="Starting register size. Evolution can grow/shrink "
                        "from here. Must be >= ceil(log2 K) for the dataset.")
    p.add_argument("--epochs_inner", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_genomes", type=int, default=80,
                   help="Bigger than Stage B's 50 because the search space "
                        "now also includes N.")
    p.add_argument("--pop_size", type=int, default=20)
    p.add_argument("--out", type=str,
                   default="src/Ryan_cookin/results/stage_c.csv")
    p.add_argument("--weights_dir", type=str,
                   default="src/Ryan_cookin/results/weights_stage_c")
    p.add_argument("--logging_level", type=str, default="WARNING")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    n_runs = len(args.datasets) * len(args.encoders) * len(args.seeds)
    print(f"Stage C: {n_runs} runs "
          f"(datasets={args.datasets}, encoders={args.encoders}, "
          f"seeds={args.seeds}, initial_N={args.initial_n_qubits}, "
          f"n_genomes={args.n_genomes}, pop_size={args.pop_size}, "
          f"epochs_inner={args.epochs_inner})")

    rows = []
    for dataset in args.datasets:
        for encoder in args.encoders:
            for seed in args.seeds:
                print(
                    f"  -> {dataset:14s} {encoder:16s} seed={seed} ",
                    end="", flush=True,
                )
                row = train_one(
                    dataset_name=dataset,
                    encoder_name=encoder,
                    seed=seed,
                    initial_n_qubits=args.initial_n_qubits,
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
                    f"N->{row['best_n_qubits']} gates={row['best_n_gates']} "
                    f"({row['elapsed_s']:.1f}s)"
                )
                rows.append(row)
                append_csv(args.out, [row])

    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
