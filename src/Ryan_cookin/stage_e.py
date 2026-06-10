"""Stage E: encoder structure + ansatz structure + N, ALL evolved.

The original Stage B plan, restored. A single CircuitGenome contains both
*encoder gates* (feature-dependent rotations like RY(a*x[i] + b)) and
*ansatz gates* (regular rotations and entanglers). Evolution mutates the
entire structure — which gates exist, where they sit in depth, which
qubits they touch — and the register size N is evolved via the
grow/shrink mutations Stage C added.

This is the most ambitious of the stages. Two design choices to be aware of:

1. Encoder gates are identified by method-name prefix `enc_`. We define
   three: enc_ry, enc_rx, enc_rz. Each has TWO trainable parameters (a, b)
   and applies `R<axis>(a * x[feature_index] + b)` on its qubit. The
   `feature_index` is implicit — it's the qubit's index modulo D, same
   convention LearnedAngleEncoder uses. We don't evolve the feature
   assignment, only the gate identity, depth, qubit, and parameters.

2. The pennylane op name on enc_* gate specs is a stub (`EncoderRY` etc.)
   — there's no `qml.EncoderRY`. Stage E's qnode body intercepts these
   gates by name and applies them with feature-dependent rotation.
   `Gate.add_to_pennylane_circuit` is NEVER called on enc_* gates in
   Stage E; calling it elsewhere would error, which is the intended
   behavior (loud failure if someone routes them through the wrong code
   path).

Why a single genome with both types instead of two paired genomes? Two
genomes would mean two independent EXAQC searches that have to agree on
N. One unified genome with depth-ordered gates handles that automatically
(encoder gates naturally fall early because their depth-uniform random
init tends to spread, and mutation reordering can rearrange).

Run:
    PYTHONPATH=. python -m src.Ryan_cookin.stage_e \
        --datasets iris wine seeds breast_cancer \
        --initial_n_qubits 4 --n_genomes 120 --pop_size 25

Output:
    src/Ryan_cookin/results/stage_e.csv
    src/Ryan_cookin/results/weights_stage_e/<cell>.pt
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
from src.circuits.gate_specifications import GateSpecification, GateSpecifications
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


DATASET_REGISTRY = {
    "iris":          (IrisDataset,         4,  3),
    "wine":          (WineDataset,        13,  3),
    "seeds":         (SeedsDataset,        7,  3),
    "breast_cancer": (BreastCancerDataset, 30, 2),
}


# ---------- new encoder gate specs ----------------------------------------
#
# These are added to the global pennylane_gate_specifications registry
# (because Gate.__init__ looks specs up there) and to a Stage E-specific
# curated pool that EXAQC mutates from. We do not provide real PennyLane
# op names — enc_* gates are intercepted by stage_e's qnode loop, never
# routed through Gate.add_to_pennylane_circuit.

def _register_encoder_gate_specs():
    """Idempotently add enc_ry, enc_rx, enc_rz specs to the global registry.

    Each has 2 trainable parameters (a, b). Single qubit. The pennylane_op
    field is a stub that intentionally would error if invoked — encoder
    gates must be handled by the Stage E qnode body, not by the generic
    add_to_pennylane_circuit path.
    """
    if "enc_ry" in pennylane_gate_specifications.specifications:
        return
    pennylane_gate_specifications["enc_ry"] = GateSpecification(
        name="Encoder RY (feature-dependent)",
        qubits=["qubit"],
        parameters=["a", "b"],
        pennylane_op="EncoderRY",  # stub; not a real PennyLane op
    )
    pennylane_gate_specifications["enc_rx"] = GateSpecification(
        name="Encoder RX (feature-dependent)",
        qubits=["qubit"],
        parameters=["a", "b"],
        pennylane_op="EncoderRX",
    )
    pennylane_gate_specifications["enc_rz"] = GateSpecification(
        name="Encoder RZ (feature-dependent)",
        qubits=["qubit"],
        parameters=["a", "b"],
        pennylane_op="EncoderRZ",
    )


def _build_stage_e_gate_specs() -> GateSpecifications:
    """A curated gate-spec pool for Stage E.

    The full `pennylane_gate_specifications` has 30+ gates. Mutating
    uniformly from that pool means each gate type sees < 4% of add_gate
    mutations, which dilutes the enc_* gates we care about. We assemble
    a smaller pool that gives encoder gates and the most useful ansatz
    primitives roughly equal weight.

    Pool:
        encoder side: enc_ry, enc_rx, enc_rz
        ansatz side: ry, rx, rz, cnot, cz
    """
    _register_encoder_gate_specs()
    return pennylane_gate_specifications.use_only(
        ["enc_ry", "enc_rx", "enc_rz", "ry", "rx", "rz", "cx", "cz"]
    )


# ---------- objective ------------------------------------------------------

def _compare_by_test_loss(a: CircuitGenome, b: CircuitGenome) -> float:
    return a.fitness["test_loss"] - b.fitness["test_loss"]


def _build_initial_qubits(initial_n_qubits: int, n_classes: int):
    n_output = max(1, math.ceil(math.log2(max(n_classes, 2))))
    if initial_n_qubits < n_output:
        raise ValueError(
            f"initial_n_qubits={initial_n_qubits} < n_output={n_output} "
            f"required for {n_classes} classes."
        )
    input_qubits = [("q", i) for i in range(initial_n_qubits)]
    output_qubits = [("q", i) for i in range(n_output)]
    return input_qubits, output_qubits, n_output


_ENC_AXIS_OP = {
    "enc_ry": qml.RY,
    "enc_rx": qml.RX,
    "enc_rz": qml.RZ,
}


class GenomeEncoderAnsatzObjective(Objective):
    """Fitness for a Stage E genome.

    No external encoder. The encoder *is* the enc_* gates inside the genome.
    The qnode applies all enabled gates in depth order: enc_* gates as
    feature-dependent rotations, other gates as their usual PennyLane ops.
    """

    def __init__(
        self,
        *,
        train_data,
        test_data,
        n_features: int,
        n_classes: int,
        epochs: int,
        lr: float,
        batch_size: int,
        seed: int,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.n_features = n_features
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.target = "pennylane"

    def _make_qnode(self, genome: CircuitGenome, n_qubits: int):
        dev = qml.device("default.qubit", wires=n_qubits)
        genome.sort_gates()
        gates_snapshot = list(genome.gates)
        qubits = genome.qubits
        output_indexes = list(genome.output_indexes)
        n_features = self.n_features

        # Pre-compute, for each enc_* gate: the qubit's wire index and its
        # feature index (q % D). This avoids repeating the lookup per
        # forward pass.
        enc_meta: dict[int, tuple[int, int, str]] = {}
        for gate in gates_snapshot:
            if gate.method_name in _ENC_AXIS_OP:
                wire = qubits.index(gate.qubits[0])
                feat_idx = wire % n_features
                enc_meta[id(gate)] = (wire, feat_idx, gate.method_name)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(x, params_dict):
            for gate in gates_snapshot:
                if not gate.enabled:
                    continue
                meta = enc_meta.get(id(gate))
                if meta is not None:
                    wire, feat_idx, mname = meta
                    a = params_dict[f"{gate.innovation_number}:a"]
                    b = params_dict[f"{gate.innovation_number}:b"]
                    angle = a * x[feat_idx] + b
                    _ENC_AXIS_OP[mname](angle, wires=wire)
                else:
                    gate.add_to_pennylane_circuit(qubits, params=params_dict)
            return qml.probs(wires=output_indexes)

        return circ

    def _forward(self, circ, x, params_dict, eps: float = 1e-12) -> torch.Tensor:
        probs = circ(x.to(torch.float64), params_dict)
        probs = torch.as_tensor(probs, dtype=torch.float32)
        probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
        probs = probs[: self.n_classes]
        probs = probs / (probs.sum() + 1e-12)
        return probs

    @staticmethod
    def _cross_entropy(probs: torch.Tensor, y_onehot: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return -(y_onehot * torch.log(probs.clamp_min(eps))).sum()

    def _evaluate(self, circ, ds, params_dict) -> tuple[float, float]:
        with torch.no_grad():
            losses = []
            correct = 0
            total = 0
            for x, y, _ in ds:
                p = self._forward(circ, x, params_dict)
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
        circ = self._make_qnode(genome, n_qubits)

        params_dict = genome_to_torch_params(genome)
        trainable = list(params_dict.values())

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
                        p = self._forward(circ, x, params_dict)
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

        train_loss, train_acc = self._evaluate(circ, self.train_data, params_dict)
        test_loss, test_acc = self._evaluate(circ, self.test_data, params_dict)

        genome.fitness = {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        genome.metadata["n_qubits"] = n_qubits

        n_enc_gates = sum(
            1 for g in genome.gates if g.enabled and g.method_name in _ENC_AXIS_OP
        )
        n_ansatz_gates = sum(
            1 for g in genome.gates if g.enabled and g.method_name not in _ENC_AXIS_OP
        )
        genome.metadata["n_enc_gates"] = n_enc_gates
        genome.metadata["n_ansatz_gates"] = n_ansatz_gates

        logger.info(
            f"[{genome.genome_number:04d}] N={n_qubits:2d} "
            f"enc={n_enc_gates:2d} ans={n_ansatz_gates:2d} "
            f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
            f"test_loss={test_loss:.4f}"
        )


# ---------- driver --------------------------------------------------------

def train_one(
    *,
    dataset_name: str,
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

    objective = GenomeEncoderAnsatzObjective(
        train_data=train_ds,
        test_data=test_ds,
        n_features=n_features,
        n_classes=n_classes,
        epochs=epochs_inner,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )

    input_qubits, output_qubits, n_output = _build_initial_qubits(
        initial_n_qubits, n_classes
    )

    profiler_scratch = tempfile.mkdtemp(prefix=f"exaqc_e_{dataset_name}_")
    profiler = EXAQCProfiler(out_dir=profiler_scratch)
    population = SteadyStatePopulation(
        max_population_size=pop_size,
        compare=_compare_by_test_loss,
        out_dir=None,
        profiler=profiler,
    )

    exaqc = EXAQC(
        gate_specifications=_build_stage_e_gate_specs(),
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
    n_enc = best.metadata.get("n_enc_gates", 0)
    n_ansatz = best.metadata.get("n_ansatz_gates", 0)

    if weights_dir is not None:
        os.makedirs(weights_dir, exist_ok=True)
        out_path = os.path.join(
            weights_dir, f"{dataset_name}_seed{seed}.pt",
        )
        torch.save(
            {
                "dataset": dataset_name,
                "initial_n_qubits": initial_n_qubits,
                "best_n_qubits": best_n_qubits,
                "n_features": n_features,
                "n_classes": n_classes,
                "seed": seed,
                "n_genomes": n_genomes,
                "pop_size": pop_size,
                "epochs_inner": epochs_inner,
                "best_genome_dict": best.to_dict(),
                "best_genome_number": best.genome_number,
                "n_enc_gates": n_enc,
                "n_ansatz_gates": n_ansatz,
                "train_loss": best.fitness["train_loss"],
                "train_acc": best.fitness["train_acc"],
                "test_loss": best.fitness["test_loss"],
                "test_acc": best.fitness["test_acc"],
            },
            out_path,
        )

    return {
        "dataset": dataset_name,
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
        "n_enc_gates": n_enc,
        "n_ansatz_gates": n_ansatz,
        "best_genome_number": best.genome_number,
        "elapsed_s": elapsed,
    }


CSV_COLUMNS = [
    "dataset", "seed",
    "initial_n_qubits", "best_n_qubits",
    "epochs_inner", "lr", "n_genomes", "pop_size",
    "train_loss", "train_acc", "test_loss", "test_acc",
    "n_enc_gates", "n_ansatz_gates",
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
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--initial_n_qubits", type=int, default=4)
    p.add_argument("--epochs_inner", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_genomes", type=int, default=120,
                   help="Bigger than Stage C's 80 because we evolve more (encoder structure + ansatz structure + N).")
    p.add_argument("--pop_size", type=int, default=25)
    p.add_argument("--out", type=str, default="src/Ryan_cookin/results/stage_e.csv")
    p.add_argument("--weights_dir", type=str, default="src/Ryan_cookin/results/weights_stage_e")
    p.add_argument("--logging_level", type=str, default="WARNING")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    n_runs = len(args.datasets) * len(args.seeds)
    print(f"Stage E: {n_runs} runs "
          f"(datasets={args.datasets}, seeds={args.seeds}, "
          f"initial_N={args.initial_n_qubits}, "
          f"n_genomes={args.n_genomes}, pop_size={args.pop_size}, "
          f"epochs_inner={args.epochs_inner})")

    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"  -> {dataset:14s} seed={seed} ", end="", flush=True)
            row = train_one(
                dataset_name=dataset,
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
                f"N->{row['best_n_qubits']} enc={row['n_enc_gates']} "
                f"ans={row['n_ansatz_gates']} ({row['elapsed_s']:.1f}s)"
            )
            rows.append(row)
            append_csv(args.out, [row])

    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
