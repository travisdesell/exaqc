"""Stage F: universal-gate-set co-evolution. Two variants.

Background: Stage E lost to staged Stages B/C because its encoder gates
(`enc_ry/rx/rz`) were each single-axis, single-feature, only 2 trainable
params. Even with the universal-1-qubit identity Rot = Rz(g) Ry(b) Rz(a)
in principle, evolution rarely composed the right 3-gate sequence.

Stage F packages universality into single primitives so evolution can
place ONE gate to get full SU(2) capacity:

    enc_rot(q):  Rot(a_alpha*x[q%D] + b_alpha,
                    a_beta *x[q%D] + b_beta,
                    a_gamma*x[q%D] + b_gamma)
                  on qubit q -- universal 1-qubit feature-dep, 6 params.

Two variants choose what 2-qubit primitives the genome can place:

  v1 (hybrid):
      - enc_rot for 1-qubit feature-dep encoding (only)
      - rxx, ryy, rzz as TRAINABLE-SCALAR entanglers (no feature dep)
      - ry, rx, rz for plain 1-qubit ansatz rotations
      - cx, cz for free entanglers

    Param cost per encoder: 6 (constant in D).

  v2 (full feature-dep):
      - enc_rot for 1-qubit (same)
      - enc_xx(q0, q1):  IsingXX(a_q0*x[q0%D] + a_q1*x[q1%D] + b),
        analogously for enc_yy, enc_zz -- 3 params each, 2-qubit
        feature-dep
      - ry, rx, rz, cx, cz still in the pool for flexibility

    Param cost per 2-qubit feature-dep: 3 (still constant in D --
    each angle reads two feature components).

Both variants exercise EXAQC's structural search + register evolution
exactly the same way as Stage E. Direct comparison answers:
  - v1 vs Stage E: does universal-1-qubit-in-one-gate help?
  - v2 vs v1: does feature-dep entangling help on top?
  - v1/v2 vs Stage B + reupload_euler: does evolving the universal-gate
    placement beat a hand-built reuploading encoder?

Run:
    PYTHONPATH=. python -m src.Ryan_cookin.stage_f \
        --datasets iris wine seeds breast_cancer \
        --variant v1 --seeds 0 \
        --initial_n_qubits 4 --n_genomes 120 --pop_size 25

Output:
    src/Ryan_cookin/results/stage_f_<variant>.csv
    src/Ryan_cookin/results/weights_stage_f_<variant>/<cell>.pt
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
from src.utils.helpers import (
    GATE_COMPLEXITY,
    genome_to_torch_params,
    torch_params_to_genome,
)
from src.utils.profiler import EXAQCProfiler


DATASET_REGISTRY = {
    "iris":          (IrisDataset,         4,  3),
    "wine":          (WineDataset,        13,  3),
    "seeds":         (SeedsDataset,        7,  3),
    "breast_cancer": (BreastCancerDataset, 30, 2),
}


# ---------- new universal-encoder gate specs ------------------------------

ENC_ROT_PARAMS = ["a_alpha", "b_alpha", "a_beta", "b_beta", "a_gamma", "b_gamma"]
ENC_2Q_PARAMS = ["a_q0", "a_q1", "b"]


def _register_stage_f_specs():
    """Idempotently add enc_rot + enc_xx/yy/zz specs to the global registry.

    The pennylane_op field is a stub that would error if invoked through
    the generic Gate.add_to_pennylane_circuit path. Stage F's qnode body
    intercepts these gates by method name and applies the feature-
    dependent operation manually. GATE_COMPLEXITY entries are added so
    the profiler doesn't crash on them.
    """
    if "enc_rot" not in pennylane_gate_specifications.specifications:
        pennylane_gate_specifications["enc_rot"] = GateSpecification(
            name="Encoder Rot (universal 1-qubit, feature-dep)",
            qubits=["qubit"],
            parameters=ENC_ROT_PARAMS,
            pennylane_op="EncoderRot",  # stub
        )
    for axis in ("xx", "yy", "zz"):
        key = f"enc_{axis}"
        if key not in pennylane_gate_specifications.specifications:
            pennylane_gate_specifications[key] = GateSpecification(
                name=f"Encoder R{axis.upper()} (2-qubit feature-dep)",
                qubits=["qubit1", "qubit2"],
                parameters=ENC_2Q_PARAMS,
                pennylane_op=f"EncoderR{axis.upper()}",  # stub
            )

    # Profiler needs complexity entries for any gate it might see.
    for key in ("enc_rot", "enc_xx", "enc_yy", "enc_zz"):
        if key not in GATE_COMPLEXITY:
            if key == "enc_rot":
                GATE_COMPLEXITY[key] = {"gate_count": 1, "cnot_count": 0, "rot_count": 3}
            else:
                GATE_COMPLEXITY[key] = {"gate_count": 1, "cnot_count": 2, "rot_count": 1}


def _build_stage_f_gate_specs(variant: str) -> GateSpecifications:
    """Curated gate pool per variant. Both pools have 9 gates."""
    _register_stage_f_specs()
    if variant == "v1":
        pool = ["enc_rot", "ry", "rx", "rz", "rxx", "ryy", "rzz", "cx", "cz"]
    elif variant == "v2":
        pool = ["enc_rot", "enc_xx", "enc_yy", "enc_zz",
                "ry", "rx", "rz", "cx", "cz"]
    else:
        raise ValueError(f"unknown Stage F variant: {variant}")
    return pennylane_gate_specifications.use_only(pool)


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


# Method names recognized as "encoder gates" -- applied via the qnode
# interceptor rather than Gate.add_to_pennylane_circuit. _ENC_2Q maps
# the 2-qubit ones to their underlying PennyLane Ising op.
_ENC_1Q = {"enc_rot"}
_ENC_2Q = {
    "enc_xx": qml.IsingXX,
    "enc_yy": qml.IsingYY,
    "enc_zz": qml.IsingZZ,
}
_ENC_ALL = _ENC_1Q | set(_ENC_2Q.keys())


class StageFObjective(Objective):
    """Fitness for a Stage F genome (v1 or v2).

    Same overall shape as Stage E's objective: no external encoder, all
    encoder behavior lives inside the genome via enc_* gates. The qnode
    applies enc_rot and enc_xx/yy/zz with feature-dependent
    computations; all other gates go through Gate.add_to_pennylane_circuit.
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

        # Precompute the wire indexes + feature indexes for each enc_*
        # gate so the per-forward loop doesn't redo the lookup.
        #   enc_rot:        (w0, feat0)
        #   enc_xx/yy/zz:   (w0, w1, feat0, feat1, axis_op)
        enc1_meta: dict[int, tuple[int, int]] = {}
        enc2_meta: dict[int, tuple[int, int, int, int, object]] = {}
        for gate in gates_snapshot:
            if gate.method_name == "enc_rot":
                w0 = qubits.index(gate.qubits[0])
                enc1_meta[id(gate)] = (w0, w0 % n_features)
            elif gate.method_name in _ENC_2Q:
                w0 = qubits.index(gate.qubits[0])
                w1 = qubits.index(gate.qubits[1])
                enc2_meta[id(gate)] = (
                    w0, w1, w0 % n_features, w1 % n_features,
                    _ENC_2Q[gate.method_name],
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
                    qml.Rot(
                        a_a * xi + b_a,
                        a_b * xi + b_b,
                        a_g * xi + b_g,
                        wires=wire,
                    )
                    continue
                m2 = enc2_meta.get(id(gate))
                if m2 is not None:
                    w0, w1, f0, f1, axis_op = m2
                    a0 = params_dict[f"{inn}:a_q0"]
                    a1 = params_dict[f"{inn}:a_q1"]
                    b  = params_dict[f"{inn}:b"]
                    angle = a0 * x[f0] + a1 * x[f1] + b
                    axis_op(angle, wires=[w0, w1])
                    continue
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
    def _cross_entropy(probs, y_onehot, eps: float = 1e-12) -> torch.Tensor:
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
            1 for g in genome.gates if g.enabled and g.method_name in _ENC_ALL
        )
        n_ansatz_gates = sum(
            1 for g in genome.gates if g.enabled and g.method_name not in _ENC_ALL
        )
        genome.metadata["n_enc_gates"] = n_enc_gates
        genome.metadata["n_ansatz_gates"] = n_ansatz_gates


# ---------- driver --------------------------------------------------------

def train_one(
    *,
    dataset_name: str,
    seed: int,
    variant: str,
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

    objective = StageFObjective(
        train_data=train_ds,
        test_data=test_ds,
        n_features=n_features,
        n_classes=n_classes,
        epochs=epochs_inner,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )

    input_qubits, output_qubits, _ = _build_initial_qubits(
        initial_n_qubits, n_classes
    )

    profiler_scratch = tempfile.mkdtemp(prefix=f"exaqc_f_{variant}_{dataset_name}_")
    profiler = EXAQCProfiler(out_dir=profiler_scratch)
    population = SteadyStatePopulation(
        max_population_size=pop_size,
        compare=_compare_by_test_loss,
        out_dir=None,
        profiler=profiler,
    )

    exaqc = EXAQC(
        gate_specifications=_build_stage_f_gate_specs(variant),
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
                "variant": variant,
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
        "variant": variant,
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
    "dataset", "variant", "seed",
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
    p.add_argument("--variant", choices=["v1", "v2"], required=True,
                   help="v1 = hybrid (enc_rot + scalar 2q); "
                        "v2 = full feature-dep (enc_rot + enc_xx/yy/zz)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--initial_n_qubits", type=int, default=4)
    p.add_argument("--epochs_inner", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_genomes", type=int, default=120,
                   help="Matches Stage E's default for direct comparison.")
    p.add_argument("--pop_size", type=int, default=25)
    p.add_argument("--out", type=str, default=None,
                   help="Default: src/Ryan_cookin/results/stage_f_<variant>.csv")
    p.add_argument("--weights_dir", type=str, default=None,
                   help="Default: src/Ryan_cookin/results/weights_stage_f_<variant>")
    p.add_argument("--logging_level", type=str, default="WARNING")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    if args.out is None:
        args.out = f"src/Ryan_cookin/results/stage_f_{args.variant}.csv"
    if args.weights_dir is None:
        args.weights_dir = f"src/Ryan_cookin/results/weights_stage_f_{args.variant}"

    n_runs = len(args.datasets) * len(args.seeds)
    print(f"Stage F ({args.variant}): {n_runs} runs "
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
                variant=args.variant,
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
                f"N->{row['best_n_qubits']} "
                f"enc={row['n_enc_gates']} ans={row['n_ansatz_gates']} "
                f"({row['elapsed_s']:.1f}s)"
            )
            rows.append(row)
            append_csv(args.out, [row])

    print()
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
