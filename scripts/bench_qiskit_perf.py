"""Benchmark the qiskit path of CircuitGenome against pennylane and against
candidate faster qiskit configurations.

Phases timed per variant and circuit size:
  build     -- generate_{qiskit,pennylane}_circuit() wall time
  forward   -- batched forward pass through the resulting torch model
  fwd+bwd   -- forward + loss.backward() (i.e. one training-step worth of grads)

Variants:
  pennylane    -- default.qubit, diff_method=backprop (reference / target speed)
  qiskit_ref   -- exactly what generate_qiskit_circuit() builds today
                  (reference Sampler + ParamShiftSamplerGradient)
  qiskit_aer   -- same circuit, SamplerQNN backed by qiskit-aer SamplerV2
                  (shot-based, 1024 shots -- NOT exact)
  qiskit_aer_exact -- same circuit, exact probabilities via batched Aer
                  statevector (scripts/aer_exact_sampler.py); drop-in,
                  same param-shift gradients, same results as qiskit_ref
  qiskit_spsa  -- same circuit, SPSASamplerGradient (O(1) circuits per grad,
                  stochastic; relevant for evolutionary loops)

Usage:
  python scripts/bench_qiskit_perf.py --sizes 4x2,6x2,8x2 --batch 8 --reps 3
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from loguru import logger  # noqa: E402

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.circuits.registers import expand_registers  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")


def build_genome(target: str, n_qubits: int, n_layers: int) -> CircuitGenome:
    """A deterministic layered ansatz: per layer, RY on every qubit followed
    by a CP entangling ring. Every rotation is a trainable parameter, so
    n_params = 2 * n_qubits * n_layers."""
    genome = CircuitGenome(
        genome_number=0,
        target=target,
        input_qubits=expand_registers({"a": n_qubits}),
    )
    genome.hyperparameters = {
        "quantum_input_mode": "rx",
        "quantum_output_mode": "probs",
    }

    depth = 0.0
    step = 1.0 / (2 * n_qubits * n_layers + 1)
    for _ in range(n_layers):
        for q in range(n_qubits):
            depth += step
            genome.add_gate(
                depth=depth,
                method_name="ry",
                qubits=[("a", q)],
                parameters={"theta": 0.1 * (q + 1)},
            )
        for q in range(n_qubits):
            depth += step
            genome.add_gate(
                depth=depth,
                method_name="cp",
                qubits=[("a", q), ("a", (q + 1) % n_qubits)],
                parameters={"theta": 0.05 * (q + 1)},
            )
    return genome


def time_reps(fn, reps: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def make_qiskit_variant_model(genome: CircuitGenome, variant: str):
    """Rebuild the SamplerQNN/TorchConnector around the circuit that
    generate_qiskit_circuit() assembled, swapping the runtime primitives."""
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.gradients import (
        ParamShiftSamplerGradient,
        SPSASamplerGradient,
    )
    from qiskit_machine_learning.neural_networks import SamplerQNN

    circuit = genome.qiskit_circuit

    if variant == "qiskit_aer":
        # NOTE: Aer SamplerV2 is shot-based (no exact-probability mode), so
        # this variant trades exactness for speed; 1024 shots is the usual
        # default. The reference sampler returns exact probabilities.
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

        sampler = AerSamplerV2(default_shots=1024)
        gradient = ParamShiftSamplerGradient(sampler=sampler)
    elif variant == "qiskit_aer_exact":
        from scripts.aer_exact_sampler import AerExactSampler

        sampler = AerExactSampler()
        gradient = ParamShiftSamplerGradient(sampler=sampler)
    elif variant == "qiskit_spsa":
        from qiskit_machine_learning.primitives import QMLSampler as Sampler

        sampler = Sampler()
        gradient = SPSASamplerGradient(sampler=sampler, epsilon=0.01, batch_size=1)
    else:
        raise ValueError(variant)

    qnn = SamplerQNN(
        circuit=circuit,
        sampler=sampler,
        gradient=gradient,
        input_params=genome.qiskit_input_vector,
        weight_params=genome.weight_vector,
        input_gradients=True,
        interpret=lambda x: x,
        output_shape=2**circuit.num_clbits,
    )
    return TorchConnector(qnn)


def bench_variant(
    variant: str,
    n_qubits: int,
    n_layers: int,
    batch: int,
    reps: int,
) -> list[dict]:
    target = "pennylane" if variant == "pennylane" else "qiskit"
    genome = build_genome(target, n_qubits, n_layers)
    n_params = len(genome.get_parameters_as_list())
    rows = []

    def label(phase, seconds):
        return {
            "variant": variant,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "n_params": n_params,
            "batch": batch,
            "phase": phase,
            "median_s": round(seconds, 6),
        }

    # -- build phase -------------------------------------------------------
    if variant == "pennylane":

        def do_build():
            genome.generate_pennylane_circuit()

    else:

        def do_build():
            # gate.qiskit_parameters is cached on first build; drop it so
            # every rep measures a cold build like evolution would see.
            for gate in genome.gates:
                if hasattr(gate, "qiskit_parameters"):
                    del gate.qiskit_parameters
            genome.generate_qiskit_circuit()

    rows.append(label("build", time_reps(do_build, reps)))

    model = genome.torch_model
    if variant in ("qiskit_aer", "qiskit_aer_exact", "qiskit_spsa"):
        model = make_qiskit_variant_model(genome, variant)

    x = torch.rand(batch, genome.n_quantum_inputs(), dtype=torch.float64)

    # -- forward (inference) ------------------------------------------------
    def do_forward():
        with torch.no_grad():
            model(x)

    rows.append(label("forward", time_reps(do_forward, reps)))

    # -- forward + backward (training step) ---------------------------------
    def do_fwd_bwd():
        model.zero_grad()
        out = model(x)
        out.sum().backward()

    rows.append(label("fwd+bwd", time_reps(do_fwd_bwd, reps)))

    return rows


def bench_register_construction(n_qubits: int, reps: int) -> list[dict]:
    """Isolate the cost of one single-qubit QuantumRegister per qubit
    (current implementation) vs one n-qubit register."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    def many_registers():
        regs = [QuantumRegister(1, name=f"a-{i}") for i in range(n_qubits)]
        QuantumCircuit(*regs, ClassicalRegister(n_qubits, name="c"))

    def one_register():
        QuantumCircuit(
            QuantumRegister(n_qubits, name="a"), ClassicalRegister(n_qubits, name="c")
        )

    return [
        {
            "variant": "construct_many_regs",
            "n_qubits": n_qubits,
            "n_layers": 0,
            "n_params": 0,
            "batch": 0,
            "phase": "build",
            "median_s": round(time_reps(many_registers, reps, warmup=2), 6),
        },
        {
            "variant": "construct_one_reg",
            "n_qubits": n_qubits,
            "n_layers": 0,
            "n_params": 0,
            "batch": 0,
            "phase": "build",
            "median_s": round(time_reps(one_register, reps, warmup=2), 6),
        },
    ]


def verify_aer_exact(n_qubits: int = 4, n_layers: int = 2, batch: int = 4):
    """Check that the AerExactSampler variant reproduces the reference
    implementation's forward outputs and weight gradients."""
    torch.manual_seed(0)
    genome = build_genome("qiskit", n_qubits, n_layers)
    genome.generate_qiskit_circuit()
    ref_model = genome.torch_model
    aer_model = make_qiskit_variant_model(genome, "qiskit_aer_exact")

    with torch.no_grad():
        aer_model.weight.copy_(ref_model.weight)

    x = torch.rand(batch, genome.n_quantum_inputs(), dtype=torch.float64)

    ref_out = ref_model(x)
    aer_out = aer_model(x)
    fwd_err = (ref_out - aer_out).abs().max().item()

    ref_model.zero_grad()
    ref_out.sum().backward()
    aer_model.zero_grad()
    aer_out.sum().backward()
    grad_err = (ref_model.weight.grad - aer_model.weight.grad).abs().max().item()

    print(f"verify qiskit_aer_exact: max |forward diff| = {fwd_err:.3e}")
    print(f"verify qiskit_aer_exact: max |weight-grad diff| = {grad_err:.3e}")
    ok = fwd_err < 1e-8 and grad_err < 1e-8
    print("verify:", "PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="4x2,6x2,8x2")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--variants",
        default="pennylane,qiskit_ref,qiskit_aer,qiskit_aer_exact,qiskit_spsa",
    )
    parser.add_argument("--out", default="bench_qiskit_perf.csv")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify and not verify_aer_exact():
        sys.exit(1)

    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    variants = args.variants.split(",")

    rows = []
    for n_qubits, n_layers in sizes:
        rows.extend(bench_register_construction(n_qubits, args.reps))
        for variant in variants:
            print(
                f"benchmarking {variant} n_qubits={n_qubits} n_layers={n_layers}...",
                flush=True,
            )
            try:
                rows.extend(
                    bench_variant(variant, n_qubits, n_layers, args.batch, args.reps)
                )
            except Exception as exc:  # keep the sweep alive on a bad variant
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    with open(args.out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {args.out}\n")
    header = f"{'variant':<22}{'qubits':>7}{'layers':>7}{'params':>7}{'phase':>9}{'median_s':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['variant']:<22}{row['n_qubits']:>7}{row['n_layers']:>7}"
            f"{row['n_params']:>7}{row['phase']:>9}{row['median_s']:>12.6f}"
        )


if __name__ == "__main__":
    main()
