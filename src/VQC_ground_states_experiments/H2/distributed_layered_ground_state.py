import argparse
import csv
from pathlib import Path

from mpi4py import MPI
import pennylane as qml
from pennylane import numpy as np


def optimize_layer_count(
    hamiltonian,
    n_qubits,
    initial_state,
    layers,
    steps=300,
    stepsize=0.1,
    seed=0,
):
    """Optimize one fixed-depth VQC for a supplied Hamiltonian."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(params):
        qml.BasisState(initial_state, wires=range(n_qubits))

        for layer in range(layers):
            for qubit in range(n_qubits):
                qml.RX(params[layer, qubit, 0], wires=qubit)
                qml.RY(params[layer, qubit, 1], wires=qubit)
                qml.RZ(params[layer, qubit, 2], wires=qubit)

            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return qml.expval(hamiltonian)

    # Reproducible per layer count, independent of MPI rank assignment.
    np.random.seed(seed + layers)

    params = np.random.rand(
        layers,
        n_qubits,
        3,
        requires_grad=True,
    )

    optimizer = qml.GradientDescentOptimizer(stepsize=stepsize)
    energy_history = []

    for _ in range(steps):
        params = optimizer.step(circuit, params)
        energy_history.append(float(circuit(params)))

    minimum_energy = min(energy_history)
    minimum_step = energy_history.index(minimum_energy) + 1

    return {
        "layers": layers,
        "final_energy": energy_history[-1],
        "minimum_energy": minimum_energy,
        "minimum_step": minimum_step,
        "parameters": 3 * n_qubits * layers,
    }


def run_distributed_layer_sweep(
    hamiltonian,
    n_qubits,
    initial_state,
    max_layers,
    steps=300,
    stepsize=0.1,
    seed=0,
    comm=MPI.COMM_WORLD,
):
    """Distribute independent layer-count experiments across MPI ranks."""
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    assigned_layers = list(range(rank + 1, max_layers + 1, world_size))

    print(
        f"[rank {rank}/{world_size}] assigned layers: {assigned_layers}",
        flush=True,
    )

    local_results = []

    for layers in assigned_layers:
        print(f"[rank {rank}] optimizing {layers} layer(s)", flush=True)

        result = optimize_layer_count(
            hamiltonian=hamiltonian,
            n_qubits=n_qubits,
            initial_state=initial_state,
            layers=layers,
            steps=steps,
            stepsize=stepsize,
            seed=seed,
        )

        local_results.append(result)

        print(
            f"[rank {rank}] layers={layers} "
            f"minimum_energy={result['minimum_energy']:.10f}",
            flush=True,
        )

    gathered_results = comm.gather(local_results, root=0)

    if rank != 0:
        return None

    results = [
        result for worker_results in gathered_results for result in worker_results
    ]
    results.sort(key=lambda result: result["layers"])
    return results


def load_h2_problem(comm=MPI.COMM_WORLD):
    """Load H2 on rank 0, then broadcast Hamiltonian and HF state."""
    rank = comm.Get_rank()

    hamiltonian = None
    initial_state = None

    if rank == 0:
        dataset = qml.data.load(
            "qchem",
            molname="H2",
            bondlength=0.742,
            basis="STO-3G",
        )[0]

        hamiltonian = dataset.hamiltonian
        initial_state = np.array([1, 1, 0, 0], dtype=int)

    hamiltonian = comm.bcast(hamiltonian, root=0)
    initial_state = comm.bcast(initial_state, root=0)

    return hamiltonian, 4, initial_state


def write_results_csv(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layers",
        "final_energy",
        "minimum_energy",
        "minimum_step",
        "parameters",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_results(results):
    print()
    print("==============================================================")
    print("DISTRIBUTED LAYERED VQC RESULTS")
    print("==============================================================")
    print(
        f"{'Layers':>6} "
        f"{'Final Energy':>16} "
        f"{'Minimum Energy':>16} "
        f"{'Min Step':>10} "
        f"{'Params':>8}"
    )

    for result in results:
        print(
            f"{result['layers']:>6d} "
            f"{result['final_energy']:>16.10f} "
            f"{result['minimum_energy']:>16.10f} "
            f"{result['minimum_step']:>10d} "
            f"{result['parameters']:>8d}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI-distributed fixed-layer VQC ground-state experiment."
    )
    parser.add_argument("--problem", choices=["h2"], default="h2")
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--stepsize", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="layered_vqc_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.max_layers < 1:
        if rank == 0:
            raise ValueError("--max-layers must be at least 1.")
        return

    if args.problem == "h2":
        hamiltonian, n_qubits, initial_state = load_h2_problem(comm)
    else:
        raise ValueError(f"Unknown problem: {args.problem}")

    start_time = MPI.Wtime()

    results = run_distributed_layer_sweep(
        hamiltonian=hamiltonian,
        n_qubits=n_qubits,
        initial_state=initial_state,
        max_layers=args.max_layers,
        steps=args.steps,
        stepsize=args.stepsize,
        seed=args.seed,
        comm=comm,
    )

    comm.Barrier()
    end_time = MPI.Wtime()

    elapsed_time = end_time - start_time
    if rank == 0:
        print_results(results)
        write_results_csv(results, args.output)

        print()
        print("==============================================================")
        print("PARALLEL PERFORMANCE")
        print("==============================================================")
        print(f"MPI ranks:       {comm.Get_size()}")
        print(f"Maximum layers:  {args.max_layers}")
        print(f"Steps per layer: {args.steps}")
        print(f"Wall-clock time: {elapsed_time:.4f} seconds")

        print()
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
