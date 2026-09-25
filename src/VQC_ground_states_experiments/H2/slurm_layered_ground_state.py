import argparse
import csv
import time
from pathlib import Path

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


def load_h2_problem():
    dataset = qml.data.load(
        "qchem",
        molname="H2",
        bondlength=0.742,
        basis="STO-3G",
    )[0]

    hamiltonian = dataset.hamiltonian
    initial_state = np.array([1, 1, 0, 0], dtype=int)

    return hamiltonian, 4, initial_state


def write_result_csv(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--stepsize", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    hamiltonian, n_qubits, initial_state = load_h2_problem()

    start = time.perf_counter()

    result = optimize_layer_count(
        hamiltonian=hamiltonian,
        n_qubits=n_qubits,
        initial_state=initial_state,
        layers=args.layers,
        steps=args.steps,
        stepsize=args.stepsize,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - start

    result["wall_time_seconds"] = elapsed

    print("============================================")
    print(f"Layers:          {result['layers']}")
    print(f"Minimum energy:  {result['minimum_energy']:.10f}")
    print(f"Minimum step:    {result['minimum_step']}")
    print(f"Parameters:      {result['parameters']}")
    print(f"Wall time:       {elapsed:.4f} seconds")
    print("============================================")

    write_result_csv(result, args.output)


if __name__ == "__main__":
    main()
