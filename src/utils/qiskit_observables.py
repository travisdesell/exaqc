"""Qiskit observable helpers for probability readout.

To match PennyLane's `qml.probs(wires=output)` (which returns the joint
distribution over output qubits, indexed in PennyLane convention where
the first wire in the list is the MSB), we construct one Pauli projector
per output bitstring: |b><b| = (1/2^n) prod_j (I + (-1)^{b_j} Z_j).

A list of these projectors plugged into `EstimatorQNN.observables` makes
the QNN's forward output a vector of joint probabilities.
"""
from __future__ import annotations

from typing import List

from qiskit.quantum_info import SparsePauliOp


def bitstring_projector(
    bitstring: int,
    n_total_qubits: int,
    output_qubit_indices: List[int],
) -> SparsePauliOp:
    """Pauli expansion of |bitstring><bitstring| on the listed output qubits.

    PennyLane convention: the FIRST element of `output_qubit_indices` is the
    MSB of `bitstring`.

    Args:
        bitstring: integer 0..2^n-1 selecting one joint outcome of the n
            output qubits.
        n_total_qubits: total qubits in the circuit (for Pauli string length).
        output_qubit_indices: qiskit qubit indices that compose the output
            register, in PennyLane wire order.

    Returns:
        SparsePauliOp whose expectation on |psi> equals prob(output==bitstring).
    """
    n = len(output_qubit_indices)
    paulis: list[str] = []
    coeffs: list[complex] = []

    for term in range(2 ** n):
        pauli_chars = ["I"] * n_total_qubits
        sign = 1
        for j, q in enumerate(output_qubit_indices):
            if (term >> j) & 1:
                # qiskit string: rightmost char = qubit 0
                pauli_chars[n_total_qubits - 1 - q] = "Z"
                # bit j of bitstring (in PL convention) is bit (n-1-j) of the int
                if (bitstring >> (n - 1 - j)) & 1:
                    sign = -sign
        paulis.append("".join(pauli_chars))
        coeffs.append(sign / (2 ** n))

    return SparsePauliOp(paulis, coeffs)


def output_projector_observables(
    n_total_qubits: int,
    output_qubit_indices: List[int],
) -> list[SparsePauliOp]:
    """Build one projector observable for each joint output bitstring.

    The resulting list has length 2^|output|. Plugged into
    `EstimatorQNN.observables`, the QNN's forward output is the joint
    probability distribution in PennyLane index order.
    """
    n = len(output_qubit_indices)
    return [
        bitstring_projector(b, n_total_qubits, output_qubit_indices)
        for b in range(2 ** n)
    ]
