# src/datasets/half_adder.py

from __future__ import annotations
from typing import Tuple, List
import torch

from src.datasets.base import QuantumDataset


def bits_to_statevector(bits: List[int], n_qubits: int) -> torch.Tensor:
    """
    Encode classical bits as a computational basis statevector.
    """
    index = 0
    for i, bit in enumerate(bits):
        index |= (bit & 1) << (n_qubits - i - 1)

    state = torch.zeros(2**n_qubits, dtype=torch.complex128)
    state[index] = 1.0 + 0.0j
    return state

class HalfAdderDataset(QuantumDataset):
    """
    Torch-style dataset for the half-adder truth table.

    Qubit layout (default):
        q0, q1 : inputs (a, b)
        q2     : sum
        q3     : carry
    """

    def __init__(
        self,
        *,
        n_qubits: int = 4,
        input_qubits: Tuple[int, int] = (0, 1),
        output_qubits: Tuple[int, int] = (2, 3),
    ):
        self.n_qubits = n_qubits
        self.input_qubits = input_qubits
        self.output_qubits = output_qubits

        # Precompute dataset
        self._data = []
        self._build()

    def _build(self):
        truth_table = {
            (0, 0): (0, 0),
            (0, 1): (1, 0),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
        }

        for (a, b), (s, c) in truth_table.items():
            input_bits = torch.zeros(self.n_qubits, dtype=torch.int64)
            input_bits[self.input_qubits[0]] = a
            input_bits[self.input_qubits[1]] = b

            output_bits = [0] * self.n_qubits
            output_bits[self.input_qubits[0]] = a
            output_bits[self.input_qubits[1]] = b
            output_bits[self.output_qubits[0]] = s
            output_bits[self.output_qubits[1]] = c

            target_state = bits_to_statevector(output_bits, self.n_qubits)
            self._data.append((input_bits, target_state))


    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
