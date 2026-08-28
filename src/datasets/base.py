# src/datasets/base.py

from __future__ import annotations
from typing import Tuple
import torch


class QuantumDataset:
    """
    Base class for quantum training datasets.

    Each item returns:
        input_bits: torch.IntTensor of shape (n_qubits,)
        target_state: torch.ComplexTensor of shape (2**n_qubits,)
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
