import numpy as np
import torch
from collections import deque
from typing import Optional

from src.circuits.circuit import CircuitGenome


def register_wire_map(registers: dict[str, int]) -> dict:
    """Return a dict mapping register names to PennyLane wires."""
    wire_map = {}
    offset = 0
    for name, size in registers.items():
        wire_map[name] = list(range(offset, offset + size))
        offset += size
    return wire_map


def sample_batch(
    data: list, batch_size: int, shuffle_each_step: bool, step: int
) -> list:
    """Creates a mini-batch from provided data list of given size

    Args:
        data (list): The dataset list where each element is a tuple (x, y, cls)
        batch_size (int): Size of the batch
        shuffle_each_step (bool): If you want random shuffling or sequential
        step (int): The current step in training

    Returns:
        list: The mini-batch of data
    """
    n = len(data)
    if batch_size is None:
        return data
    if shuffle_each_step:
        rng = np.random.default_rng(seed=42)
        idx = rng.integers(low=0, high=n, size=(batch_size,))
        return [data[i] for i in idx.tolist()]
    start = (step * batch_size) % n
    return [data[(start + i) % n] for i in range(batch_size)]


class BalancedBatchSampler:
    """Draws mini-batches with equal class representation.

    Each class maintains its own independent queue of indices.  When a
    class queue is exhausted it is refilled (and optionally shuffled)
    before sampling continues.  Because class sizes differ, queues empty
    at different rates, so shuffles are triggered independently per class.

    Args:
        data:             List of (features, y_onehot, cls_name) tuples.
        batch_size:       Total samples per batch.  Rounded down to the
                          nearest multiple of the number of classes.
                          Pass ``None`` to return one full epoch worth of
                          data with minority classes oversampled to match
                          the majority class size.
        shuffle:          Whether to shuffle a class bucket when it is
                          exhausted and refilled.
    """

    def __init__(self, data: list, batch_size: Optional[int], shuffle: bool) -> None:
        self.data = data
        self.shuffle = shuffle

        class_indices: dict[str, list[int]] = {}
        for i, (_, _, cls) in enumerate(data):
            class_indices.setdefault(cls, []).append(i)

        self.classes: list[str] = sorted(class_indices.keys())
        self.num_classes: int = len(self.classes)
        self.class_indices = class_indices

        if batch_size is None:
            self.samples_per_class: int = max(len(v) for v in class_indices.values())
            self.batch_size: int = self.samples_per_class * self.num_classes
        else:
            self.samples_per_class = batch_size // self.num_classes
            self.batch_size = self.samples_per_class * self.num_classes

        # One deque per class — these persist across sample() calls
        self._queues: dict[str, deque[int]] = {
            cls: self._make_queue(cls) for cls in self.classes
        }

    def _make_queue(self, cls: str) -> deque[int]:
        indices = list(self.class_indices[cls])
        if self.shuffle:
            np.random.shuffle(indices)
        return deque(indices)

    def _draw(self, cls: str, n: int) -> list[int]:
        """Pull n indices from the class queue, refilling when exhausted."""
        out: list[int] = []
        queue = self._queues[cls]
        while len(out) < n:
            if not queue:
                queue.extend(self._make_queue(cls))
            out.append(queue.popleft())
        if not queue:
            queue.extend(self._make_queue(cls))
        return out

    def sample(self) -> list:
        """Return one balanced batch, advancing each class queue."""
        batch = []
        for cls in self.classes:
            indices = self._draw(cls, self.samples_per_class)
            batch.extend(self.data[i] for i in indices)
        return batch

    def reset(self) -> None:
        """Rebuild all queues from scratch (useful between epochs)."""
        self._queues = {cls: self._make_queue(cls) for cls in self.classes}


def genome_to_torch_params(genome: CircuitGenome) -> dict[str, torch.nn.Parameter]:
    """Extract trainable genome parameters into torch Parameters.

    Iterates over enabled gates in the genome and converts each gate parameter
    into a `torch.nn.Parameter`. Parameters are keyed using the stable identifier
    `<innovation_number>:<parameter_name>`.

    Args:
        genome (CircuitGenome): Quantum circuit genome with parametric gates.

    Returns:
        dict[str, torch.nn.Parameter]: Mapping from parameter keys to torch Parameters.
    """
    params: dict[str, torch.nn.Parameter] = {}
    for gate in genome.gates:
        if gate.enabled:
            for name, value in gate.parameters.items():
                key = f"{gate.innovation_number}:{name}"
                params[key] = torch.nn.Parameter(
                    torch.tensor(float(value), dtype=torch.float64)
                )
    return params


def _extract_param_value(v: torch.Tensor | float) -> float:
    """Convert a tensor or scalar parameter to a Python float.

    Args:
        v (torch.Tensor | float): Parameter value.

    Returns:
        float: Extracted scalar value.
    """
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def torch_params_to_genome(
    genome: CircuitGenome, trained_params: dict[str, torch.Tensor] | dict[str, float]
):
    """Write trained torch parameters back into a genome.

    Parameters are matched using `<innovation_number>:<parameter_name>` keys.

    Args:
        genome (CircuitGenome): Genome to update.
        trained_params (dict[str, torch.Tensor | float]): Trained parameters.
    """
    for gate in genome.gates:
        if gate.enabled:
            for name in gate.parameters.keys():
                key = f"{gate.innovation_number}:{name}"
                if key in trained_params:
                    gate.parameters[name] = _extract_param_value(trained_params[key])
