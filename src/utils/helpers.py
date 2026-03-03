import numpy as np
import torch

from collections import defaultdict
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
        idx = np.randint(low=0, high=n, size=(batch_size,))
        return [data[i] for i in idx.tolist()]
    start = (step * batch_size) % n
    return [data[(start + i) % n] for i in range(batch_size)]


def sample_even_batch(
    data, batch_size: int | None, shuffle_each_step: bool, seed: int = 42, **kwargs
):
    """Creates a mini-batch from provided classification data list of given size
        Maintains equal class distrubution in batch

    Args:
        data (list): The dataset list where each element is a tuple (x, y, cls)
        batch_size (int): Size of the batch
        shuffle_each_step (bool): If you want random shuffling or sequential
        seed (int): The seed for shuffling

    Returns:
        list: The mini-batch of data
    """
    if batch_size is None:
        return data

    rng = np.random.default_rng(seed)

    idx_by_class = defaultdict(list)
    for i, (_, y_onehot, _) in enumerate(data):
        k = int(np.array(y_onehot.cpu()).argmax().item())
        idx_by_class[k].append(i)

    classes = sorted(idx_by_class.keys())
    K = len(classes)
    if batch_size < K:
        raise ValueError("batch_size must be >= number of classes.")

    per_class = batch_size // K
    remainder = batch_size - per_class * K

    # shuffle each pool initially
    for c in classes:
        rng.shuffle(idx_by_class[c])

    ptr = {c: 0 for c in classes}  # <-- persists!

    batch_idx = []

    for c in classes:
        pool = idx_by_class[c]
        start = ptr[c]
        end = start + per_class

        if end > len(pool):
            # wrap-around oversampling for minority classes
            if shuffle_each_step:
                rng.shuffle(pool)
            start, end = 0, per_class

        batch_idx.extend(pool[start:end])
        ptr[c] = end

    if remainder > 0:
        extra_classes = rng.choice(classes, size=remainder, replace=True)
        for c in extra_classes:
            pool = idx_by_class[c]
            j = rng.integers(0, len(pool))
            batch_idx.append(int(pool[j]))

    if shuffle_each_step:
        rng.shuffle(batch_idx)

    return [data[i] for i in batch_idx]


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
