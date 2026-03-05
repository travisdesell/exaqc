import numpy as np
import torch

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


def sample_batch_balanced(
    data: list, batch_size: int, shuffle_each_step: bool, step: int
) -> list | None:
    """Creates a mini-batch with equal class distribution from provided data list.

    Args:
        data (list): The dataset list where each element is a tuple (x, y_onehot, cls)
        batch_size (int): Size of the batch (will be rounded down to nearest multiple of num_classes)
        shuffle_each_step (bool): If True, randomly sample from each class; if False, sequential
        step (int): The current step in training

    Returns:
        list: The mini-batch of data with equal representation from each class
    """

    if data is None:
        return None

    class_buckets: dict[str, list[int]] = {}
    for i, (_, _, cls) in enumerate(data):
        class_buckets.setdefault(cls, []).append(i)

    classes = sorted(class_buckets.keys())
    num_classes = len(classes)

    if batch_size is None or batch_size > len(data):
        max_count = max(len(bucket) for bucket in class_buckets.values())
        batch = []
        for cls in classes:
            bucket = class_buckets[cls]
            n = len(bucket)
            batch.extend(data[bucket[i]] for i in range(n))
            remainder = max_count - n
            batch.extend(data[bucket[i % n]] for i in range(remainder))  # Oversample
        return batch

    samples_per_class = batch_size // num_classes

    rng = np.random.default_rng(seed=42)
    batch = []
    for cls in classes:
        bucket = class_buckets[cls]
        n = len(bucket)

        if shuffle_each_step:
            chosen = rng.integers(low=0, high=n, size=(samples_per_class,))
            batch.extend(data[bucket[i]] for i in chosen)
        else:
            start = (step * samples_per_class) % n
            batch.extend(
                data[bucket[(start + i) % n]] for i in range(samples_per_class)
            )

    return batch


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
