import numpy as np
from collections import defaultdict

def register_wire_map(registers: dict[str, int]) -> dict:
    """Return a dict mapping register names to PennyLane wires."""
    wire_map = {}
    offset = 0
    for name, size in registers.items():
        wire_map[name] = list(range(offset, offset + size))
        offset += size
    return wire_map


def sample_even_batch(
    data,
    batch_size: int | None,
    shuffle_each_step: bool,
    seed: int = 42,
    **kwargs
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
    n = len(data)
    if batch_size is None:
        return data

    rng = np.random.default_rng(seed)

    idx_by_class = defaultdict(list)
    for i, (_, y_onehot, _) in enumerate(data):
        k = int(np.array(y_onehot).argmax().item())
        idx_by_class[k].append(i)

    classes = sorted(idx_by_class.keys())
    K = len(classes)

    if batch_size < K:
        raise ValueError("batch_size must be >= number of classes.")
    per_class = batch_size // K
    remainder = batch_size - per_class * K

    # Shuffle each pool initially
    for c in classes:
        rng.shuffle(idx_by_class[c])

    ptr = dict.fromkeys(classes, 0)

    batch_idx = []

    # Draw equal items per class
    for c in classes:
        pool = idx_by_class[c]
        start = ptr[c]
        end = start + per_class

        if end > len(pool):
            # wrap-around (oversample) + reshuffle
            if shuffle_each_step:
                rng.shuffle(pool)
            start, end = 0, per_class

        batch_idx.extend(pool[start:end])
        ptr[c] = end

    # Fill remainder (if batch_size not divisible by K)
    if remainder > 0:
        extra_classes = rng.choice(classes, size=remainder, replace=True)
        for c in extra_classes:
            pool = idx_by_class[c]
            j = rng.integers(0, len(pool))
            batch_idx.append(int(pool[j]))

    if shuffle_each_step:
        rng.shuffle(batch_idx)

    return [data[i] for i in batch_idx]