from __future__ import annotations

from collections import Counter

import pytest

from src.utils.helpers import sample_batch_balanced
from src.quantum_datasets import (
    IrisDataset,
    WineDataset,
    SeedsDataset,
    BreastCancerDataset,
)


DATASETS = [
    ("iris", IrisDataset, 4, 3),
    ("wine", WineDataset, 13, 3),
    ("seeds", SeedsDataset, 7, 3),
    ("breast_cancer", BreastCancerDataset, 30, 2),
]

BATCH_SIZES = [None, 8, 12]  # None → oversample; int → fixed mini-batch


def is_evenly_distributed(batch: list, n_classes: int) -> bool:
    """Return True iff every class appears exactly len(batch) // n_classes times."""
    counts = Counter(cls for _, _, cls in batch)
    expected = len(batch) // n_classes
    return len(counts) == n_classes and all(v == expected for v in counts.values())


@pytest.mark.parametrize("ds_name, ds_cls, input_size, n_classes", DATASETS)
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_even_batching(ds_name, ds_cls, input_size, n_classes, shuffle, batch_size):
    dataset = ds_cls(split="train")
    data = [dataset[i] for i in range(len(dataset))]

    effective_batch_size = (
        batch_size if batch_size is None else (batch_size // n_classes) * n_classes
    )

    batch = sample_batch_balanced(data, effective_batch_size, shuffle, step=0)

    assert is_evenly_distributed(batch, n_classes), (
        f"[{ds_name}] uneven distribution — "
        f"batch_size={batch_size}, shuffle={shuffle}, "
        f"counts={dict(Counter(cls for _, _, cls in batch))}"
    )
