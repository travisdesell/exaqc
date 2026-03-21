from __future__ import annotations

from collections import Counter

import pytest
import torch

from src.utils.helpers import BalancedBatchSampler
from src.quantum_datasets import (
    IrisDataset,
    WineDataset,
    SeedsDataset,
    BreastCancerDataset,
)


def make_data(class_names: tuple[str, ...], counts: tuple[int, ...]) -> list:
    data = []
    n_classes = len(class_names)
    for cls_idx, n in enumerate(counts):
        for i in range(n):
            x = torch.tensor([float(cls_idx * 100 + i)])
            y_onehot = torch.zeros(n_classes, dtype=torch.float32)
            y_onehot[cls_idx] = 1.0
            data.append((x, y_onehot, class_names[cls_idx]))
    return data


def test_queues_persist_between_samples():
    data = make_data(("A", "B"), counts=(6, 6))
    sampler = BalancedBatchSampler(data, batch_size=4, shuffle=False)

    # 2 samples per class per batch × 3 batches = 6 draws per class → one full cycle
    seen_ids: set[int] = set()
    for _ in range(3):
        batch = sampler.sample()
        batch_ids = [id(item[0]) for item in batch]
        assert not (
            set(batch_ids) & seen_ids
        ), "An index was reused before its class queue was exhausted"
        seen_ids.update(batch_ids)


def test_shuffle_triggered_on_exhaustion_not_every_step():
    data = make_data(("X",), counts=(6,))

    # Draw 2 samples at a time — queue drains across calls but is not yet exhausted
    sampler = BalancedBatchSampler(data, batch_size=2, shuffle=True)
    order_before = list(sampler._queues["X"])

    sampler.sample()
    sampler.sample()
    order_mid = list(sampler._queues["X"])

    assert (
        order_mid == order_before[4:]
    ), "Queue was reshuffled mid-cycle before exhaustion"

    # Third sample() exhausts the queue → refill triggers a reshuffle
    sampler.sample()
    assert (
        len(sampler._queues["X"]) == 6
    ), "Queue was not refilled to full size after exhaustion"


def test_different_class_sizes_exhaust_independently():
    data = make_data(("A", "B"), counts=(3, 9))
    # 1 sample per class per batch
    sampler = BalancedBatchSampler(data, batch_size=2, shuffle=False)

    # After 3 samples:
    #   A has drawn 3/3 → exhausted, eagerly refilled back to 3 (nothing further drawn)
    #   B has drawn 3/9 → 6 remaining, never refilled
    for _ in range(3):
        sampler.sample()

    assert len(sampler._queues["B"]) == 6, "B refilled too early"
    assert len(sampler._queues["A"]) == 3, "A did not refill at the right time"


def test_reset_rebuilds_all_queues():
    """reset() should restore every queue to its original state."""
    data = make_data(("A", "B", "C"), counts=(10, 10, 10))
    sampler = BalancedBatchSampler(data, batch_size=6, shuffle=False)

    initial_queues = {cls: list(q) for cls, q in sampler._queues.items()}

    for _ in range(3):
        sampler.sample()

    sampler.reset()

    assert {cls: list(q) for cls, q in sampler._queues.items()} == initial_queues


DATASETS = [
    ("iris", IrisDataset, 4, 3),
    ("wine", WineDataset, 13, 3),
    ("seeds", SeedsDataset, 7, 3),
    ("breast_cancer", BreastCancerDataset, 30, 2),
]

BATCH_SIZES = [4, 8, 12]  # int → fixed mini-batch


def is_evenly_distributed(batch: list, n_classes: int) -> bool:
    """Return True iff every class appears exactly len(batch) // n_classes times."""
    counts = Counter(cls for _, _, cls in batch)
    expected = len(batch) // n_classes
    return len(counts) == n_classes and all(v == expected for v in counts.values())


@pytest.fixture(scope="module")
def loaded_datasets() -> dict:
    """Load each dataset once"""
    return {
        ds_name: [ds_cls(split="train")[i] for i in range(len(ds_cls(split="train")))]
        for ds_name, ds_cls, _, _ in DATASETS
    }


@pytest.mark.parametrize("ds_name, ds_cls, input_size, n_classes", DATASETS)
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_even_batching(
    ds_name, ds_cls, input_size, n_classes, shuffle, batch_size, loaded_datasets
):
    data = loaded_datasets[ds_name]

    sampler = BalancedBatchSampler(data, batch_size, shuffle)

    for _ in range(max(1, len(data) // sampler.samples_per_class)):
        batch = sampler.sample()
        assert is_evenly_distributed(batch, n_classes), (
            f"[{ds_name}] uneven distribution — "
            f"batch_size={batch_size}, shuffle={shuffle}, "
            f"counts={dict(Counter(cls for _, _, cls in batch))}"
        )
