from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.datasets.base import QuantumDataset


class ImageEmbeddingDataset(QuantumDataset):
    """Quantum-compatible dataset wrapper for saved image embeddings.

    This dataset loads precomputed low-dimensional image embeddings and their
    aligned class labels from disk, and exposes them in a format suitable for
    quantum learning workflows.

    Expected directory structure:
        src/embeddings/images/<dataset_name>/
            train_embeddings.npy
            train_labels.npy
            test_embeddings.npy
            test_labels.npy

    Each sample consists of:
      - a low-dimensional embedding vector
      - a one-hot encoded class label
      - a human-readable class name

    Subclasses must define:
      - dataset_name
      - labels
    """

    dataset_name: str = ""
    labels: list[str] = []

    def __init__(
        self,
        *,
        split: str = "train",
        embedding_root: str = "src/embeddings/images",
    ) -> None:
        """Initialize the embedding dataset.

        Args:
            split: Dataset split to use. Must be `"train"` or `"test"`.
            embedding_root: Root directory containing saved embedding folders.

        Raises:
            ValueError: If `split` is not `"train"` or `"test"`.
            FileNotFoundError: If the required embedding files do not exist.
        """
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        if not self.dataset_name:
            raise ValueError("Subclasses must define dataset_name.")

        if not self.labels:
            raise ValueError("Subclasses must define labels.")

        dataset_dir = Path(embedding_root) / self.dataset_name

        emb_path = dataset_dir / f"{split}_embeddings.npy"
        label_path = dataset_dir / f"{split}_labels.npy"

        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")

        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        X = np.load(emb_path)
        y = np.load(label_path)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.num_classes = len(self.labels)

        unique, counts = np.unique(self.y.cpu().numpy(), return_counts=True)
        self.counts = counts
        self.class_counts = {
            self.labels[int(cls_idx)]: int(count)
            for cls_idx, count in zip(unique, counts)
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples in the selected split.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Retrieve a single dataset sample.

        Args:
            idx: Index of the sample.

        Returns:
            tuple:
                - x: Embedding vector of shape `[embedding_dim]`.
                - y_onehot: One-hot encoded label of shape `[num_classes]`.
                - cls_name: Human-readable class label.
        """
        x = self.X[idx]
        cls = int(self.y[idx].item())

        y_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        y_onehot[cls] = 1.0

        return x, y_onehot, self.labels[cls]


class MNISTEmbeddingDataset(ImageEmbeddingDataset):
    """Quantum-compatible MNIST embedding dataset.

    Class mapping:
      - 0 → 0
      - 1 → 1
      - 2 → 2
      - 3 → 3
      - 4 → 4
      - 5 → 5
      - 6 → 6
      - 7 → 7
      - 8 → 8
      - 9 → 9
    """

    dataset_name = "mnist"
    labels = [str(i) for i in range(10)]


class FashionMNISTEmbeddingDataset(ImageEmbeddingDataset):
    """Quantum-compatible Fashion-MNIST embedding dataset.

    Class mapping:
      - 0 → T-shirt/top
      - 1 → Trouser
      - 2 → Pullover
      - 3 → Dress
      - 4 → Coat
      - 5 → Sandal
      - 6 → Shirt
      - 7 → Sneaker
      - 8 → Bag
      - 9 → Ankle boot
    """

    dataset_name = "fashion_mnist"
    labels = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


class CIFAR10EmbeddingDataset(ImageEmbeddingDataset):
    """Quantum-compatible CIFAR-10 embedding dataset.

    Class mapping:
      - 0 → airplane
      - 1 → automobile
      - 2 → bird
      - 3 → cat
      - 4 → deer
      - 5 → dog
      - 6 → frog
      - 7 → horse
      - 8 → ship
      - 9 → truck
    """

    dataset_name = "cifar10"
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
