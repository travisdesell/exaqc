from __future__ import annotations

import torch
from src.datasets.base import QuantumDataset
from .base import QuantumDataset
import numpy as np


class IrisDataset(QuantumDataset):
    """Quantum-compatible Iris dataset wrapper.

    This dataset adapts the classic Iris classification problem for use in
    quantum machine learning pipelines. Features are scaled to the range
    [0, 1], making them suitable for angle-based quantum encodings
    (e.g., RY(pi * x)).

    Each sample consists of:
      - a 4-dimensional feature vector
      - a 3-dimensional one-hot encoded label
      - a human-readable class name

    The dataset supports stratified train/test splits and exposes class
    frequency statistics for balanced or cost-sensitive training.
    """

    def __init__(self, *, split: str = "train", train_frac: float = 0.8, seed: int = 0):
        """Initialize the Iris dataset.

        Loads the Iris dataset, applies min–max normalization to features,
        performs a stratified train/test split, and prepares tensors for
        quantum learning workflows.

        Args:
            split (str): Dataset split to use. Must be `"train"` or `"test"`.
            train_frac (float): Fraction of data to allocate to training.
            seed (int): Random seed for reproducible splitting.

        Raises:
            ValueError: If `split` is not `"train"` or `"test"`.
        """
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        iris = load_iris()
        X = iris.data  # (150, 4)
        y = iris.target  # (150,)
        self.labels = iris.target_names

        self.num_classes = len(self.labels)

        # Scale features to [0, 1] (suitable for RY(pi * x) encodings)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_frac,
            random_state=seed,
            stratify=y,
        )

        if split == "train":
            x_use, y_use = X_train, y_train
        elif split == "test":
            x_use, y_use = X_test, y_test
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.X = torch.tensor(x_use, dtype=torch.float32)
        self.y = torch.tensor(y_use, dtype=torch.long)

        _, self.counts = np.unique(self.y.cpu().numpy(), return_counts=True)
        self.class_counts = dict(zip(iris.target_names, self.counts))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Retrieve a single dataset sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple:
                - x (torch.Tensor): Feature vector of shape `[4]`.
                - y_onehot (torch.Tensor): One-hot label of shape `[3]`.
                - cls_name (str): Human-readable class label.
        """
        x = self.X[idx]  # [4]
        cls = int(self.y[idx].item())
        y_onehot = torch.zeros(3, dtype=torch.float32)
        y_onehot[cls] = 1.0
        return x, y_onehot, self.labels[cls]
