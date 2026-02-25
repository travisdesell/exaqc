from __future__ import annotations

import torch
from typing import Tuple
from .base import QuantumDataset
import numpy as np


class BreastCancerDataset(QuantumDataset):
    """Quantum-compatible Breast Cancer Wisconsin (Diagnostic) dataset.

    This dataset adapts the Breast Cancer Wisconsin (Diagnostic) dataset
    for quantum machine learning workflows. All features are scaled to
    the range [0, 1], making them suitable for angle-based quantum
    encodings such as RY(pi * x).

    Each sample consists of:
      - a 30-dimensional real-valued feature vector
      - a 2-dimensional one-hot encoded label
      - a human-readable class name

    Class mapping:
      - 0 → benign
      - 1 → malignant
    """

    def __init__(
        self,
        *,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 0,
    ):
        """Initialize the Breast Cancer dataset.

        Loads the Breast Cancer Wisconsin (Diagnostic) dataset, applies
        min–max feature scaling, performs a stratified train/test split,
        and converts data into PyTorch tensors suitable for quantum
        learning pipelines.

        Args:
            split (str): Dataset split to use. Must be `"train"` or `"test"`.
            train_frac (float): Fraction of data to allocate to training.
            seed (int): Random seed for reproducible dataset splitting.

        Raises:
            ValueError: If `split` is not `"train"` or `"test"`.
        """
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        data = load_breast_cancer()

        X = data.data  # (569, 30)
        y = data.target  # {0,1}
        self.labels = data.target_names

        self.num_classes = 2

        # ---- Scale to [0,1] for angle embedding ----
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
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.long)
        elif split == "test":
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.long)
        else:
            raise ValueError("split must be 'train' or 'test'")

        _, self.counts = np.unique(self.y.cpu().numpy(), return_counts=True)
        self.class_counts = dict(zip(data.target_names, self.counts))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the selected split.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Retrieve a single dataset sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple:
                - x (torch.Tensor): Feature vector of shape `[30]`.
                - y_onehot (torch.Tensor): One-hot encoded label of shape `[2]`.
                - cls_name (str): Human-readable class label.
        """
        x = self.X[idx]  # [30]
        cls = int(self.y[idx].item())

        y_onehot = torch.zeros(2, dtype=torch.float32)
        y_onehot[cls] = 1.0

        return x, y_onehot, self.labels[cls]
