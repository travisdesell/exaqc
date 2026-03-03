from __future__ import annotations

import torch
from src.datasets.base import QuantumDataset
import numpy as np


class WineDataset(QuantumDataset):
    """Quantum-compatible UCI Wine dataset.

    This dataset wraps the UCI Wine dataset (as provided by
    `sklearn.datasets.load_wine`) for use in quantum machine learning
    workflows. All features are min–max scaled to the range [0, 1],
    making them suitable for angle-based quantum encodings such as
    RY(pi * x).

    Each sample consists of:
      - a 13-dimensional real-valued feature vector
      - a 3-dimensional one-hot encoded class label
      - a human-readable class name
    """

    def __init__(self, *, split: str = "train", train_frac: float = 0.8, seed: int = 0):
        """Initialize the Wine dataset.

        Loads the UCI Wine dataset, rescales features to [0, 1], performs
        a stratified train/test split, and converts data into PyTorch
        tensors suitable for quantum learning pipelines.

        Args:
            split (str): Dataset split to use. Must be `"train"` or `"test"`.
            train_frac (float): Fraction of samples used for training.
            seed (int): Random seed for reproducible dataset splitting.

        Raises:
            ValueError: If `split` is not `"train"` or `"test"`.
        """
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        data = load_wine()
        X = data.data  # (178, 13)
        y = data.target  # (178,)
        self.labels = data.target_names
        n_classes = len(set(y.tolist()))

        self.num_classes = n_classes  # IMPORTANT: int (3)

        # ---- Scale features to [0, 1] ----
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
        self.class_counts = dict(zip(data.target_names, self.counts))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the selected split.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Retrieve a single dataset sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple:
                - x (torch.Tensor): Feature vector of shape `[13]`.
                - y_onehot (torch.Tensor): One-hot encoded label of shape `[3]`.
                - cls_name (str): Human-readable class label.
        """
        x = self.X[idx]  # [13]
        cls = int(self.y[idx].item())
        y_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        y_onehot[cls] = 1.0
        return x, y_onehot, self.labels[cls]
