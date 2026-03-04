from __future__ import annotations

import torch
from typing import Tuple
from src.datasets.base import QuantumDataset


class SeedsDataset(QuantumDataset):
    """Quantum-compatible UCI Seeds (Wheat Kernels) dataset.

    This dataset adapts the UCI Seeds dataset for quantum machine learning
    workflows. Each data point describes geometric properties of wheat
    kernels and is scaled to the range [0, 1], making it suitable for
    angle-based quantum feature encodings (e.g., RY(pi * x)).

    Each sample consists of:
      - a 7-dimensional real-valued feature vector
      - a 3-dimensional one-hot encoded label
      - a human-readable class name

    Class mapping:
      - 0 → Kama
      - 1 → Rosa
      - 2 → Canadian
    """

    def __init__(
        self,
        *,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 0,
    ):
        """Initialize the Seeds dataset.

        Loads the UCI Seeds dataset from a local text file, rescales features
        to [0, 1], performs a stratified train/test split, and converts
        the data into PyTorch tensors compatible with quantum learning
        pipelines.

        Args:
            split (str): Dataset split to use. Must be `"train"` or `"test"`.
            train_frac (float): Fraction of samples used for training.
            seed (int): Random seed for reproducible splitting.

        Raises:
            ValueError: If `split` is not `"train"` or `"test"`.
        """
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split

        # ---- Load raw data ----
        # Format: 7 features + 1 class label (1,2,3)
        data = np.loadtxt("src/datasets/classification/data/seeds_dataset.txt")
        # Format: 7 features + 1 class label (1, 2, 3)
        target_names = ["Kama", "Rosa", "Canadian"]

        X = data[:, :7]  # (210, 7)
        y = data[:, 7].astype(int)  # {1, 2, 3}
        self.labels = target_names

        # Convert labels -> {0, 1, 2}
        y = y - 1

        self.num_classes = 3

        # ---- Scale features to [0, 1] ----
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # ---- Train/test split ----
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
        self.class_counts = dict(zip(target_names, self.counts))

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
                - x (torch.Tensor): Feature vector of shape `[7]`.
                - y_onehot (torch.Tensor): One-hot encoded label of shape `[3]`.
                - cls_name (str): Human-readable class label.
        """
        x = self.X[idx]  # [7]
        cls = int(self.y[idx].item())

        y_onehot = torch.zeros(3, dtype=torch.float32)
        y_onehot[cls] = 1.0

        return x, y_onehot, self.labels[cls]
