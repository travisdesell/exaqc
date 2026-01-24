from __future__ import annotations

import torch
from typing import Tuple
from .base import QuantumDataset


class SeedsDataset(QuantumDataset):
    """
    UCI Seeds dataset (Wheat kernels)

    Returns:
      x: torch.float32 shape [7]
      y: torch.float32 shape [3] (one-hot)

    Classes:
      0 -> Kama
      1 -> Rosa
      2 -> Canadian
    """

    def __init__(
        self,
        *,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 0,
    ):
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split

        # ---- Load raw data ----
        # Format: 7 features + 1 class label (1,2,3)
        data = np.loadtxt(
            "src/quantum_datasets/data/seeds_dataset.txt"
        )

        X = data[:, :7]               # (210, 7)
        y = data[:, 7].astype(int)    # {1,2,3}

        # Convert labels -> {0,1,2}
        y = y - 1

        self.num_classes = 3

        # ---- Scale features to [0,1] ----
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

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # [7]
        cls = int(self.y[idx].item())

        y_onehot = torch.zeros(3, dtype=torch.float32)
        y_onehot[cls] = 1.0

        return x, y_onehot
