from __future__ import annotations

import torch
from typing import Tuple
from .base import QuantumDataset


class BreastCancerDataset(QuantumDataset):
    """
    Breast Cancer Wisconsin (Diagnostic)

    Returns:
      x: torch.float32 shape [30]
      y: torch.float32 shape [2] (one-hot)

    Classes:
      0 -> benign
      1 -> malignant
    """

    def __init__(
        self,
        *,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 0,
    ):
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        data = load_breast_cancer()

        X = data.data  # (569, 30)
        y = data.target  # {0,1}

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

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # [30]
        cls = int(self.y[idx].item())

        y_onehot = torch.zeros(2, dtype=torch.float32)
        y_onehot[cls] = 1.0

        return x, y_onehot
