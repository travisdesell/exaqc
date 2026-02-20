from __future__ import annotations
import torch
from .base import QuantumDataset
from imblearn.combine import SMOTEENN
import numpy as np


class IrisDataset(QuantumDataset):
    """
    Returns:
      x: torch.float32 shape [4]
      y: torch.float32 shape [3] one-hot
    """

    def __init__(self, *, split: str = "train", train_frac: float = 0.8, seed: int = 0):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        iris = load_iris()
        X = iris.data  # (150,4)
        y = iris.target  # (150,)
        self.labels = iris.target_names

        self.num_classes = len(self.labels)

        # scale to [0,1] (nice for RY(pi*x))
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_frac, random_state=seed, stratify=y
        )

        if split == "train":
            smote = SMOTEENN(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            x_use, y_use = X_train, y_train
        elif split == "test":
            x_use, y_use = X_test, y_test
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.X = torch.tensor(x_use, dtype=torch.float32)
        self.y = torch.tensor(y_use, dtype=torch.long)

        _, counts = np.unique(self.y, return_counts=True)
        self.class_counts = dict(zip(iris.target_names, counts))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # [4]
        cls = int(self.y[idx].item())
        y_onehot = torch.zeros(3, dtype=torch.float32)
        y_onehot[cls] = 1.0
        return x, y_onehot, self.labels[cls]
