import torch

from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, features, targets):
        # Convert NumPy arrays to PyTorch Tensors
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
