from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision import transforms


class ImageDataset(Dataset):
    """Raw image dataset for quantum image classification.

    Returns image tensors and one-hot labels. The image is intentionally not
    embedded here; embedding is learned by a classical module before the
    quantum circuit.
    """

    def __init__(
        self,
        *,
        dataset: str,
        root: str = "./data",
        split: str = "train",
        n_classes: int = 10,
        max_samples: int | None = None,
        normalize: bool = True,
    ):
        self.dataset = dataset
        self.split = split
        self.n_classes = n_classes

        transform_list = [transforms.ToTensor()]

        if normalize:
            if dataset == "cifar10":
                transform_list.append(
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616),
                    )
                )
            else:
                transform_list.append(
                    transforms.Normalize(
                        mean=(0.5,),
                        std=(0.5,),
                    )
                )

        transform = transforms.Compose(transform_list)

        train = split == "train"

        if dataset == "cifar10":
            base = CIFAR10(root=root, train=train, download=True, transform=transform)
            self.input_dim = 3 * 32 * 32
        elif dataset == "mnist":
            base = MNIST(root=root, train=train, download=True, transform=transform)
            self.input_dim = 1 * 28 * 28
        elif dataset == "fashion_mnist":
            base = FashionMNIST(
                root=root,
                train=train,
                download=True,
                transform=transform,
            )
            self.input_dim = 1 * 28 * 28
        else:
            raise ValueError(f"Unknown dataset={dataset}")

        if max_samples is not None:
            self.indices = list(range(min(max_samples, len(base))))
        else:
            self.indices = list(range(len(base)))

        self.base = base

        labels = [int(self.base[i][1]) for i in self.indices]
        self.counts = [labels.count(k) for k in range(n_classes)]
        self.class_counts = {str(k): self.counts[k] for k in range(n_classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.base[real_idx]

        y_onehot = F.one_hot(
            torch.tensor(y, dtype=torch.long),
            num_classes=self.n_classes,
        ).to(torch.float32)

        return x, y_onehot, str(y)
