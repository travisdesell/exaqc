from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)
from torchvision import datasets, transforms

from src.datasets.classification import ClassificationDataset

UCI_DATASETS = {
    "iris",
    "wine",
    "seeds",
    "breast_cancer",
}
IMAGE_DATASETS = {
    "mnist",
    "fashion_mnist",
    "cifar10",
}
CLASSIFICATION_DATASETS = sorted(UCI_DATASETS | IMAGE_DATASETS)


@dataclass(frozen=True)
class ClassificationDataSpec:
    """Describes the shape and labels of a classification dataset.

    Attributes:
        name: Canonical dataset name.
        input_shape: Per-sample input shape.
        n_features: Flattened number of input values.
        n_labels: Number of classes.
        is_image: Whether inputs are images.
    """

    name: str
    input_shape: tuple[int, ...]
    n_features: int
    n_labels: int
    is_image: bool


def _attach_loader_metadata(
    loader: DataLoader,
    labels: torch.Tensor,
    data_spec: ClassificationDataSpec,
) -> None:
    """Attaches EXAQC metadata to a dataloader.

    Args:
        loader: Dataloader to update.
        labels: Labels represented by the loader.
        data_spec: Dataset specification.
    """
    label_counts = torch.bincount(
        labels.long(),
        minlength=data_spec.n_labels,
    )
    label_weights = 1.0 / label_counts.clamp_min(1).float()

    loader.label_counts = label_counts
    loader.label_weights = label_weights
    loader.n_labels = data_spec.n_labels
    loader.n_features = data_spec.n_features
    loader.input_shape = data_spec.input_shape
    loader.is_image = data_spec.is_image
    loader.data_spec = data_spec


def _extract_targets(dataset: Dataset) -> torch.Tensor:
    """Extracts labels from a torchvision dataset or subset.

    Args:
        dataset: Dataset whose targets are required.

    Returns:
        One-dimensional integer target tensor.

    Raises:
        TypeError: If targets cannot be extracted.
    """
    if isinstance(dataset, Subset):
        parent_targets = _extract_targets(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return parent_targets[indices]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError(
            f"Dataset type {type(dataset).__name__} has no targets attribute."
        )
    return torch.as_tensor(targets, dtype=torch.long)


def _stratified_indices(
    labels: torch.Tensor,
    selected_size: int | None,
    seed: int,
) -> np.ndarray:
    """Selects a reproducible stratified subset.

    Args:
        labels: Full label tensor.
        selected_size: Number of samples to keep. ``None`` keeps all samples.
        seed: Random seed.

    Returns:
        Selected integer indices.

    Raises:
        ValueError: If the requested size is invalid.
    """
    n_samples = int(labels.numel())
    if selected_size is None or selected_size == n_samples:
        return np.arange(n_samples)

    if selected_size <= 0 or selected_size > n_samples:
        raise ValueError(
            f"selected_size must be in [1, {n_samples}], " f"received {selected_size}."
        )

    all_indices = np.arange(n_samples)
    selected, _ = train_test_split(
        all_indices,
        train_size=selected_size,
        random_state=seed,
        stratify=labels.numpy(),
    )
    return np.asarray(selected)


def _image_transform(dataset: str, training: bool = False) -> transforms.Compose:
    """Creates the deterministic image preprocessing transform.

    Args:
        dataset: Image dataset name.
        training: Whether to apply training-time data augmentation.

    Returns:
        A torchvision transform pipeline.

    Raises:
        ValueError: If the image dataset is unknown.
    """
    normalization = {
        "mnist": ((0.1307,), (0.3081,)),
        "fashion_mnist": ((0.2860,), (0.3530,)),
        "cifar10": (
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    }
    if dataset not in normalization:
        raise ValueError(f"Unknown image dataset: {dataset}")

    mean, std = normalization[dataset]

    transform_list = []

    # Apply augmentation only to CIFAR-10 training data.
    if dataset == "cifar10" and training:
        transform_list.extend(
            [
                transforms.RandomCrop(
                    32,
                    padding=4,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transforms.Compose(transform_list)


def _load_image_training_dataset(
    dataset: str,
    data_dir: str | Path,
    download: bool,
    training_transform: bool = False,
) -> Dataset:
    """Loads the official training split of an image dataset.

    Args:
        dataset: Image dataset name.
        data_dir: Dataset storage directory.
        download: Whether torchvision may download missing files.
        training_transform: Whether to apply training augmentation.

    Returns:
        The official torchvision training dataset.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    transform = _image_transform(dataset, training=training_transform)
    root = str(Path(data_dir))

    constructors = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }
    try:
        constructor = constructors[dataset]
    except KeyError as error:
        raise ValueError(f"Unknown image dataset: {dataset}") from error

    return constructor(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )


def get_image_dataloaders(
    dataset: str,
    data_dir: str | Path = "data",
    batch_size: int = 32,
    validation_batch_size: int | None = None,
    validation_fraction: float = 0.1,
    training_samples: int | None = None,
    validation_samples: int | None = None,
    seed: int = 0,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Builds batched image-classification dataloaders.

    The official training split is divided into stratified training and
    validation subsets. Optional sample limits are applied after that split.

    Args:
        dataset: One of ``mnist``, ``fashion_mnist``, or ``cifar10``.
        data_dir: Dataset storage directory.
        batch_size: Training batch size.
        validation_batch_size: Validation batch size. Defaults to
            ``batch_size``.
        validation_fraction: Fraction of the official training split reserved
            for validation.
        training_samples: Optional stratified training subset size.
        validation_samples: Optional stratified validation subset size.
        seed: Random seed.
        download: Whether missing data may be downloaded.
        num_workers: Number of dataloader worker processes.
        pin_memory: Whether dataloaders should pin host memory.

    Returns:
        Training loader and validation loader.

    Raises:
        ValueError: If arguments or dataset names are invalid.
    """
    if dataset not in IMAGE_DATASETS:
        raise ValueError(
            f"Image dataset must be one of {sorted(IMAGE_DATASETS)}, "
            f"received {dataset!r}."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if validation_batch_size is not None and validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")

    # full_dataset = _load_image_training_dataset(
    #     dataset,
    #     data_dir,
    #     download,
    # )
    # labels = _extract_targets(full_dataset)
    # all_indices = np.arange(len(full_dataset))

    training_full_dataset = _load_image_training_dataset(
        dataset,
        data_dir,
        download,
        training_transform=True,
    )

    validation_full_dataset = _load_image_training_dataset(
        dataset,
        data_dir,
        download,
        training_transform=False,
    )

    labels = _extract_targets(training_full_dataset)

    all_indices = np.arange(len(training_full_dataset))

    training_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels.numpy(),
    )

    training_labels = labels[torch.as_tensor(training_indices)]
    validation_labels = labels[torch.as_tensor(validation_indices)]

    selected_training_positions = _stratified_indices(
        training_labels,
        selected_size=training_samples,
        seed=seed,
    )
    selected_validation_positions = _stratified_indices(
        validation_labels,
        selected_size=validation_samples,
        seed=seed + 1,
    )

    training_indices = np.asarray(training_indices)[selected_training_positions]
    validation_indices = np.asarray(validation_indices)[selected_validation_positions]

    # training_dataset = Subset(
    #     full_dataset,
    #     training_indices.tolist(),
    # )
    # validation_dataset = Subset(
    #     full_dataset,
    #     validation_indices.tolist(),
    # )

    training_dataset = Subset(
        training_full_dataset,
        training_indices.tolist(),
    )

    validation_dataset = Subset(
        validation_full_dataset,
        validation_indices.tolist(),
    )

    final_training_labels = _extract_targets(training_dataset)
    final_validation_labels = _extract_targets(validation_dataset)

    # sample, _ = full_dataset[0]
    sample, _ = validation_full_dataset[0]
    input_shape = tuple(int(value) for value in sample.shape)
    data_spec = ClassificationDataSpec(
        name=dataset,
        input_shape=input_shape,
        n_features=int(np.prod(input_shape)),
        n_labels=int(labels.max().item() + 1),
        is_image=True,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size or batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    _attach_loader_metadata(
        training_loader,
        labels=final_training_labels,
        data_spec=data_spec,
    )
    _attach_loader_metadata(
        validation_loader,
        labels=final_validation_labels,
        data_spec=data_spec,
    )

    logger.info(
        "Loaded {} with {} training samples, {} validation samples, "
        "training batch size {}, and validation batch size {}.",
        dataset,
        len(training_dataset),
        len(validation_dataset),
        batch_size,
        validation_batch_size or batch_size,
    )
    return training_loader, validation_loader


def get_image_test_dataloader(
    dataset: str,
    data_dir: str | Path = "data",
    testing_samples: int | None = None,
    batch_size: int = 1,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Loads the official test split for an image dataset.

    Args:
        dataset: One of ``mnist``, ``fashion_mnist``, or ``cifar10``.
        data_dir: Directory used to store the dataset.
        batch_size: Test dataloader batch size.
        download: Whether torchvision may download missing files.
        num_workers: Number of dataloader workers.
        pin_memory: Whether to pin host memory.

    Returns:
        The test dataloader and dataset specification.

    Raises:
        ValueError: If the dataset is not supported or the batch size is
            invalid.
    """
    if dataset not in IMAGE_DATASETS:
        raise ValueError(
            f"Image dataset must be one of {sorted(IMAGE_DATASETS)}, "
            f"received {dataset!r}."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    transform = _image_transform(dataset, training=False)

    constructors = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }

    dataset_class = constructors[dataset]

    test_dataset = dataset_class(
        root=str(Path(data_dir)),
        train=False,
        transform=transform,
        download=download,
    )

    testing_indices = np.arange(len(test_dataset))
    test_labels = _extract_targets(test_dataset)

    selected_testing_positions = _stratified_indices(
        test_labels,
        selected_size=testing_samples,
        seed=42,
    )

    testing_indices = np.asarray(testing_indices)[selected_testing_positions]

    test_dataset = Subset(
        test_dataset,
        testing_indices.tolist(),
    )

    test_labels = _extract_targets(test_dataset)

    sample, _ = test_dataset[0]
    input_shape = tuple(int(value) for value in sample.shape)

    data_spec = ClassificationDataSpec(
        name=dataset,
        input_shape=input_shape,
        n_features=int(np.prod(input_shape)),
        n_labels=int(test_labels.max().item() + 1),
        is_image=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    _attach_loader_metadata(
        test_dataloader,
        labels=test_labels,
        data_spec=data_spec,
    )

    logger.info(
        "Loaded {} official test split with {} samples and batch size {}.",
        dataset,
        len(test_dataset),
        batch_size,
    )

    return test_dataloader


def get_uci_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a supported UCI-style tabular dataset.

    Args:
        dataset: Dataset name.

    Returns:
        Feature matrix and integer labels.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    if dataset == "iris":
        data = load_iris()
        return data.data, data.target

    if dataset == "wine":
        data = load_wine()
        return data.data, data.target

    if dataset == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target

    if dataset == "seeds":
        data = np.loadtxt("src/datasets/classification/data/seeds_dataset.txt")
        return data[:, :7], data[:, 7].astype(int) - 1

    raise ValueError(f"Unknown UCI dataset: {dataset}")


def get_uci_dataloaders(
    dataset: str,
    normalize: str = "minmax",
    training_size: float = 0.8,
    batch_size: int = 1,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Builds fully batched UCI dataloaders.

    Using ``batch_size=1`` preserves sample-by-sample execution while retaining
    the same leading-batch-dimension contract used by image datasets.

    Args:
        dataset: UCI dataset name.
        normalize: One of ``none``, ``zscore``, or ``minmax``.
        training_size: Fraction used for training.
        batch_size: Batch size. Use one for per-sample execution.
        seed: Split seed.

    Returns:
        Training loader, validation loader, and dataset specification.

    Raises:
        ValueError: If arguments are invalid.
    """
    if normalize not in {"none", "zscore", "minmax"}:
        raise ValueError("normalize must be one of 'none', 'zscore', or 'minmax'.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    x, y = get_uci_dataset(dataset)

    if normalize == "minmax":
        x = MinMaxScaler().fit_transform(x) * math.pi

    x = x.astype(np.float32)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        train_size=training_size,
        random_state=seed,
        stratify=y,
    )

    if normalize == "zscore":
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean) / (std + 1e-8)
        x_validation = (x_validation - mean) / (std + 1e-8)

    training_dataset = ClassificationDataset(x_train, y_train)
    validation_dataset = ClassificationDataset(
        x_validation,
        y_validation,
    )

    n_labels = int(max(y.max(), 0) + 1)
    data_spec = ClassificationDataSpec(
        name=dataset,
        input_shape=(x.shape[1],),
        n_features=int(x.shape[1]),
        n_labels=n_labels,
        is_image=False,
    )

    training_labels = training_dataset.y.long()
    label_counts = torch.bincount(
        training_labels,
        minlength=n_labels,
    )
    label_weights = 1.0 / label_counts.clamp_min(1).float()

    if not torch.all(label_counts == label_counts[0]):
        sample_weights = label_weights[training_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            shuffle=False,
        )
    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    _attach_loader_metadata(
        training_loader,
        labels=training_labels,
        data_spec=data_spec,
    )
    _attach_loader_metadata(
        validation_loader,
        labels=validation_dataset.y.long(),
        data_spec=data_spec,
    )

    return training_loader, validation_loader


def get_classification_dataloaders(
    dataset: str,
    **kwargs: Any,
) -> tuple[DataLoader, DataLoader]:
    """Dispatches to the tabular or image dataloader builder.

    Args:
        dataset: Classification dataset name.
        **kwargs: Loader-specific keyword arguments.

    Returns:
        Training loader, validation loader, and dataset specification.

    Raises:
        ValueError: If the dataset is unknown.
    """
    if dataset in IMAGE_DATASETS:
        return get_image_dataloaders(dataset, **kwargs)
    if dataset in UCI_DATASETS:
        return get_uci_dataloaders(dataset, **kwargs)
    raise ValueError(
        f"Unknown classification dataset {dataset!r}. "
        f"Choose from {CLASSIFICATION_DATASETS}."
    )
