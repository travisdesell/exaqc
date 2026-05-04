#!/usr/bin/env python3
"""
Train a DenseNet or WideResNet image encoder and extract low-dimensional embeddings.

Supported datasets
------------------
- mnist
- fashion_mnist
- cifar10

Supported backbones
-------------------
- densenet121
- densenet169
- densenet201
- wide_resnet50_2
- wide_resnet101_2

Embedding head
--------------
Use --embedding-dims as a single nargs list:
- --embedding-dims 15              -> feature_dim -> 15
- --embedding-dims 64 15           -> feature_dim -> 64 -> 15
- --embedding-dims 256 64 15       -> feature_dim -> 256 -> 64 -> 15

The final value in --embedding-dims is the extracted embedding size and must be <= 15.

Examples
--------
Train a DenseNet121 on CIFAR-10 with a direct 15-d embedding:
    python train_extract_image_embeddings.py \
        --dataset cifar10 \
        --mode train \
        --backbone densenet121 \
        --embedding-dims 15 \
        --epochs 20 \
        --pretrained

Train a WideResNet with one hidden embedding layer 64 -> 15:
    python train_extract_image_embeddings.py \
        --dataset cifar10 \
        --mode train \
        --backbone wide_resnet50_2 \
        --embedding-dims 64 15 \
        --epochs 20 \
        --pretrained

Train with two hidden embedding layers 256 -> 64 -> 15:
    python train_extract_image_embeddings.py \
        --dataset cifar10 \
        --mode train \
        --backbone densenet169 \
        --embedding-dims 256 64 15 \
        --epochs 20 \
        --pretrained

Freeze backbone and train only embedding/classifier layers:
    python train_extract_image_embeddings.py \
        --dataset mnist \
        --mode train \
        --backbone densenet121 \
        --pretrained \
        --freeze-backbone \
        --embedding-dims 64 15

Extract embeddings after training:
    python train_extract_image_embeddings.py \
        --dataset cifar10 \
        --mode extract \
        --checkpoint outputs/cifar10_densenet121_emb-64x15.pt \
        --embedding-save-dir src/embeddings/images
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def setup_logger() -> None:
    """Configure Loguru logging."""
    os.makedirs("logs", exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | {message}",
    )

    logger.add(
        "logs/{time:YYYY-MM-DD_HH-mm-ss}.log",
        level="DEBUG",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


@dataclass
class Config:
    """Runtime configuration."""

    dataset: str
    mode: str
    backbone: str
    pretrained: bool
    data_dir: str
    output_dir: str
    checkpoint: str | None
    batch_size: int
    epochs: int
    lr: float
    embedding_dims: list[int]
    split: str
    num_workers: int
    device: str
    embedding_save_dir: str
    image_size: int
    freeze_backbone: bool


def build_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    """Build a torchvision DenseNet or WideResNet backbone.

    Args:
        backbone_name: Backbone name.
        pretrained: Whether to load pretrained weights.

    Returns:
        Instantiated torchvision model.

    Raises:
        ValueError: If the backbone is unsupported.
    """
    if backbone_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
    elif backbone_name == "densenet169":
        weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
        model = models.densenet169(weights=weights)
    elif backbone_name == "densenet201":
        weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
    elif backbone_name == "wide_resnet50_2":
        weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        model = models.wide_resnet50_2(weights=weights)
    elif backbone_name == "wide_resnet101_2":
        weights = models.Wide_ResNet101_2_Weights.DEFAULT if pretrained else None
        model = models.wide_resnet101_2(weights=weights)
    else:
        raise ValueError(
            "Unsupported backbone. Choose from: "
            "densenet121, densenet169, densenet201, "
            "wide_resnet50_2, wide_resnet101_2."
        )
    return model


def adapt_first_layer_for_input_channels(model: nn.Module, in_channels: int) -> None:
    """Adapt the first convolution layer to support 1-channel inputs.

    Args:
        model: Backbone model.
        in_channels: Number of input channels.
    """
    if in_channels == 3:
        return

    if hasattr(model, "features") and hasattr(model.features, "conv0"):
        old_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
    elif hasattr(model, "conv1"):
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
    else:
        raise ValueError("Unsupported model structure for input-channel adaptation.")


def get_feature_dim_and_strip_classifier(model: nn.Module) -> int:
    """Remove the classifier layer and return the backbone feature dimension.

    Args:
        model: Backbone model.

    Returns:
        Feature dimension produced by the backbone.
    """
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
        return feature_dim

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return feature_dim

    raise ValueError("Unsupported model structure for classifier stripping.")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    """Build an MLP with ReLU activations between hidden layers.

    Args:
        input_dim: Input dimension.
        hidden_dims: Hidden layer sizes.
        output_dim: Output dimension.

    Returns:
        MLP as nn.Sequential.
    """
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))

        is_last = i == len(dims) - 2
        if not is_last:
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ImageEmbeddingModel(nn.Module):
    """Image encoder with configurable low-dimensional embedding head.

    Architecture:
        image -> backbone -> embedding_head -> classifier

    Args:
        backbone_name: Backbone model name.
        embedding_dims: Embedding head dimensions. Final value is output size.
        num_classes: Number of classes.
        in_channels: Input channel count.
        pretrained: Whether to use pretrained backbone weights.
        freeze_backbone: Whether to freeze backbone parameters.
    """

    def __init__(
        self,
        backbone_name: str,
        embedding_dims: Sequence[int],
        num_classes: int,
        in_channels: int,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if not embedding_dims:
            raise ValueError("embedding_dims must contain at least one value.")

        if embedding_dims[-1] > 15:
            raise ValueError("Final embedding dimension must be <= 15.")

        backbone = build_backbone(backbone_name, pretrained)
        adapt_first_layer_for_input_channels(backbone, in_channels)
        feature_dim = get_feature_dim_and_strip_classifier(backbone)

        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.backbone = backbone
        self.embedding_dims = list(embedding_dims)
        self.embedding_head = build_mlp(
            input_dim=feature_dim,
            hidden_dims=embedding_dims[:-1],
            output_dim=embedding_dims[-1],
        )
        self.classifier = nn.Linear(embedding_dims[-1], num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings and logits.

        Args:
            x: Input tensor.

        Returns:
            Tuple of:
                - embedding tensor [B, final_embedding_dim]
                - logits tensor [B, num_classes]
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        logits = self.classifier(embedding)
        return embedding, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only embeddings.

        Args:
            x: Input tensor.

        Returns:
            Embedding tensor [B, final_embedding_dim].
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return embedding


def build_transforms(
    dataset_name: str,
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Create train and test transforms.

    Args:
        dataset_name: Dataset name.
        image_size: Resize target.

    Returns:
        Train and test transforms.
    """
    if dataset_name in ["mnist", "fashion_mnist"]:
        mean, std = (0.1307,), (0.3081,)
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError(
            "dataset_name must be one of: mnist, fashion_mnist, cifar10."
        )

    return train_transform, test_transform


def build_datasets(
    dataset_name: str,
    data_dir: str,
    image_size: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, int]:
    """Create train/test datasets.

    Args:
        dataset_name: Dataset name.
        data_dir: Data directory.
        image_size: Resize target.

    Returns:
        train_dataset, test_dataset, num_classes, in_channels
    """
    train_transform, test_transform = build_transforms(dataset_name, image_size)

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
        num_classes = 10
        in_channels = 1

    elif dataset_name == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
        num_classes = 10
        in_channels = 1

    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
        num_classes = 10
        in_channels = 3

    else:
        raise ValueError(
            "dataset_name must be one of: mnist, fashion_mnist, cifar10."
        )

    return train_dataset, test_dataset, num_classes, in_channels


def build_dataloaders(
    cfg: Config,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Create train/test dataloaders.

    Args:
        cfg: Runtime config.

    Returns:
        train_loader, test_loader, num_classes, in_channels
    """
    train_dataset, test_dataset, num_classes, in_channels = build_datasets(
        dataset_name=cfg.dataset,
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"Dataset loaded | dataset={cfg.dataset} | "
        f"train_size={len(train_dataset)} | test_size={len(test_dataset)}"
    )

    return train_loader, test_loader, num_classes, in_channels


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute batch accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: ImageEmbeddingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        _, logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)
        total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


@torch.no_grad()
def evaluate(
    model: ImageEmbeddingModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _, logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)
        total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def save_checkpoint(
    model: ImageEmbeddingModel,
    cfg: Config,
    num_classes: int,
    in_channels: int,
    path: str,
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": cfg.dataset,
            "backbone": cfg.backbone,
            "pretrained": cfg.pretrained,
            "embedding_dims": cfg.embedding_dims,
            "num_classes": num_classes,
            "in_channels": in_channels,
            "image_size": cfg.image_size,
            "freeze_backbone": cfg.freeze_backbone,
        },
        path,
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ImageEmbeddingModel, dict]:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = ImageEmbeddingModel(
        backbone_name=ckpt["backbone"],
        embedding_dims=ckpt["embedding_dims"],
        num_classes=ckpt["num_classes"],
        in_channels=ckpt["in_channels"],
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_embeddings(
    model: ImageEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and aligned labels."""
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        embeddings = model.encode(images)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    embeddings_np = np.concatenate(all_embeddings, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return embeddings_np, labels_np


def train_mode(cfg: Config) -> None:
    """Run training mode."""
    train_loader, test_loader, num_classes, in_channels = build_dataloaders(cfg)
    device = torch.device(cfg.device)

    logger.info(f"Using device: {device}")
    logger.info(
        f"Initializing model | backbone={cfg.backbone} | "
        f"embedding_dims={cfg.embedding_dims} | pretrained={cfg.pretrained} | "
        f"freeze_backbone={cfg.freeze_backbone}"
    )

    model = ImageEmbeddingModel(
        backbone_name=cfg.backbone,
        embedding_dims=cfg.embedding_dims,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params}/{total_params}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_name = (
        f"{cfg.dataset}_{cfg.backbone}_"
        f"emb-{'x'.join(map(str, cfg.embedding_dims))}.pt"
    )
    default_ckpt = os.path.join(cfg.output_dir, ckpt_name)

    best_test_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        logger.info(
            f"[{cfg.dataset} | {cfg.backbone}] "
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                model=model,
                cfg=cfg,
                num_classes=num_classes,
                in_channels=in_channels,
                path=default_ckpt,
            )
            logger.success(f"Saved best checkpoint to: {default_ckpt}")

    logger.info(f"Best test accuracy: {best_test_acc:.4f}")


@torch.no_grad()
def extract_mode(cfg: Config) -> None:
    """Run extraction mode.

    Saves embeddings and aligned labels for both train and test splits.
    """
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required in extract mode.")

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {cfg.checkpoint}")

    model, ckpt = load_model_from_checkpoint(cfg.checkpoint, device)
    image_size = ckpt.get("image_size", cfg.image_size)

    extract_cfg = Config(
        dataset=cfg.dataset,
        mode=cfg.mode,
        backbone=ckpt["backbone"],
        pretrained=False,
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        checkpoint=cfg.checkpoint,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        embedding_dims=ckpt["embedding_dims"],
        split=cfg.split,
        num_workers=cfg.num_workers,
        device=cfg.device,
        embedding_save_dir=cfg.embedding_save_dir,
        image_size=image_size,
        freeze_backbone=False,
    )

    train_loader, test_loader, _, _ = build_dataloaders(extract_cfg)

    save_dir = os.path.join(cfg.embedding_save_dir, cfg.dataset)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Saving embeddings to: {save_dir}")

    logger.info("Extracting TRAIN embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    np.save(os.path.join(save_dir, "train_embeddings.npy"), train_embeddings)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)

    logger.info("Extracting TEST embeddings...")
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    np.save(os.path.join(save_dir, "test_embeddings.npy"), test_embeddings)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)

    meta = {
        "dataset": cfg.dataset,
        "backbone": ckpt["backbone"],
        "embedding_dims": ckpt["embedding_dims"],
        "num_classes": ckpt["num_classes"],
        "in_channels": ckpt["in_channels"],
        "pretrained": ckpt.get("pretrained", False),
        "image_size": ckpt.get("image_size", image_size),
        "freeze_backbone": ckpt.get("freeze_backbone", False),
    }
    np.save(os.path.join(save_dir, "meta.npy"), meta)

    logger.success("Embedding extraction completed.")
    logger.info(f"Train embeddings shape: {train_embeddings.shape}")
    logger.info(f"Train labels shape: {train_labels.shape}")
    logger.info(f"Test embeddings shape: {test_embeddings.shape}")
    logger.info(f"Test labels shape: {test_labels.shape}")


def parse_args() -> Config:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a DenseNet or WideResNet encoder and extract "
            "low-dimensional image embeddings."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "extract"],
        help="Run mode.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=[
            "densenet121",
            "densenet169",
            "densenet201",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ],
        help="Backbone model.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained ImageNet weights.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone parameters and train only embedding/classifier layers.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for extraction mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        nargs="+",
        default=[15],
        help=(
            "Embedding MLP dimensions. "
            "Examples: '15' for direct feature->15, "
            "'64 15' for feature->64->15, "
            "'256 64 15' for feature->256->64->15. "
            "Final value must be <= 15."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Unused during current extract mode; both train and test are saved.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    parser.add_argument(
        "--embedding-save-dir",
        type=str,
        default="src/embeddings/images",
        help="Base directory to save extracted embeddings.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image resize used before feeding into the model.",
    )

    args = parser.parse_args()

    return Config(
        dataset=args.dataset,
        mode=args.mode,
        backbone=args.backbone,
        pretrained=args.pretrained,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        embedding_dims=args.embedding_dims,
        split=args.split,
        num_workers=args.num_workers,
        device=args.device,
        embedding_save_dir=args.embedding_save_dir,
        image_size=args.image_size,
        freeze_backbone=args.freeze_backbone,
    )


def main() -> None:
    """Program entry point."""
    setup_logger()
    cfg = parse_args()
    logger.info(f"Full config: {cfg}")

    if not cfg.embedding_dims:
        raise ValueError("embedding_dims must contain at least one value.")

    if cfg.embedding_dims[-1] > 15:
        raise ValueError("Final embedding dimension must be <= 15.")

    if cfg.mode == "train":
        train_mode(cfg)
    elif cfg.mode == "extract":
        extract_mode(cfg)
    else:
        raise ValueError("Unsupported mode.")


if __name__ == "__main__":
    main()