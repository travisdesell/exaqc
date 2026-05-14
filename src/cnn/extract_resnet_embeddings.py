from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def setup_logger() -> None:
    """Configure Loguru logging."""
    os.makedirs("logs/resnet", exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | {message}",
    )

    logger.add(
        "logs/resnet/{time:YYYY-MM-DD_HH-mm-ss}.log",
        level="DEBUG",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


@dataclass
class Config:
    """Configuration for training or extraction."""

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
    embedding_dim: int
    split: str
    num_workers: int
    device: str
    embedding_save_dir: str


def build_resnet_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    """Build a torchvision ResNet model.

    Args:
        backbone_name: Name of the ResNet backbone.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A torchvision ResNet model.

    Raises:
        ValueError: If the backbone is unsupported.
    """
    backbone_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights),
        "resnet34": (models.resnet34, models.ResNet34_Weights),
        "resnet50": (models.resnet50, models.ResNet50_Weights),
        "resnet101": (models.resnet101, models.ResNet101_Weights),
        "resnet152": (models.resnet152, models.ResNet152_Weights),
    }

    if backbone_name not in backbone_map:
        raise ValueError(
            "Unsupported backbone. Choose from: "
            "resnet18, resnet34, resnet50, resnet101, resnet152."
        )

    model_fn, weights_class = backbone_map[backbone_name]
    weights = weights_class.DEFAULT if pretrained else None
    return model_fn(weights=weights)


class ResNetEmbeddingModel(nn.Module):
    """ResNet encoder with an embedding head and classifier.

    The backbone produces a feature vector, which is projected into a
    low-dimensional embedding space. The classifier is trained on top of
    that embedding.

    Args:
        backbone_name: Which ResNet variant to use.
        embedding_dim: Embedding size, must be <= 15.
        num_classes: Number of classes.
        in_channels: Number of input channels.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        num_classes: int,
        in_channels: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        if embedding_dim > 15:
            raise ValueError("embedding_dim must be <= 15.")

        backbone = build_resnet_backbone(backbone_name, pretrained)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.backbone = backbone
        self.embedding_head = nn.Linear(feature_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Tuple of:
                - embeddings of shape [B, embedding_dim]
                - logits of shape [B, num_classes]
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        logits = self.classifier(embedding)
        return embedding, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only embeddings.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Embeddings of shape [B, embedding_dim].
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return embedding


def build_transforms(
    dataset_name: str,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Create train and test transforms.

    Args:
        dataset_name: Dataset name.

    Returns:
        A tuple of (train_transform, test_transform).
    """
    if dataset_name in ["mnist", "fashion_mnist"]:
        mean, std = (0.1307,), (0.3081,)
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError("dataset_name must be one of: mnist, fashion_mnist, cifar10.")

    return train_transform, test_transform


def build_datasets(
    dataset_name: str,
    data_dir: str,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, int]:
    """Create train and test datasets.

    Args:
        dataset_name: Dataset name.
        data_dir: Dataset directory.

    Returns:
        train_dataset, test_dataset, num_classes, in_channels
    """
    train_transform, test_transform = build_transforms(dataset_name)

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
        raise ValueError("dataset_name must be one of: mnist, fashion_mnist, cifar10.")

    return train_dataset, test_dataset, num_classes, in_channels


def build_dataloaders(
    cfg: Config,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Create train and test dataloaders.

    Args:
        cfg: Runtime configuration.

    Returns:
        train_loader, test_loader, num_classes, in_channels
    """
    train_dataset, test_dataset, num_classes, in_channels = build_datasets(
        cfg.dataset,
        cfg.data_dir,
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
    """Compute batch accuracy.

    Args:
        logits: Logits tensor.
        targets: Ground-truth labels.

    Returns:
        Batch accuracy.
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: ResNetEmbeddingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        loader: Training dataloader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Average loss and average accuracy for the epoch.
    """
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
    model: ResNetEmbeddingModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model.

    Args:
        model: Model to evaluate.
        loader: Evaluation dataloader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Average loss and average accuracy.
    """
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
    model: ResNetEmbeddingModel,
    cfg: Config,
    num_classes: int,
    in_channels: int,
    path: str,
) -> None:
    """Save a checkpoint.

    Args:
        model: Model to save.
        cfg: Runtime configuration.
        num_classes: Number of classes.
        in_channels: Input channel count.
        path: Output checkpoint path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": cfg.dataset,
            "backbone": cfg.backbone,
            "pretrained": cfg.pretrained,
            "embedding_dim": cfg.embedding_dim,
            "num_classes": num_classes,
            "in_channels": in_channels,
        },
        path,
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ResNetEmbeddingModel, dict]:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.
        device: Torch device.

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ResNetEmbeddingModel(
        backbone_name=ckpt["backbone"],
        embedding_dim=ckpt["embedding_dim"],
        num_classes=ckpt["num_classes"],
        in_channels=ckpt["in_channels"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_embeddings(
    model: ResNetEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and aligned labels.

    The returned embeddings and labels are index-aligned:
    embeddings[i] corresponds to labels[i].

    Args:
        model: Trained model.
        loader: DataLoader to process.
        device: Torch device.

    Returns:
        Tuple of:
            - embeddings array with shape [N, embedding_dim]
            - labels array with shape [N]
    """
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
    """Run training mode.

    Args:
        cfg: Runtime configuration.
    """
    train_loader, test_loader, num_classes, in_channels = build_dataloaders(cfg)
    device = torch.device(cfg.device)

    logger.info(f"Using device: {device}")
    logger.info(
        f"Initializing model | backbone={cfg.backbone} | "
        f"embedding_dim={cfg.embedding_dim} | pretrained={cfg.pretrained}"
    )

    model = ResNetEmbeddingModel(
        backbone_name=cfg.backbone,
        embedding_dim=cfg.embedding_dim,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=cfg.pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = -1.0
    default_ckpt = os.path.join(
        cfg.output_dir,
        f"{cfg.dataset}_{cfg.backbone}_emb{cfg.embedding_dim}_b{cfg.batch_size}_ep{cfg.epochs}.pt",
    )

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

    Saves both embeddings and aligned class labels for train and test sets.

    Args:
        cfg: Runtime configuration.
    """
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required in extract mode.")

    train_loader, test_loader, _, _ = build_dataloaders(cfg)
    device = torch.device(cfg.device)

    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {cfg.checkpoint}")

    model, ckpt = load_model_from_checkpoint(cfg.checkpoint, device)

    save_dir = os.path.join(cfg.embedding_save_dir, cfg.dataset)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Saving embeddings to: {save_dir}")

    logger.info("Extracting TRAIN embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    np.save(os.path.join(save_dir, "train_embeddings.npy"), train_embeddings)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
    logger.debug(f"Train embeddings shape: {train_embeddings.shape}")
    logger.debug(f"Train labels shape: {train_labels.shape}")

    logger.info("Extracting TEST embeddings...")
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    np.save(os.path.join(save_dir, "test_embeddings.npy"), test_embeddings)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)
    logger.debug(f"Test embeddings shape: {test_embeddings.shape}")
    logger.debug(f"Test labels shape: {test_labels.shape}")

    meta = {
        "dataset": cfg.dataset,
        "backbone": ckpt["backbone"],
        "embedding_dim": ckpt["embedding_dim"],
        "num_classes": ckpt["num_classes"],
        "in_channels": ckpt["in_channels"],
        "pretrained": ckpt.get("pretrained", False),
    }
    np.save(os.path.join(save_dir, "meta.npy"), meta)

    logger.success("Embedding extraction completed.")


def parse_args() -> Config:
    """Parse command-line arguments.

    Returns:
        Parsed configuration object.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a configurable ResNet embedding model and extract "
            "low-dimensional embeddings."
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
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="ResNet backbone.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights.",
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
        default=128,
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
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=15,
        help="Embedding dimension. Must be <= 15.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Unused during current extract mode; both splits are saved.",
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
        default="./src/embeddings/images",
        help="Base directory to save extracted embeddings.",
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
        embedding_dim=args.embedding_dim,
        split=args.split,
        num_workers=args.num_workers,
        device=args.device,
        embedding_save_dir=args.embedding_save_dir,
    )


def main() -> None:
    """Program entry point."""
    setup_logger()
    cfg = parse_args()

    logger.info(f"Full config: {cfg}")

    if cfg.embedding_dim > 15:
        raise ValueError("Embedding dimension must be <= 15.")

    if cfg.mode == "train":
        train_mode(cfg)
    elif cfg.mode == "extract":
        extract_mode(cfg)
    else:
        raise ValueError("Unsupported mode.")


if __name__ == "__main__":
    main()
