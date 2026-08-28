from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.models.torchvision_image_classifier import (
    TORCHVISION_IMAGE_MODELS,
    create_torchvision_image_model,
)

CLASSICAL_IMAGE_MODELS = (
    "linear",
    "mlp",
    "cnn",
    *TORCHVISION_IMAGE_MODELS,
)


def _activation(name: str) -> torch.nn.Module:
    """Creates an activation module.

    Args:
        name: Activation name.

    Returns:
        PyTorch activation module.

    Raises:
        ValueError: If the activation is unsupported.
    """
    activations = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "identity": torch.nn.Identity,
    }

    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


def _pooling(
    config: Mapping[str, Any] | None,
) -> torch.nn.Module | None:
    """Creates a two-dimensional pooling layer.

    Args:
        config: Pooling configuration, or ``None`` for no pooling.

    Returns:
        Configured pooling layer or ``None``.

    Raises:
        ValueError: If the pooling type is unsupported.
    """
    if config is None:
        return None

    pool_type = str(config.get("type", "max"))
    kernel_size = int(config.get("kernel_size", 2))
    stride = int(config.get("stride", kernel_size))

    if pool_type == "max":
        return torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
        )

    if pool_type == "avg":
        return torch.nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
        )

    raise ValueError(f"Unsupported pooling type: {pool_type}")


class LinearClassifier(torch.nn.Module):
    """Single-layer image classification baseline."""

    def __init__(
        self,
        input_shape: Sequence[int],
        n_classes: int,
    ) -> None:
        """Initializes a linear image classifier.

        Args:
            input_shape: Per-sample image shape ``[C, H, W]``.
            n_classes: Number of output classes.
        """
        super().__init__()

        self.input_shape = tuple(int(value) for value in input_shape)
        self.n_inputs = 1
        for value in self.input_shape:
            self.n_inputs *= value
        self.n_classes = int(n_classes)

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(self.n_inputs, self.n_classes),
        )

        torch.nn.init.xavier_uniform_(self.model[-1].weight)
        torch.nn.init.zeros_(self.model[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a batched forward pass.

        Args:
            inputs: Image batch shaped ``[B, C, H, W]``.

        Returns:
            Class logits shaped ``[B, n_classes]``.
        """
        return self.model(inputs.float())


class MLPClassifier(torch.nn.Module):
    """Configurable fully connected image classification baseline."""

    def __init__(
        self,
        input_shape: Sequence[int],
        n_classes: int,
        hidden_layers: Sequence[int] = (512, 128),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """Initializes an MLP image classifier.

        Args:
            input_shape: Per-sample image shape ``[C, H, W]``.
            n_classes: Number of output classes.
            hidden_layers: Hidden fully connected layer dimensions.
            activation: Hidden activation function.
            dropout: Dropout probability between hidden layers.

        Raises:
            ValueError: If layer sizes or dropout are invalid.
        """
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_shape = tuple(int(value) for value in input_shape)
        self.n_inputs = 1
        for value in self.input_shape:
            self.n_inputs *= value
        self.n_classes = int(n_classes)
        self.hidden_layers = tuple(int(value) for value in hidden_layers)

        layers: list[torch.nn.Module] = [torch.nn.Flatten(start_dim=1)]
        current_size = self.n_inputs

        for hidden_size in self.hidden_layers:
            if hidden_size <= 0:
                raise ValueError("All hidden layer dimensions must be positive.")

            layers.append(torch.nn.Linear(current_size, hidden_size))
            layers.append(_activation(activation))

            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))

            current_size = hidden_size

        layers.append(torch.nn.Linear(current_size, self.n_classes))

        self.model = torch.nn.Sequential(*layers)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initializes linear layers using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a batched forward pass.

        Args:
            inputs: Image batch shaped ``[B, C, H, W]``.

        Returns:
            Class logits shaped ``[B, n_classes]``.
        """
        return self.model(inputs.float())


class CNNClassifier(torch.nn.Module):
    """Configurable CNN image classification baseline."""

    def __init__(
        self,
        input_shape: Sequence[int],
        n_classes: int,
        conv_blocks: Sequence[Mapping[str, Any]],
        adaptive_pool_size: Sequence[int] = (4, 4),
        fully_connected_layers: Sequence[int] = (),
        dropout: float = 0.0,
    ) -> None:
        """Initializes a configurable CNN classifier.

        Args:
            input_shape: Per-sample image shape ``[C, H, W]``.
            n_classes: Number of output classes.
            conv_blocks: Sequence of convolution block configurations.
            adaptive_pool_size: Output spatial dimensions of adaptive pooling.
            fully_connected_layers: Hidden classifier dimensions.
            dropout: Dropout probability between fully connected layers.

        Raises:
            ValueError: If the architecture configuration is invalid.
        """
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(
                "CNN input_shape must contain channels, height, and width."
            )
        if not conv_blocks:
            raise ValueError("CNNClassifier requires at least one convolution block.")
        if len(adaptive_pool_size) != 2:
            raise ValueError("adaptive_pool_size must contain height and width.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_shape = tuple(int(value) for value in input_shape)
        self.n_classes = int(n_classes)
        self.conv_blocks_config = [dict(block) for block in conv_blocks]
        self.adaptive_pool_size = tuple(int(value) for value in adaptive_pool_size)
        self.fully_connected_layers = tuple(
            int(value) for value in fully_connected_layers
        )
        self.dropout = float(dropout)

        self.features, final_channels = self._build_features()
        self.classifier = self._build_classifier(final_channels)
        self._initialize_parameters()

    def _build_features(
        self,
    ) -> tuple[torch.nn.Sequential, int]:
        """Builds the convolutional feature extractor.

        Returns:
            Feature extractor and number of final output channels.
        """
        layers: list[torch.nn.Module] = []
        in_channels = self.input_shape[0]

        for index, block in enumerate(self.conv_blocks_config):
            out_channels = int(block["out_channels"])
            kernel_size = int(block.get("kernel_size", 3))
            stride = int(block.get("stride", 1))
            padding = int(block.get("padding", 1))
            dilation = int(block.get("dilation", 1))
            batch_norm = bool(block.get("batch_norm", True))

            if out_channels <= 0:
                raise ValueError(f"Conv block {index} has invalid out_channels.")

            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not batch_norm,
                )
            )

            if batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channels))

            layers.append(_activation(str(block.get("activation", "relu"))))

            block_dropout = float(block.get("dropout", 0.0))
            if block_dropout > 0.0:
                layers.append(torch.nn.Dropout2d(block_dropout))

            pooling = _pooling(block.get("pool"))
            if pooling is not None:
                layers.append(pooling)

            in_channels = out_channels

        layers.append(torch.nn.AdaptiveAvgPool2d(self.adaptive_pool_size))

        return torch.nn.Sequential(*layers), in_channels

    def _build_classifier(
        self,
        final_channels: int,
    ) -> torch.nn.Sequential:
        """Builds the fully connected classification head.

        Args:
            final_channels: Number of channels produced by the feature
                extractor.

        Returns:
            Classification head producing raw class logits.
        """
        pooled_height, pooled_width = self.adaptive_pool_size
        current_size = final_channels * pooled_height * pooled_width

        layers: list[torch.nn.Module] = [torch.nn.Flatten(start_dim=1)]

        for hidden_size in self.fully_connected_layers:
            if hidden_size <= 0:
                raise ValueError("Fully connected layer dimensions must be positive.")

            layers.append(torch.nn.Linear(current_size, hidden_size))
            layers.append(torch.nn.ReLU())

            if self.dropout > 0.0:
                layers.append(torch.nn.Dropout(self.dropout))

            current_size = hidden_size

        layers.append(torch.nn.Linear(current_size, self.n_classes))
        return torch.nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        """Initializes convolutional and linear model parameters."""
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a batched CNN forward pass.

        Args:
            inputs: Image batch shaped ``[B, C, H, W]``.

        Returns:
            Raw class logits shaped ``[B, n_classes]``.
        """
        features = self.features(inputs.float())
        return self.classifier(features)


def create_classical_image_model(
    model_name: str,
    input_shape: Sequence[int],
    n_classes: int,
    config: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    """Creates a classical image classification baseline.

    Args:
        model_name: One of ``linear``, ``mlp``, or ``cnn``.
        input_shape: Per-sample image shape.
        n_classes: Number of output classes.
        config: Optional architecture configuration.

    Returns:
        Configured PyTorch model.

    Raises:
        ValueError: If the model name is unsupported.
    """
    model_config = dict(config or {})

    if model_name == "linear":
        return LinearClassifier(
            input_shape=input_shape,
            n_classes=n_classes,
        )

    if model_name == "mlp":
        return MLPClassifier(
            input_shape=input_shape,
            n_classes=n_classes,
            hidden_layers=model_config.get(
                "hidden_layers",
                (512, 128),
            ),
            activation=str(model_config.get("activation", "relu")),
            dropout=float(model_config.get("dropout", 0.0)),
        )

    if model_name == "cnn":
        return CNNClassifier(
            input_shape=input_shape,
            n_classes=n_classes,
            conv_blocks=model_config.get(
                "conv_blocks",
                [
                    {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "batch_norm": True,
                        "activation": "relu",
                        "pool": {
                            "type": "max",
                            "kernel_size": 2,
                        },
                    },
                    {
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "batch_norm": True,
                        "activation": "relu",
                        "pool": {
                            "type": "max",
                            "kernel_size": 2,
                        },
                    },
                ],
            ),
            adaptive_pool_size=model_config.get(
                "adaptive_pool_size",
                (4, 4),
            ),
            fully_connected_layers=model_config.get(
                "fully_connected_layers",
                (128,),
            ),
            dropout=float(model_config.get("dropout", 0.0)),
        )

    if model_name in TORCHVISION_IMAGE_MODELS:
        return create_torchvision_image_model(
            model_name=model_name,
            input_shape=input_shape,
            n_classes=n_classes,
            pretrained=bool(model_config.get("pretrained", False)),
            small_image_stem=bool(model_config.get("small_image_stem", True)),
        )

    raise ValueError(
        f"Unknown classical image model {model_name!r}. "
        f"Choose from {CLASSICAL_IMAGE_MODELS}."
    )
