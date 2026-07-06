from __future__ import annotations

import torch
import torch.nn as nn


class CNNImageEncoder(nn.Module):
    """CNN image encoder for quantum embeddings.

    Maps raw image tensors into a low-dimensional quantum-compatible embedding.

    Args:
        in_channels: Number of image channels. Use 1 for MNIST/Fashion-MNIST
            and 3 for CIFAR-10.
        embedding_dim: Output dimension, usually equal to the number of quantum
            input features.
        image_size: Input image size. Use 28 for MNIST/Fashion-MNIST and 32
            for CIFAR-10.
        conv_channels: Channel widths for convolution blocks.
        hidden_dims: Optional MLP hidden layers after convolution.
        activation: Final activation, either ``"tanh"`` or ``"sigmoid"``.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        image_size: int,
        conv_channels: list[int] | None = None,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
    ):
        super().__init__()

        conv_channels = conv_channels or [16, 32]
        hidden_dims = hidden_dims or []

        layers = []
        c_prev = in_channels

        for c in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(c_prev, c, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            c_prev = c

        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            flat_dim = int(self.conv(dummy).flatten(start_dim=1).shape[1])

        dims = [flat_dim] + hidden_dims + [embedding_dim]

        mlp_layers = []
        for i in range(len(dims) - 1):
            mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp_layers.append(nn.ReLU())

        if activation == "sigmoid":
            mlp_layers.append(nn.Sigmoid())
        elif activation == "tanh":
            mlp_layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unknown activation={activation}")

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensors into quantum-compatible embeddings."""
        x = x.to(torch.float32)

        single = False

        if x.dim() == 2:
            # [H, W]
            x = x.unsqueeze(0).unsqueeze(0)
            single = True

        elif x.dim() == 3:
            # [C, H, W]
            x = x.unsqueeze(0)
            single = True

        elif x.dim() == 4:
            # [B, C, H, W]
            pass

        else:
            raise ValueError(
                f"CNNImageEncoder expects image tensor [H,W], [C,H,W], or [B,C,H,W], "
                f"got {tuple(x.shape)}"
            )

        z = self.conv(x)
        z = z.flatten(start_dim=1)
        z = self.mlp(z)

        return z.squeeze(0) if single else z