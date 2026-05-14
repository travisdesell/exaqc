from __future__ import annotations

import torch
import torch.nn as nn


class LinearImageEncoder(nn.Module):
    """Multi-layer perceptron image encoder for quantum embeddings.

    This module flattens an image and maps it into a low-dimensional
    quantum-compatible embedding using configurable hidden layers.

    Example:
        input -> 512 -> 256 -> 15

    Args:
        input_dim:
            Flattened image dimension.
        embedding_dim:
            Final embedding dimension (usually number of input qubits).
        hidden_dims:
            Hidden layer widths.
        activation:
            Activation module constructor.
        use_sigmoid:
            Whether to squash final outputs into [0, 1].
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: list[int],
        activation: str = "tanh",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.use_sigmoid = activation == "sigmoid"

        dims = [input_dim] + hidden_dims + [embedding_dim]

        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        if self.use_sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image or already-flattened features into quantum embedding."""
        x = x.to(torch.float32)

        single = False

        if x.dim() == 1:
            # Already flattened: [D]
            x = x.unsqueeze(0)
            single = True

        elif x.dim() == 2:
            # Could be [B, D] or grayscale image [H, W]
            if x.shape[0] == 1 or x.shape[0] != self.input_dim:
                x = x.unsqueeze(0)
                single = True
            x = x.flatten(start_dim=1)

        elif x.dim() == 3:
            # Single image: [C, H, W]
            x = x.unsqueeze(0)
            single = True
            x = x.flatten(start_dim=1)

        elif x.dim() == 4:
            # Batch image: [B, C, H, W]
            x = x.flatten(start_dim=1)

        else:
            raise ValueError(f"Unsupported input shape for image encoder: {tuple(x.shape)}")

        z = self.network(x)

        return z.squeeze(0) if single else z