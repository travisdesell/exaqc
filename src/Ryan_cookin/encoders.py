"""Classical-to-quantum encoders for the Stage A comparison.

All encoders share the interface:

    encoder = SomeEncoder(n_qubits=N, n_features=D, ...)
    encoder.apply(x, theta_enc)   # PennyLane ops appended inside a QNode body
    n_params = encoder.n_params   # length of the theta_enc vector to allocate

The runner allocates a torch tensor of length `encoder.n_params`, passes it
as `theta_enc` on every forward, and includes it in the optimizer's
parameter group when applicable.
"""
from __future__ import annotations

from typing import Sequence

import pennylane as qml
import torch


class FixedAngleEncoder:
    """Baseline: RY(pi * x_i) per qubit, no trainable parameters.

    If `n_qubits > n_features`, features are recycled (cycled). If
    `n_qubits < n_features`, the surplus features are ignored. This
    matches the simplest interpretation of "angle encoding".
    """

    name = "fixed_angle"

    def __init__(self, *, n_qubits: int, n_features: int):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_params = 0

    def apply(self, x: torch.Tensor, theta_enc: torch.Tensor) -> None:
        # theta_enc unused; signature kept uniform for the runner.
        for q in range(self.n_qubits):
            feat = x[q % self.n_features]
            qml.RY(torch.pi * feat, wires=q)


class FixedAmplitudeEncoder:
    """Baseline: AmplitudeEmbedding into a single block of qubits.

    Pads / truncates the feature vector to length 2**n_qubits. When the
    feature count exceeds 2**n_qubits the tail is dropped; we do not try
    to be clever about which features to keep, since the goal of this
    baseline is to be the canonical amplitude encoding, not an optimized
    one. Vector is L2-normalized inside this method so the user does
    not have to.
    """

    name = "fixed_amplitude"

    def __init__(self, *, n_qubits: int, n_features: int):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_params = 0
        self._target_len = 2 ** n_qubits

    def apply(self, x: torch.Tensor, theta_enc: torch.Tensor) -> None:
        # Pad or truncate to target length.
        if x.shape[0] < self._target_len:
            pad = torch.zeros(self._target_len - x.shape[0], dtype=x.dtype)
            features = torch.cat([x, pad], dim=0)
        else:
            features = x[: self._target_len]
        # Normalize. Add a tiny eps so an all-zero input doesn't NaN.
        norm = torch.linalg.norm(features) + 1e-12
        features = features / norm
        qml.AmplitudeEmbedding(
            features=features,
            wires=list(range(self.n_qubits)),
            normalize=False,
        )


class LearnedAngleEncoder:
    """Data re-uploading style trainable encoder.

    For each of `depth` layers and each of `n_qubits` qubits, we apply
    `RY(a * x[q % D] + b)` with `(a, b)` learnable scalars, then a CNOT
    chain across the register as an entangler. Total trainable
    parameters: `2 * n_qubits * depth`.

    This is a deliberately small encoder so it fits cleanly into the
    same optimizer step as the downstream ansatz. The point of Stage A
    is to see whether *any* learnable encoding helps relative to the
    fixed baselines, not to find the best one.
    """

    name = "learned"

    def __init__(self, *, n_qubits: int, n_features: int, depth: int = 2):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.depth = depth
        # 2 params per (layer, qubit): one slope `a`, one bias `b`.
        self.n_params = 2 * n_qubits * depth

    def apply(self, x: torch.Tensor, theta_enc: torch.Tensor) -> None:
        idx = 0
        for layer in range(self.depth):
            for q in range(self.n_qubits):
                a = theta_enc[idx]
                idx += 1
                b = theta_enc[idx]
                idx += 1
                feat = x[q % self.n_features]
                qml.RY(a * feat + b, wires=q)
            # Linear CNOT entangler.
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])


ENCODERS = {
    cls.name: cls
    for cls in (FixedAngleEncoder, FixedAmplitudeEncoder, LearnedAngleEncoder)
}


def make_encoder(name: str, *, n_qubits: int, n_features: int) -> object:
    """Instantiate an encoder by name."""
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    return ENCODERS[name](n_qubits=n_qubits, n_features=n_features)


def initial_encoder_params(n_params: int, *, seed: int = 0) -> torch.Tensor:
    """Reasonable init for the trainable encoder.

    Slopes (`a`) start near pi (so the learned encoder begins close to
    the fixed_angle baseline) and biases (`b`) start near zero. This
    keeps the encoder identifiable at step 0 — the comparison starts
    with learned ~= fixed_angle and any divergence over training is
    attributable to the encoder learning.
    """
    if n_params == 0:
        return torch.zeros(0, dtype=torch.float64)
    g = torch.Generator()
    g.manual_seed(seed)
    # Init pattern: pairs of (a, b). Slope ~ pi + small noise, bias ~ small noise.
    n_pairs = n_params // 2
    a = torch.full((n_pairs,), float(torch.pi), dtype=torch.float64) + 0.05 * torch.randn(n_pairs, generator=g, dtype=torch.float64)
    b = 0.05 * torch.randn(n_pairs, generator=g, dtype=torch.float64)
    out = torch.empty(n_params, dtype=torch.float64)
    out[0::2] = a
    out[1::2] = b
    return out
