"""Classical-to-quantum encoders for the Stage A / Stage B comparison.

All encoders share the interface:

    encoder = SomeEncoder(n_qubits=N, n_features=D, ...)
    encoder.apply(x, theta_enc)   # PennyLane ops appended inside a QNode body
    n_params = encoder.n_params   # length of the theta_enc vector to allocate

The runner allocates a torch tensor of length `encoder.n_params`, passes it
as `theta_enc` on every forward, and includes it in the optimizer's
parameter group when applicable.
"""
from __future__ import annotations

from typing import Iterable, Sequence

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

    def init_params(self, seed: int = 0) -> torch.Tensor:
        """Slopes near pi (so first forward ~ fixed_angle), biases near zero."""
        g = torch.Generator().manual_seed(seed)
        n_pairs = self.n_params // 2
        a = torch.full((n_pairs,), float(torch.pi), dtype=torch.float64) + 0.05 * torch.randn(n_pairs, generator=g, dtype=torch.float64)
        b = 0.05 * torch.randn(n_pairs, generator=g, dtype=torch.float64)
        out = torch.empty(self.n_params, dtype=torch.float64)
        out[0::2] = a
        out[1::2] = b
        return out


class LinearProjectionEncoder:
    """Universal data-reuploading encoder: RY(W @ x + b) per qubit, per layer.

    Generalizes `LearnedAngleEncoder` from feature-cycling (each qubit sees
    one feature) to a *full linear projection* of all features into each
    qubit's rotation angle. With `depth` layers (default 2) interleaved with
    CNOT entanglers, this is a universal data re-uploading scheme — any
    smooth function of x can be approximated to arbitrary accuracy as depth
    and N grow (Pérez-Salinas et al., Quantum 2020, "Data re-uploading for
    a universal quantum classifier").

    Per layer:
        angle_q = (W[layer, q] · x) + b[layer, q]
        apply RY(angle_q) on qubit q

    Then a linear CNOT chain across the register. Trainable parameter
    count: depth * N * (D + 1) — slopes for every (layer, qubit, feature)
    pair plus a per-(layer, qubit) bias. For iris (D=4, N=4, depth=2):
    40 params. For breast_cancer (D=30, N=10, depth=2): 620 params.

    Compare to:
        - FixedAngleEncoder: 0 params, no mixing.
        - LearnedAngleEncoder: 2*N*depth params (single slope+bias per
          qubit, feature-cycling — equivalent to W being a sparse N×D
          matrix with one nonzero per row).
        - LinearProjectionEncoder (this one): full N×D matrix per layer.

    `theta_enc` layout (length depth*N*(D+1)):
        For each (layer in 0..depth-1, qubit in 0..N-1):
            D consecutive entries = W[layer, q, :]
            1 entry             = b[layer, q]
    """

    name = "linear_proj"

    def __init__(self, *, n_qubits: int, n_features: int, depth: int = 2):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.depth = depth
        self._block = n_features + 1  # entries per (layer, qubit)
        self.n_params = depth * n_qubits * self._block

    def apply(self, x: torch.Tensor, theta_enc: torch.Tensor) -> None:
        idx = 0
        for _layer in range(self.depth):
            for q in range(self.n_qubits):
                w = theta_enc[idx : idx + self.n_features]
                b = theta_enc[idx + self.n_features]
                idx += self._block
                # Linear combination of features + bias.
                angle = (w * x).sum() + b
                qml.RY(angle, wires=q)
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

    def init_params(self, seed: int = 0) -> torch.Tensor:
        """Init: W ≈ identity-style sparse (feature q%D gets weight ~pi, rest small),
        biases small. This makes the first forward roughly match `fixed_angle` for
        the natural feature-to-qubit assignment so the learnable comparison
        starts from a sensible baseline."""
        g = torch.Generator().manual_seed(seed)
        out = torch.zeros(self.n_params, dtype=torch.float64)
        idx = 0
        for _layer in range(self.depth):
            for q in range(self.n_qubits):
                # Slopes
                slopes = 0.05 * torch.randn(self.n_features, generator=g, dtype=torch.float64)
                slopes[q % self.n_features] = float(torch.pi) + 0.05 * float(
                    torch.randn(1, generator=g, dtype=torch.float64)
                )
                out[idx : idx + self.n_features] = slopes
                # Bias
                out[idx + self.n_features] = 0.05 * float(
                    torch.randn(1, generator=g, dtype=torch.float64)
                )
                idx += self._block
        return out


class ReuploadEulerEncoder:
    """Multi-axis universal data-reuploading encoder via per-qubit Euler rotations.

    Each qubit per layer receives a Rot(α, β, γ) — three Euler-angle
    rotations — where each angle is its own learnable linear function of
    the input features:

        α_q^l = (W_α[l, q] · x) + b_α[l, q]
        β_q^l = (W_β[l, q] · x) + b_β[l, q]
        γ_q^l = (W_γ[l, q] · x) + b_γ[l, q]
        apply Rot(α, β, γ) on qubit q

    This is a strictly more expressive feature map than
    `LinearProjectionEncoder` (3x the params, 3x the rotation axes) and
    is closer to the canonical Pérez-Salinas universal data-reuploading
    construction: 3 angles per qubit is the most a single-qubit unitary
    can be parameterized by, so we are sweeping every available degree
    of freedom in the encoding rotations.

    Trainable parameter count: 3 * depth * N * (D + 1). For iris
    (D=4, N=4, depth=2): 120 params. For breast_cancer (D=30, N=10,
    depth=2): 1860 params.

    `theta_enc` layout (length 3*depth*N*(D+1)):
        Three back-to-back LinearProjection-style blocks: first one
        feeds α, second feeds β, third feeds γ.
    """

    name = "reupload_euler"

    def __init__(self, *, n_qubits: int, n_features: int, depth: int = 2):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.depth = depth
        self._block = n_features + 1
        self._per_axis = depth * n_qubits * self._block
        self.n_params = 3 * self._per_axis

    def apply(self, x: torch.Tensor, theta_enc: torch.Tensor) -> None:
        # Slice the theta_enc vector into three axis blocks.
        a_block = theta_enc[: self._per_axis]
        b_block = theta_enc[self._per_axis : 2 * self._per_axis]
        c_block = theta_enc[2 * self._per_axis : 3 * self._per_axis]

        idx = 0
        for _layer in range(self.depth):
            for q in range(self.n_qubits):
                w_a = a_block[idx : idx + self.n_features]
                b_a = a_block[idx + self.n_features]
                w_b = b_block[idx : idx + self.n_features]
                b_b = b_block[idx + self.n_features]
                w_c = c_block[idx : idx + self.n_features]
                b_c = c_block[idx + self.n_features]
                idx += self._block

                alpha = (w_a * x).sum() + b_a
                beta = (w_b * x).sum() + b_b
                gamma = (w_c * x).sum() + b_c
                qml.Rot(alpha, beta, gamma, wires=q)
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

    def init_params(self, seed: int = 0) -> torch.Tensor:
        """Init: alpha-block resembles fixed_angle (slope ≈ pi on the natural
        feature for each qubit), beta and gamma blocks small random. So the
        initial circuit is close to a single-axis angle encoding and the other
        two rotation axes are perturbations to be learned."""
        g = torch.Generator().manual_seed(seed)
        out = torch.zeros(self.n_params, dtype=torch.float64)

        # First block (alpha): same init as LinearProjectionEncoder
        idx = 0
        for _layer in range(self.depth):
            for q in range(self.n_qubits):
                slopes = 0.05 * torch.randn(self.n_features, generator=g, dtype=torch.float64)
                slopes[q % self.n_features] = float(torch.pi) + 0.05 * float(
                    torch.randn(1, generator=g, dtype=torch.float64)
                )
                out[idx : idx + self.n_features] = slopes
                out[idx + self.n_features] = 0.05 * float(
                    torch.randn(1, generator=g, dtype=torch.float64)
                )
                idx += self._block

        # Second and third blocks (beta, gamma): small random noise
        for ax in range(1, 3):
            start = ax * self._per_axis
            out[start : start + self._per_axis] = (
                0.05 * torch.randn(self._per_axis, generator=g, dtype=torch.float64)
            )
        return out


class BasisEncoder:
    """Computational-basis encoding via per-feature thresholding.

    Basis encoding prepares a computational-basis state |b_0 b_1 ... b_{N-1}>
    on the N input qubits. Each bit b_q is obtained by *discretizing* a
    classical feature: pick a threshold tau_q for qubit q, set b_q = 1 if
    x[feature_index(q)] > tau_q else 0, and prepare |b_q> on qubit q by
    applying PauliX whenever b_q = 1 (the device default is |0>).

    Why this matters for QML: angle and amplitude encodings keep the input
    in a *continuous* state, so a small change in x produces a small change
    in the state. Basis encoding throws that away — it maps everything to
    one of 2^N discrete states. The upside is interpretability: you can
    point at a qubit and say "qubit q is the bit that says feature i is
    above its median". The downside is that you've quantized your features.

    --- Design choices (the ones you'd reconsider for your own problem) ---

    1. Discretization scheme. We use 1 bit per qubit and per-feature
       MEDIAN thresholds fitted on the *training* set (see
       `fit_basis_thresholds`). Median is robust to skew / outliers and
       gives roughly balanced bins; mean would too if features are
       symmetric. For uniformly-scaled features in [0,1] (the same
       convention `FixedAngleEncoder` assumes), a constant 0.5 works as
       a no-fitting fallback — that's the default if no thresholds are
       supplied.

    2. Qubit-to-feature mapping. Qubit q encodes feature x[q % D] — same
       cyclic pattern as `FixedAngleEncoder`. So when N > D, features
       are *cycled across qubits* (redundancy, not higher resolution).
       When N < D, only the first N features are used (the rest are
       dropped). A more sophisticated allocation would pick the
       highest-variance or highest-mutual-info features for the
       N-qubit budget when N < D.

    3. Bits per feature. We use exactly one bit per qubit. To get K
       bins per feature you'd quantize x_i into log2(K) bits (via
       K-quantile boundaries on the training set) and pack those bits
       across consecutive qubits. Then qubit allocation becomes
       partition-the-bit-budget-across-features. That's a natural
       extension; see the K-bin scheme used in many BasisEmbedding
       tutorials.

    4. Bit ordering (within a multi-bit-per-feature scheme). Plain
       binary is the obvious default, but Gray code preserves locality
       (adjacent bins differ by 1 bit) which can help downstream
       trainability. With 1 bit per feature like we use here, this
       choice doesn't apply.

    --- Trainable parameters: none ---

    Basis encoding's "knobs" are the thresholds and the bit allocation,
    both decided up front from data statistics rather than gradient
    descent. We keep this simple by treating it as a no-train encoder
    (n_params = 0), consistent with `FixedAngleEncoder` and
    `FixedAmplitudeEncoder`.

    --- Implementation note ---

    The Python `if` inside `apply` evaluates a scalar comparison at
    runtime, so the recorded QNode circuit differs sample-to-sample
    (different inputs apply different PauliX subsets). PennyLane re-
    traces the function on each call so this is fine. We use `.item()`
    on the feature/threshold to avoid leaving a torch comparison in
    the circuit body.
    """

    name = "fixed_basis"

    def __init__(
        self,
        *,
        n_qubits: int,
        n_features: int,
        thresholds: torch.Tensor | None = None,
    ):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_params = 0

        if thresholds is None:
            # Sensible no-fitting fallback: assumes features in [0,1].
            thresholds = torch.full((n_qubits,), 0.5, dtype=torch.float64)
        else:
            thresholds = torch.as_tensor(thresholds, dtype=torch.float64)
            if thresholds.shape[0] != n_qubits:
                raise ValueError(
                    f"thresholds must have length n_qubits={n_qubits}, "
                    f"got {thresholds.shape[0]}"
                )
        # Cache as plain Python floats so the per-sample `if` in `apply`
        # doesn't keep dragging torch into the comparison.
        self.thresholds_py: list[float] = [float(t) for t in thresholds]

    def apply(self, x: torch.Tensor, theta_enc: torch.Tensor) -> None:
        # theta_enc unused; signature kept uniform for the runner.
        for q in range(self.n_qubits):
            feat = x[q % self.n_features]
            if float(feat) > self.thresholds_py[q]:
                qml.PauliX(wires=q)


def fit_basis_thresholds(
    train_data: Iterable,
    *,
    n_qubits: int,
    n_features: int,
) -> torch.Tensor:
    """Fit per-qubit basis-encoding thresholds from training data.

    For each qubit q, returns the *median* of feature column (q % D) over
    the training set. The median is the threshold that makes each bin
    (above/below) roughly balanced — about half the training samples flip
    bit q on, half leave it off. For continuous features this is the
    standard fit-on-train discretization choice; for discrete or skewed
    features you might prefer a quantile, the mode, or a domain-specific
    cutoff.

    Args:
        train_data: an iterable of (x, y_onehot, cls) triples — the same
            shape the Stage A / Stage B runners feed encoders. ONLY the
            x's are read; labels are not used.
        n_qubits: how many thresholds to return — one per qubit.
        n_features: feature dimensionality (used for the q % D cycling).

    Returns:
        A length-n_qubits torch.float64 tensor of per-qubit thresholds.

    Reproducibility caveat: thresholds must be fit on TRAIN only.
    Including test data here would let label-correlated structure leak
    into the encoder. That's why this helper takes `train_data` explicitly
    rather than fitting on whatever's handy.
    """
    X = torch.stack(
        [x.to(torch.float64) for x, _, _ in train_data], dim=0
    )
    # Per-feature median across the training set.
    medians = X.median(dim=0).values  # shape (n_features,)
    thresholds = torch.tensor(
        [float(medians[q % n_features]) for q in range(n_qubits)],
        dtype=torch.float64,
    )
    return thresholds


ENCODERS = {
    cls.name: cls
    for cls in (
        FixedAngleEncoder,
        FixedAmplitudeEncoder,
        LearnedAngleEncoder,
        BasisEncoder,
        LinearProjectionEncoder,
        ReuploadEulerEncoder,
    )
}


def make_encoder(
    name: str,
    *,
    n_qubits: int,
    n_features: int,
    **extra_kwargs,
) -> object:
    """Instantiate an encoder by name.

    `extra_kwargs` is forwarded to the encoder's constructor — used by
    BasisEncoder to receive fitted `thresholds`. Other encoders ignore
    it (or you can extend them similarly when you add new ones).
    """
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    cls = ENCODERS[name]
    # Only forward kwargs the constructor actually accepts. We do this
    # the conservative way: try-with, fall back to without. Most encoders
    # don't take extra kwargs and we don't want to break them.
    if extra_kwargs:
        try:
            return cls(n_qubits=n_qubits, n_features=n_features, **extra_kwargs)
        except TypeError:
            pass
    return cls(n_qubits=n_qubits, n_features=n_features)


def initial_encoder_params(encoder_or_count, *, seed: int = 0) -> torch.Tensor:
    """Sensible init for the trainable encoder's params.

    Polymorphic: pass either an encoder instance (preferred — dispatches
    to that encoder's `init_params` for type-aware initialization) or a
    plain `n_params` int (legacy path; uses the (a, b) pair pattern
    appropriate for `LearnedAngleEncoder`).

    The intent is that the first forward of any learnable encoder looks
    roughly like `fixed_angle` so the search starts from a known-good
    baseline and any divergence is attributable to learning.
    """
    # New path: encoder instance with its own init.
    if hasattr(encoder_or_count, "init_params"):
        return encoder_or_count.init_params(seed=seed)

    # Legacy path: bare int. Keep the old (a, b) interleaved layout for
    # backward-compat with callers that pass `encoder.n_params`.
    n_params = int(encoder_or_count)
    if n_params == 0:
        return torch.zeros(0, dtype=torch.float64)
    g = torch.Generator()
    g.manual_seed(seed)
    n_pairs = n_params // 2
    a = torch.full((n_pairs,), float(torch.pi), dtype=torch.float64) + 0.05 * torch.randn(n_pairs, generator=g, dtype=torch.float64)
    b = 0.05 * torch.randn(n_pairs, generator=g, dtype=torch.float64)
    out = torch.empty(n_params, dtype=torch.float64)
    out[0::2] = a
    out[1::2] = b
    return out
