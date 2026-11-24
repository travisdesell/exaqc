"""Shared test helpers for exercising ``src.trainer.supervised_trainer``.

This module is intentionally *not* named ``test_*.py`` so pytest will not try
to collect it directly; it is imported by the ``test_supervised_trainer_*``
modules instead (following the pattern already used by
``tests/innovation_validation.py``).

It centralizes:

* Construction of :class:`~src.circuits.circuit.CircuitGenome` instances of
  varying complexity (number of qubits/gates), for both the ``pennylane`` and
  ``qiskit`` targets.
* Construction of small, class-balanced classification dataloaders.
* Small utilities for inspecting the trainable parameters of a genome's
  encoder, decoder, and quantum circuit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import DECODING_OPTIONS, initialize_decoder
from src.circuits.encoder import ENCODING_OPTIONS, initialize_encoder
from src.circuits.registers import expand_registers

#: Complexity levels used to build circuits of increasing size/depth. Each
#: level maps to the number of qubits used in a single "q" register.
COMPLEXITY_LEVELS: tuple[str, ...] = ("minimal", "shallow", "deep")

#: The ``"multi_param"`` level additionally exercises gates that carry more
#: than one trainable parameter (the ``u`` / ``U3`` gate, three parameters).
#: It is kept out of :data:`COMPLEXITY_LEVELS` so the broad training/gradient
#: suites keep their existing single-parameter-gate coverage, and is opted
#: into explicitly (e.g. by the parameter round-trip tests) where within-gate
#: ordering matters.
COMPLEXITY_LEVELS_WITH_MULTI_PARAM: tuple[str, ...] = COMPLEXITY_LEVELS + (
    "multi_param",
)

#: encoder_name/decoder_name combinations exercised across the test suite.
#: "identity"/"clipped" have no trainable parameters of their own, while
#: "linear"/"linear" are torch.nn.Module-backed and trainable.
ENCODER_DECODER_PAIRS: tuple[tuple[str, str], ...] = (
    ("identity", "clipped"),
    ("linear", "linear"),
)

_N_QUBITS_BY_COMPLEXITY: dict[str, int] = {
    "minimal": 1,
    "shallow": 2,
    "deep": 3,
    "multi_param": 2,
}


def _gate_specs(
    complexity: str, include_parametric: bool, target: str
) -> list[dict[str, object]]:
    """Builds the list of gate specifications for a given complexity level.

    Gate method names are restricted to ones supported by both the
    ``qiskit`` and ``pennylane`` gate specification tables (``h``, ``x``,
    ``rx``, ``ry``, ``rz``, ``cx``, and the multi-parameter ``u`` gate) so
    the same specification can be used to build genomes for either target.

    Note on the "multi_param" level: it includes ``u`` gates (the qiskit
    ``u`` gate / pennylane ``U3`` gate), each carrying three trainable
    parameters, so tests can exercise within-gate parameter ordering rather
    than only cross-gate ordering. The third rotation parameter of this gate
    is named ``"lam"`` in qiskit but ``"delta"`` in pennylane, so the
    parameter dict is built target-aware (as is done for the single-qubit
    ``rx`` rotation, which is ``"theta"`` in qiskit and ``"phi"`` in
    pennylane).

    Note on the "deep" level: an ``h`` gate is deliberately placed
    immediately after the ``rz`` gate. ``rz`` (and other Z-diagonal
    rotations) only changes the *relative phase* between computational
    basis states; when the circuit's output is read out as probabilities
    (``qml.probs`` / a qiskit ``SamplerQNN``), a phase-only change is
    invisible to the measurement unless a subsequent gate (like ``h``)
    converts that phase into an amplitude difference. Without it, the
    gradient of the loss with respect to the ``rz`` parameter is
    mathematically zero, which would look like a broken gradient path
    rather than the expected quantum-mechanical behavior.

    Args:
        complexity: One of :data:`COMPLEXITY_LEVELS`.
        include_parametric: If True, include trainable (parametric) gates
            such as ``rx``/``ry``/``rz``. If False, only structural
            (non-parametric) gates are included, which is used to exercise
            the "no trainable circuit parameters" code path.
        target: the target architecture (pennylane or qiskit) as some gates
            have different parameter names

    Returns:
        A list of keyword-argument dicts suitable for
        ``CircuitGenome.add_gate(**spec)``.

    Raises:
        ValueError: If ``complexity`` is not one of :data:`COMPLEXITY_LEVELS`.
    """

    if complexity == "minimal":
        if include_parametric:
            if target == "qiskit":
                # the rx gate in qiskit uses "theta" instead of "phi"
                return [
                    dict(
                        depth=0.5,
                        method_name="rx",
                        qubits=[("q", 0)],
                        parameters={"theta": 0.4},
                    )
                ]
            else:
                # the rx gate in qiskit uses "theta" instead of "phi"
                return [
                    dict(
                        depth=0.5,
                        method_name="rx",
                        qubits=[("q", 0)],
                        parameters={"phi": 0.4},
                    )
                ]

        return [dict(depth=0.5, method_name="h", qubits=[("q", 0)], parameters={})]

    if complexity == "shallow":
        specs: list[dict[str, object]] = [
            dict(depth=0.2, method_name="h", qubits=[("q", 0)], parameters={}),
        ]
        if include_parametric:
            if target == "qiskit":
                # the rx gate in qiskit uses "theta" instead of "phi"
                specs.append(
                    dict(
                        depth=0.3,
                        method_name="rx",
                        qubits=[("q", 0)],
                        parameters={"theta": 0.4},
                    )
                )
            else:
                specs.append(
                    dict(
                        depth=0.3,
                        method_name="rx",
                        qubits=[("q", 0)],
                        parameters={"phi": 0.4},
                    )
                )
        specs.append(
            dict(
                depth=0.5, method_name="cx", qubits=[("q", 0), ("q", 1)], parameters={}
            )
        )
        if include_parametric:
            specs.append(
                dict(
                    depth=0.6,
                    method_name="ry",
                    qubits=[("q", 1)],
                    parameters={"theta": 0.1},
                )
            )
        return specs

    if complexity == "deep":
        specs = [
            dict(depth=0.1, method_name="h", qubits=[("q", 0)], parameters={}),
        ]
        if include_parametric:
            if target == "qiskit":
                # the rx gate in qiskit uses "theta" instead of "phi"
                specs.append(
                    dict(
                        depth=0.15,
                        method_name="rx",
                        qubits=[("q", 0)],
                        parameters={"theta": 0.4},
                    )
                )
            else:
                specs.append(
                    dict(
                        depth=0.15,
                        method_name="rx",
                        qubits=[("q", 0)],
                        parameters={"phi": 0.4},
                    )
                )

        specs.append(
            dict(
                depth=0.3, method_name="cx", qubits=[("q", 0), ("q", 1)], parameters={}
            )
        )
        if include_parametric:
            specs.append(
                dict(
                    depth=0.35,
                    method_name="ry",
                    qubits=[("q", 1)],
                    parameters={"theta": 0.1},
                )
            )
        specs.append(
            dict(
                depth=0.5, method_name="cx", qubits=[("q", 1), ("q", 2)], parameters={}
            )
        )
        if include_parametric:
            # NOTE: qiskit's RZ gate expects a keyword argument named "phi"
            # (see src/circuits/qiskit_gate_specifications.py); using any
            # other parameter name (e.g. "theta") raises a TypeError deep
            # inside Gate.add_to_qiskit_circuit.
            specs.append(
                dict(
                    depth=0.55,
                    method_name="rz",
                    qubits=[("q", 2)],
                    parameters={"phi": 0.2},
                )
            )
        specs.append(dict(depth=0.6, method_name="h", qubits=[("q", 2)], parameters={}))
        specs.append(dict(depth=0.7, method_name="x", qubits=[("q", 0)], parameters={}))
        return specs

    if complexity == "multi_param":
        # rx's rotation parameter and the u gate's third parameter are named
        # differently between the two targets (see the docstring).
        rx_parameter = "theta" if target == "qiskit" else "phi"
        u_third_parameter = "lam" if target == "qiskit" else "delta"

        specs = [
            dict(depth=0.2, method_name="h", qubits=[("q", 0)], parameters={}),
        ]
        if include_parametric:
            # a single-parameter gate, then a three-parameter gate, so the
            # flattened parameter list mixes gates of different arities
            specs.append(
                dict(
                    depth=0.3,
                    method_name="rx",
                    qubits=[("q", 0)],
                    parameters={rx_parameter: 0.4},
                )
            )
            specs.append(
                dict(
                    depth=0.4,
                    method_name="u",
                    qubits=[("q", 0)],
                    parameters={"theta": 0.1, "phi": 0.2, u_third_parameter: 0.3},
                )
            )
        specs.append(
            dict(
                depth=0.5, method_name="cx", qubits=[("q", 0), ("q", 1)], parameters={}
            )
        )
        if include_parametric:
            # a second three-parameter gate, on a different qubit, so
            # cross-gate ordering of multi-parameter gates is exercised too
            specs.append(
                dict(
                    depth=0.6,
                    method_name="u",
                    qubits=[("q", 1)],
                    parameters={"theta": 0.5, "phi": 0.6, u_third_parameter: 0.7},
                )
            )
        return specs

    raise ValueError(f"Unknown complexity level: {complexity!r}")


def build_classification_genome(
    genome_number: int,
    target: str,
    complexity: str,
    encoder_name: str,
    decoder_name: str,
    include_parametric: bool = True,
    n_classes: int = 2,
    n_features: int = 5,
    learning_rate: float = 0.05,
    epochs: int = 2,
    quantum_input_mode: str = "u3",
    quantum_output_mode: str = "probs",
) -> tuple[CircuitGenome, int]:
    """Builds a fully wired-up (but not yet initialized) classification genome.

    The returned genome has its ``hyperparameters``, ``encoder``, ``decoder``,
    and gates all set, matching what ``SupervisedTrainer.train`` expects
    before it calls ``genome.initialize_model()`` itself.

    Args:
        genome_number: Unique identifier passed through to ``CircuitGenome``.
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: One of :data:`COMPLEXITY_LEVELS`, controlling the number
            of qubits and gates used.
        encoder_name: Either ``"identity"`` or ``"linear"``.
        decoder_name: Either ``"clipped"`` or ``"linear"``.
        include_parametric: Whether to include trainable quantum gates.
        n_classes: Number of output classes the decoder should produce.
        n_features: Number of raw classical input features to use when
            ``encoder_name == "linear"``. Ignored for ``"identity"``, since
            an identity encoding requires the raw feature count to exactly
            match the circuit's expected quantum input count.
        learning_rate: Stored in ``genome.hyperparameters`` for the trainer.
        epochs: Stored in ``genome.hyperparameters`` for the trainer.
        quantum_input_mode: One of ``QUANTUM_INPUT_MODES`` in
            ``src.circuits.circuit`` (``"u3"`` is used by default since it is
            correctly implemented for both targets; see the module docstring
            in ``test_supervised_trainer_epochs.py`` for the ``"rz"`` input
            mode bug found in qiskit's circuit generation).
        quantum_output_mode: One of ``QUANTUM_OUTPUT_MODES``. Only
            ``"probs"`` is used here because qiskit's circuit generation
            does not implement ``"expval"`` (see ``CircuitGenome.
            generate_qiskit_circuit``, which leaves ``self.torch_model`` as
            ``None`` for that mode).

    Returns:
        A tuple ``(genome, raw_n_features)`` where ``raw_n_features`` is the
        length of the raw classical feature vector callers must supply to
        ``genome.forward()`` (and therefore the feature dimension to use when
        building a matching dataset).

    Raises:
        ValueError: If ``encoder_name`` or ``decoder_name`` is not recognized.
    """

    n_qubits = _N_QUBITS_BY_COMPLEXITY[complexity]
    qubits = expand_registers({"q": n_qubits})

    genome = CircuitGenome(
        genome_number=genome_number, target=target, input_qubits=qubits
    )
    genome.hyperparameters = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "quantum_input_mode": quantum_input_mode,
        "quantum_output_mode": quantum_output_mode,
    }

    n_quantum_inputs = genome.n_quantum_inputs()
    raw_n_features = n_quantum_inputs if encoder_name == "identity" else n_features

    if encoder_name in ENCODING_OPTIONS:
        genome.encoder = initialize_encoder(
            target=target,
            encoding_str=encoder_name,
            n_inputs=raw_n_features,
            n_outputs=n_quantum_inputs,
        )
    else:
        raise ValueError(f"Unknown encoder_name: {encoder_name!r}")

    for spec in _gate_specs(complexity, include_parametric, target):
        genome.add_gate(**spec)

    n_quantum_outputs = genome.n_quantum_outputs()
    if decoder_name in DECODING_OPTIONS:
        genome.decoder = initialize_decoder(
            target=target,
            decoding_str=decoder_name,
            n_inputs=n_quantum_outputs,
            n_outputs=n_classes,
        )
    else:
        raise ValueError(f"Unknown decoder_name: {decoder_name!r}")

    return genome, raw_n_features


def make_balanced_binary_dataloaders(
    n_features: int,
    batch_size: int = 2,
    n_train_per_class: int = 2,
    n_val_per_class: int = 1,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Builds tiny, class-balanced binary classification dataloaders.

    Both classes are guaranteed to appear in both the training and
    validation split. This matters because
    :class:`~src.metrics.mean_class_accuracy.MeanClassAccuracy` divides by
    the number of samples seen *per class*; if a class is entirely absent
    from a split, ``MeanClassAccuracy.calculate`` raises
    ``ZeroDivisionError`` (a real bug found while prototyping these tests,
    reported separately). Using a fixed, balanced dataset avoids tripping
    over that unrelated bug so these tests stay focused on
    ``SupervisedTrainer`` itself.

    Args:
        n_features: Number of classical input features per sample.
        batch_size: Batch size for both dataloaders.
        n_train_per_class: Number of training samples to generate per class.
        n_val_per_class: Number of validation samples to generate per class.
        seed: Seed for the torch random generator, for reproducibility.

    Returns:
        A tuple ``(train_dataloader, validation_dataloader)``.
    """

    generator = torch.Generator().manual_seed(seed)

    def _make_split(n_per_class: int) -> TensorDataset:
        class_zero = torch.randn(n_per_class, n_features, generator=generator) - 2.0
        class_one = torch.randn(n_per_class, n_features, generator=generator) + 2.0
        x = torch.cat([class_zero, class_one], dim=0)
        y = torch.cat(
            [
                torch.zeros(n_per_class, dtype=torch.long),
                torch.ones(n_per_class, dtype=torch.long),
            ]
        )
        return TensorDataset(x, y)

    train_loader = DataLoader(_make_split(n_train_per_class), batch_size=batch_size)
    val_loader = DataLoader(_make_split(n_val_per_class), batch_size=batch_size)
    return train_loader, val_loader


def cross_entropy_on_logits(prediction: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy loss for a single, unbatched (prediction, label) pair.

    ``SupervisedTrainer.get_metrics`` calls the loss function once per
    sample with an unbatched prediction vector and a scalar integer label,
    so this wrapper adds the batch dimension ``torch.nn.functional.
    cross_entropy`` expects and removes it again internally.

    Args:
        prediction: Unbatched model output of shape ``(n_classes,)``.
        target: Scalar integer class label tensor.

    Returns:
        A scalar loss tensor.
    """
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)

    if target.ndim == 0:
        target = target.unsqueeze(0)

    return F.cross_entropy(prediction.float(), target.long())


def hybrid_named_parameters(genome: CircuitGenome) -> dict[str, torch.nn.Parameter]:
    """Returns every trainable parameter tracked by the genome's hybrid model.

    Since the refactor to ``CircuitGenome.hybrid_model`` (a
    ``torch.nn.Module`` wrapping the encoder, quantum layer, and decoder),
    a single ``named_parameters()`` call is the canonical view of everything
    the optimizer trains. Parameter names are prefixed by their submodule:

    * ``encoder.*``  -- only present when the encoder is a ``torch.nn.Module``
      (e.g. ``LinearEncoder``); absent for ``IdentityEncoder``.
    * ``decoder.*``  -- only present when the decoder is a ``torch.nn.Module``
      (e.g. ``LinearDecoder``); absent for ``ClippedDecoder``.
    * ``genome.quantum_weight_key`` -- the quantum layer's flat weight tensor
      (``quantum_layer.weights`` for pennylane, ``quantum_layer.weight`` for
      qiskit); always present, but zero-length when there are no parametric
      gates.

    Requires ``genome.initialize_model()`` to have already been called.

    Args:
        genome: The genome whose hybrid model should be inspected.

    Returns:
        A mapping from parameter name to ``torch.nn.Parameter``.
    """

    return dict(genome.hybrid_model.named_parameters())


def encoder_trainable_parameters(genome: CircuitGenome) -> list[torch.nn.Parameter]:
    """Returns the encoder's trainable parameters, if it has any.

    Reads directly from ``genome.encoder`` (rather than the hybrid model) so
    it is usable both before and after ``initialize_model()``; the hybrid
    model wraps this exact object, so the returned parameters are the ones
    the optimizer updates in place during training.

    Args:
        genome: The genome whose encoder should be inspected.

    Returns:
        A list of parameters, or an empty list if the encoder (e.g.
        ``IdentityEncoder``) is not a ``torch.nn.Module``.
    """

    if isinstance(genome.encoder, torch.nn.Module):
        return list(genome.encoder.parameters())
    return []


def decoder_trainable_parameters(genome: CircuitGenome) -> list[torch.nn.Parameter]:
    """Returns the decoder's trainable parameters, if it has any.

    Reads directly from ``genome.decoder`` (rather than the hybrid model) so
    it is usable both before and after ``initialize_model()``; the hybrid
    model wraps this exact object, so the returned parameters are the ones
    the optimizer updates in place during training.

    Args:
        genome: The genome whose decoder should be inspected.

    Returns:
        A list of parameters, or an empty list if the decoder (e.g.
        ``ClippedDecoder``) is not a ``torch.nn.Module``.
    """

    if isinstance(genome.decoder, torch.nn.Module):
        return list(genome.decoder.parameters())
    return []


def circuit_trainable_parameters(genome: CircuitGenome) -> list[torch.nn.Parameter]:
    """Returns the quantum layer's trainable weight tensor, if it has one.

    The refactored ``CircuitGenome`` stores all quantum gate parameters in a
    single flat weight tensor on the quantum layer of the hybrid model
    (``quantum_layer.weights`` for pennylane, ``quantum_layer.weight`` for
    qiskit -- both exposed via ``genome.quantum_weight_key``). This helper
    reads that tensor straight out of ``hybrid_model.named_parameters()``
    rather than reconstructing it from the removed ``get_torch_parameters()``
    method.

    Requires ``genome.initialize_model()`` to have already been called.

    Args:
        genome: The genome whose quantum-layer parameters should be inspected.

    Returns:
        A single-element list with the quantum weight tensor, or an empty
        list when the circuit has no parametric gates (a zero-length weight).
    """

    weight = hybrid_named_parameters(genome).get(genome.quantum_weight_key)
    if weight is None or weight.numel() == 0:
        return []
    return [weight]


def snapshot_gate_parameters(genome: CircuitGenome) -> dict[int, dict[str, float]]:
    """Snapshots the plain-float gate parameters of all enabled gates.

    This mirrors the representation ``CircuitGenome.set_state_dict`` writes
    back into ``gate.parameters`` after training, so it can be compared
    before/after a training run to prove the quantum circuit's own
    parameters actually changed.

    Args:
        genome: The genome whose gates should be snapshotted.

    Returns:
        A dict mapping each enabled gate's innovation number to a copy of
        its parameters dict.
    """

    return {
        gate.innovation_number: dict(gate.parameters)
        for gate in genome.gates
        if gate.enabled
    }


def state_dict_with_quantum_weights(
    genome: CircuitGenome, weights: list[float]
) -> dict[str, torch.Tensor]:
    """Builds a self-consistent hybrid-model state dict with given quantum weights.

    ``CircuitGenome.set_state_dict`` consumes a ``hybrid_model.state_dict()``
    snapshot (the shape the trainer captures for the best epoch) and pushes
    the quantum-layer weights back into ``gate.parameters``. To test that,
    a caller needs a valid state dict whose quantum weights are known values.

    Constructing one by hand is error-prone for the qiskit target, whose
    ``TorchConnector`` exposes the *same* underlying weight tensor under two
    state-dict keys (``quantum_layer.weight`` and ``quantum_layer._weights``)
    that share storage. Setting only one leaves the snapshot internally
    inconsistent. This helper avoids that by writing ``weights`` into the live
    quantum-layer parameter (so every alias updates together) and then
    snapshotting a detached clone of the whole state dict.

    Requires ``genome.initialize_model()`` to have already been called.

    Args:
        genome: The initialized genome to build a state dict for.
        weights: The quantum-layer weight values to embed, in
            ``get_parameters_as_list`` order. Must match the number of
            quantum gate parameters.

    Returns:
        A detached, cloned ``state_dict`` suitable to pass to
        ``genome.set_state_dict``.
    """

    with torch.no_grad():
        weight_param = hybrid_named_parameters(genome)[genome.quantum_weight_key]
        weight_param.copy_(torch.as_tensor(weights, dtype=weight_param.dtype))

    return genome.clone_state_dict()
