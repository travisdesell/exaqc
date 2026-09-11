"""Dataloaders for the quantum-teacher imitation task.

A teacher-imitation dataset is *generated* rather than loaded from disk: a
reference circuit (the "teacher", see
:mod:`src.circuits.teacher_circuits`) is built as a
:class:`~src.circuits.circuit.CircuitGenome`, random inputs are drawn, and the
teacher's outputs for those inputs become the targets a student genome is
evolved to reproduce.

The teacher's outputs are computed once, up front, so training never re-runs
the teacher circuit. Each returned loader carries the same kind of metadata the
classification loaders attach (``n_features``, ``input_shape``, ``data_spec``,
...), so the entry point can size a genome straight from a loader.

Teacher and student share one encoding contract: both are genomes with the same
``quantum_input_mode`` and ``quantum_output_mode`` and no classical encoder or
decoder, so an input vector means the same thing to both and the imitation task
is well posed. Inputs are drawn in ``[0, pi]``, the natural range for the
angle-encoding modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.circuits.circuit import CircuitGenome
from src.circuits.teacher_circuits import TEACHER_NAMES, build_teacher_genome

#: Quantum input modes a teacher dataset can be generated for. Each maps one
#: input value onto one input wire, so a sample is one angle per input wire.
#: ``u3`` and ``amplitude`` are excluded: they do not have that one-value-per-
#: wire correspondence, which the generated inputs rely on.
TEACHER_INPUT_MODES: tuple[str, ...] = ("rx", "ry", "rz")

#: Quantum output modes a teacher dataset can be generated for.
TEACHER_OUTPUT_MODES: tuple[str, ...] = ("probs", "expval")

#: Upper bound of the uniform range that input angles are drawn from. A full
#: half-turn sweeps an angle-encoded qubit from |0> to |1>.
INPUT_ANGLE_RANGE: float = torch.pi


@dataclass(frozen=True)
class TeacherDataSpec:
    """Describes the shape of a quantum teacher-imitation dataset.

    Attributes:
        teacher_name: Name of the teacher circuit being imitated.
        target: Quantum backend the teacher was executed on.
        n_wires: Total number of wires the teacher spans.
        input_wires: Wires driven by the generated inputs.
        output_wires: Wires whose output is imitated (the readout wires).
        input_shape: Per-sample input shape.
        n_features: Number of input values per sample.
        n_targets: Number of values in each teacher target vector.
        quantum_input_mode: Encoding used to place inputs onto the input wires.
        quantum_output_mode: Readout mode used for the targets.
    """

    teacher_name: str
    target: str
    n_wires: int
    input_wires: tuple[int, ...]
    output_wires: tuple[int, ...]
    input_shape: tuple[int, ...]
    n_features: int
    n_targets: int
    quantum_input_mode: str
    quantum_output_mode: str


def _attach_loader_metadata(
    loader: DataLoader,
    data_spec: TeacherDataSpec,
    teacher_genome: CircuitGenome,
) -> None:
    """Attaches EXAQC metadata to a teacher dataloader.

    These attributes mirror the ones the classification loaders attach, so the
    same entry-point code can size a genome from a loader.

    Args:
        loader: Dataloader to annotate.
        data_spec: Specification describing the loader's samples.
        teacher_genome: The teacher the targets were generated from, attached so
            callers can inspect or draw the circuit being imitated.

    Returns:
        None. Sets ``n_features``, ``n_targets``, ``input_shape``, ``is_image``,
        ``n_wires``, ``input_wires``, ``output_wires``, ``data_spec`` and
        ``teacher_genome`` on ``loader``.
    """

    loader.n_features = data_spec.n_features
    loader.n_targets = data_spec.n_targets
    loader.input_shape = data_spec.input_shape
    loader.is_image = False
    loader.n_wires = data_spec.n_wires
    loader.input_wires = list(data_spec.input_wires)
    loader.output_wires = list(data_spec.output_wires)
    loader.data_spec = data_spec
    loader.teacher_genome = teacher_genome


def _teacher_targets(
    teacher_genome: CircuitGenome,
    inputs: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    """Runs the teacher over the inputs to produce the imitation targets.

    Args:
        teacher_genome: An initialized teacher genome.
        inputs: Input angles of shape ``[n_samples, n_features]``.
        batch_size: How many samples to evaluate per forward pass.

    Returns:
        A ``float32`` tensor of teacher outputs, one row per input sample.
    """

    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            outputs = teacher_genome.forward(inputs[start : start + batch_size])
            rows.append(outputs.detach().float())

    return torch.cat(rows, dim=0)


def get_teacher_dataloaders(
    teacher_name: str,
    input_wires: list[int],
    output_wires: list[int],
    target: str = "pennylane",
    quantum_input_mode: str = "ry",
    quantum_output_mode: str = "probs",
    n_training_samples: int = 64,
    n_validation_samples: int = 64,
    batch_size: int = 1,
    validation_batch_size: int | None = None,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Builds training and validation teacher-imitation dataloaders.

    The teacher circuit is built and run once: random input angles are drawn for
    the training and validation splits and the teacher's outputs for them are
    precomputed as the targets. The two splits are drawn independently from the
    same distribution, so validation measures how well a student generalizes to
    inputs it was not fitted on.

    Args:
        teacher_name: One of
            :data:`~src.circuits.teacher_circuits.TEACHER_NAMES`.
        input_wires: Wires driven by the generated inputs.
        output_wires: Wires whose output is imitated.
        target: Quantum backend to run the teacher on. Students must use the
            same backend, since the two index a multi-qubit readout in opposite
            qubit orders.
        quantum_input_mode: One of :data:`TEACHER_INPUT_MODES`.
        quantum_output_mode: One of :data:`TEACHER_OUTPUT_MODES`.
        n_training_samples: Number of training samples to generate.
        n_validation_samples: Number of validation samples to generate.
        batch_size: Training batch size.
        validation_batch_size: Validation batch size. Defaults to ``batch_size``.
        seed: Seed for the generated inputs and the training shuffle.

    Returns:
        A tuple of the training loader and the validation loader, each carrying
        the metadata described in :func:`_attach_loader_metadata`.

    Raises:
        ValueError: If a mode, sample count or batch size is invalid. Invalid
            teachers, targets and wire layouts are reported by
            :func:`~src.circuits.teacher_circuits.build_teacher_genome`.
    """

    if quantum_input_mode not in TEACHER_INPUT_MODES:
        raise ValueError(
            f"Unknown quantum_input_mode={quantum_input_mode!r}; choices: "
            f"{list(TEACHER_INPUT_MODES)}"
        )
    if quantum_output_mode not in TEACHER_OUTPUT_MODES:
        raise ValueError(
            f"Unknown quantum_output_mode={quantum_output_mode!r}; choices: "
            f"{list(TEACHER_OUTPUT_MODES)}"
        )
    if quantum_output_mode == "expval" and target == "qiskit":
        # The qiskit backend does not build a model for the expval readout, so
        # it would fail deep inside initialize_model with an unhelpful
        # AttributeError. Say so here instead.
        raise ValueError(
            "quantum_output_mode='expval' is not supported on the qiskit "
            "target; use 'probs', or run this teacher on pennylane."
        )
    if n_training_samples <= 0 or n_validation_samples <= 0:
        raise ValueError(
            "The number of training and validation samples must be positive."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if validation_batch_size is not None and validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive.")

    teacher_genome = build_teacher_genome(
        teacher_name=teacher_name,
        target=target,
        input_wires=input_wires,
        output_wires=output_wires,
        quantum_input_mode=quantum_input_mode,
        quantum_output_mode=quantum_output_mode,
    )
    teacher_genome.initialize_model()

    n_features = teacher_genome.n_quantum_inputs()

    generator = torch.Generator().manual_seed(seed)
    training_inputs = (
        torch.rand(
            (n_training_samples, n_features),
            generator=generator,
            dtype=torch.float32,
        )
        * INPUT_ANGLE_RANGE
    )
    validation_inputs = (
        torch.rand(
            (n_validation_samples, n_features),
            generator=generator,
            dtype=torch.float32,
        )
        * INPUT_ANGLE_RANGE
    )

    training_targets = _teacher_targets(teacher_genome, training_inputs)
    validation_targets = _teacher_targets(teacher_genome, validation_inputs)

    data_spec = TeacherDataSpec(
        teacher_name=teacher_name,
        target=target,
        n_wires=len(teacher_genome.qubits),
        input_wires=tuple(input_wires),
        output_wires=tuple(output_wires),
        input_shape=(n_features,),
        n_features=n_features,
        n_targets=training_targets.shape[1],
        quantum_input_mode=quantum_input_mode,
        quantum_output_mode=quantum_output_mode,
    )

    shuffle_generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        TensorDataset(training_inputs, training_targets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_inputs, validation_targets),
        batch_size=validation_batch_size or batch_size,
        shuffle=False,
        drop_last=False,
    )

    _attach_loader_metadata(training_loader, data_spec, teacher_genome)
    _attach_loader_metadata(validation_loader, data_spec, teacher_genome)

    logger.info(
        "Generated teacher '{}' on {} ({} -> {}) with {} training and {} "
        "validation samples, {} features, {} targets, training batch size {}, "
        "validation batch size {}.",
        teacher_name,
        target,
        quantum_input_mode,
        quantum_output_mode,
        n_training_samples,
        n_validation_samples,
        data_spec.n_features,
        data_spec.n_targets,
        batch_size,
        validation_batch_size or batch_size,
    )

    return training_loader, validation_loader


__all__ = [
    "INPUT_ANGLE_RANGE",
    "TEACHER_INPUT_MODES",
    "TEACHER_NAMES",
    "TEACHER_OUTPUT_MODES",
    "TeacherDataSpec",
    "get_teacher_dataloaders",
]
