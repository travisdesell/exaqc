"""Tests for the generated quantum-teacher imitation dataloaders.

``get_teacher_dataloaders`` builds a teacher circuit, draws random input angles,
and precomputes the teacher's outputs as the targets a student genome is evolved
to reproduce. The loaders carry the same kind of metadata the classification
loaders attach, so the entry point can size a genome straight from a loader.

These tests pin the data contract (shapes, dtypes, metadata, input range,
determinism), that the targets really are the teacher's own outputs, that a
student can actually learn from them, and that invalid requests fail loudly.
"""

from __future__ import annotations

import pytest
import torch

from src.circuits.teacher_circuits import build_teacher_genome
from src.datasets.teacher_loaders import (
    INPUT_ANGLE_RANGE,
    TeacherDataSpec,
    get_teacher_dataloaders,
)
from src.metrics.teacher_metrics import Fidelity
from src.trainer.supervised_trainer import SupervisedTrainer

#: Every target the loaders must support.
_TARGETS = ["pennylane", "qiskit"]


def stacked(loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenates every batch of a loader back into whole tensors.

    Args:
        loader: The dataloader to drain.

    Returns:
        A tuple of the stacked inputs and stacked targets.
    """

    inputs = torch.cat([batch[0] for batch in loader])
    targets = torch.cat([batch[1] for batch in loader])
    return inputs, targets


@pytest.mark.parametrize("target", _TARGETS)
def test_loader_shapes_and_metadata(target: str) -> None:
    """Loaders expose the shapes and metadata needed to size a genome.

    Args:
        target: The quantum backend under test.
    """

    training_loader, validation_loader = get_teacher_dataloaders(
        teacher_name="half_adder",
        input_wires=[0, 1],
        output_wires=[2, 3],
        target=target,
        n_training_samples=16,
        n_validation_samples=8,
        batch_size=4,
    )

    for loader in (training_loader, validation_loader):
        assert loader.n_features == 2
        assert loader.n_targets == 4
        assert loader.input_shape == (2,)
        assert loader.is_image is False
        assert loader.n_wires == 4
        assert loader.input_wires == [0, 1]
        assert loader.output_wires == [2, 3]
        assert isinstance(loader.data_spec, TeacherDataSpec)
        assert loader.data_spec.teacher_name == "half_adder"
        assert loader.data_spec.target == target

    assert len(training_loader.dataset) == 16
    assert len(validation_loader.dataset) == 8

    inputs, targets = next(iter(training_loader))
    assert inputs.shape == (4, 2)
    assert targets.shape == (4, 4)
    assert inputs.dtype == torch.float32
    assert targets.dtype == torch.float32


@pytest.mark.parametrize("target", _TARGETS)
def test_probability_targets_are_normalized(target: str) -> None:
    """``probs`` targets are valid probability distributions.

    Args:
        target: The quantum backend under test.
    """

    training_loader, _ = get_teacher_dataloaders(
        teacher_name="input_controlled_bell",
        input_wires=[0],
        output_wires=[1, 2],
        target=target,
        n_training_samples=12,
        n_validation_samples=4,
        batch_size=4,
    )

    _, targets = stacked(training_loader)

    assert bool((targets >= -1e-6).all())
    assert torch.allclose(targets.sum(dim=-1), torch.ones(len(targets)), atol=1e-4)


@pytest.mark.parametrize("target", _TARGETS)
def test_targets_are_the_teachers_own_outputs(target: str) -> None:
    """Each target is what the teacher circuit produces for that input.

    This is the property the whole task rests on, so it is checked against an
    independently built teacher rather than trusting the loader.

    Args:
        target: The quantum backend under test.
    """

    training_loader, _ = get_teacher_dataloaders(
        teacher_name="bell_out",
        input_wires=[0, 1],
        output_wires=[2, 3],
        target=target,
        n_training_samples=8,
        n_validation_samples=4,
        batch_size=8,
    )

    inputs, targets = stacked(training_loader)

    teacher = build_teacher_genome("bell_out", target, [0, 1], [2, 3])
    teacher.initialize_model()
    with torch.no_grad():
        recomputed = teacher.forward(inputs).float()

    assert torch.allclose(targets, recomputed, atol=1e-5)


def test_inputs_lie_in_the_angle_range() -> None:
    """Generated input angles stay within the documented range."""

    training_loader, validation_loader = get_teacher_dataloaders(
        teacher_name="copy_in_to_out",
        input_wires=[0, 1],
        output_wires=[2, 3],
        n_training_samples=64,
        n_validation_samples=32,
        batch_size=16,
    )

    for loader in (training_loader, validation_loader):
        inputs, _ = stacked(loader)
        assert float(inputs.min()) >= 0.0
        assert float(inputs.max()) <= INPUT_ANGLE_RANGE


def test_training_and_validation_splits_differ() -> None:
    """The two splits are drawn independently, not duplicated."""

    training_loader, validation_loader = get_teacher_dataloaders(
        teacher_name="bell_out",
        input_wires=[0, 1],
        output_wires=[2, 3],
        n_training_samples=16,
        n_validation_samples=16,
        batch_size=16,
    )

    training_inputs, _ = stacked(training_loader)
    validation_inputs, _ = stacked(validation_loader)

    assert not torch.allclose(training_inputs, validation_inputs)


def test_generation_is_deterministic_per_seed() -> None:
    """The same seed regenerates the same data; a different seed does not."""

    def first_inputs(seed: int) -> torch.Tensor:
        """Returns the stacked training inputs generated for a seed.

        Args:
            seed: The seed to generate with.

        Returns:
            The training split's input tensor.
        """
        loader, _ = get_teacher_dataloaders(
            teacher_name="bell_out",
            input_wires=[0, 1],
            output_wires=[2, 3],
            n_training_samples=8,
            n_validation_samples=4,
            batch_size=8,
            seed=seed,
        )
        return stacked(loader)[0]

    assert torch.allclose(first_inputs(0), first_inputs(0))
    assert not torch.allclose(first_inputs(0), first_inputs(1))


def test_expval_targets_on_pennylane() -> None:
    """The ``expval`` readout yields one target per output wire."""

    training_loader, _ = get_teacher_dataloaders(
        teacher_name="bell_out",
        input_wires=[0, 1],
        output_wires=[2, 3],
        target="pennylane",
        quantum_output_mode="expval",
        n_training_samples=8,
        n_validation_samples=4,
        batch_size=4,
    )

    assert training_loader.n_targets == 2

    _, targets = stacked(training_loader)
    assert targets.shape == (8, 2)
    # Pauli-Z expectations live in [-1, 1].
    assert bool((targets >= -1.0 - 1e-5).all() and (targets <= 1.0 + 1e-5).all())


def test_expval_on_qiskit_is_rejected_clearly() -> None:
    """qiskit has no expval model, so it is refused with a clear message."""

    with pytest.raises(ValueError, match="not supported on the qiskit target"):
        get_teacher_dataloaders(
            teacher_name="bell_out",
            input_wires=[0, 1],
            output_wires=[2, 3],
            target="qiskit",
            quantum_output_mode="expval",
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"quantum_input_mode": "u3"}, "Unknown quantum_input_mode"),
        ({"quantum_output_mode": "state"}, "Unknown quantum_output_mode"),
        ({"n_training_samples": 0}, "must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"validation_batch_size": -1}, "validation_batch_size must be positive"),
    ],
)
def test_invalid_requests_are_rejected(kwargs: dict, message: str) -> None:
    """Invalid loader configurations fail loudly.

    Args:
        kwargs: The single invalid override to apply.
        message: Substring expected in the raised error.
    """

    call = {
        "teacher_name": "bell_out",
        "input_wires": [0, 1],
        "output_wires": [2, 3],
        "n_training_samples": 4,
        "n_validation_samples": 4,
    }
    call.update(kwargs)

    with pytest.raises(ValueError, match=message):
        get_teacher_dataloaders(**call)


def test_student_can_learn_a_teacher_from_the_loaders() -> None:
    """A student trained on the generated data converges onto the teacher.

    Starting from the teacher's own architecture with perturbed angles, the
    optimum is known to be reachable, so this checks the whole pipeline --
    loaders, purely quantum genomes, and the shared supervised trainer -- fits
    end to end.
    """

    training_loader, validation_loader = get_teacher_dataloaders(
        teacher_name="2layer_out_block",
        input_wires=[0, 1],
        output_wires=[2, 3],
        target="pennylane",
        n_training_samples=32,
        n_validation_samples=16,
        batch_size=8,
    )

    student = build_teacher_genome("2layer_out_block", "pennylane", [0, 1], [2, 3])
    for gate in student.gates:
        for name in gate.parameters:
            gate.parameters[name] += 0.8
    student.hyperparameters.update(
        {"learning_rate": 0.25, "epochs": 25, "improvement_cutoff": 30}
    )

    SupervisedTrainer(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        training_loss_function=torch.nn.MSELoss(),
        validation_loss_function=torch.nn.MSELoss(),
        metrics={"fidelity": Fidelity()},
    ).train(student)

    history = student.metadata["validation_epoch_metrics"]

    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["fidelity"]["mean"] > 0.99
