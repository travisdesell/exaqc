"""Tests that ``SupervisedTrainer`` supports float (non-classification) targets.

The trainer is shared by classification and regression-style tasks such as
quantum-teacher imitation. Classification yields integer class indices, while
teacher imitation yields float target *vectors* (a probability distribution per
sample). The trainer therefore passes targets to the loss function and to every
metric exactly as the dataloader yielded them, rather than coercing them to
integers -- coercion would silently truncate a distribution like
``[0.25, 0.25, 0.25, 0.25]`` to ``[0, 0, 0, 0]``.

These tests pin that contract end to end: a purely quantum genome trained
against another genome's float outputs must actually reduce its loss and raise
its fidelity, and the targets must arrive at the loss/metric unmodified.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers
from src.metrics.metric import Metric
from src.metrics.teacher_metrics import Fidelity
from src.trainer.supervised_trainer import SupervisedTrainer


def build_quantum_only_genome(
    first_angle: float,
    second_angle: float,
    epochs: int = 8,
) -> CircuitGenome:
    """Builds a purely quantum genome with two trainable rotations.

    Args:
        first_angle: Initial angle of the rotation on the first qubit.
        second_angle: Initial angle of the rotation on the second qubit.
        epochs: Training epochs recorded in the genome's hyperparameters.

    Returns:
        An initialized-model-ready :class:`CircuitGenome` with no encoder or
        decoder, so its outputs are the raw circuit probabilities.
    """

    genome = CircuitGenome(
        genome_number=1,
        target="pennylane",
        input_qubits=expand_registers({"q": 2}),
    )
    genome.hyperparameters = {
        "quantum_input_mode": "ry",
        "quantum_output_mode": "probs",
        "learning_rate": 0.3,
        "epochs": epochs,
        "improvement_cutoff": 10,
    }
    genome.encoder = None
    genome.decoder = None

    genome.add_gate(
        depth=0.4, method_name="ry", qubits=[("q", 0)], parameters={"phi": first_angle}
    )
    genome.add_gate(depth=0.5, method_name="cx", qubits=[("q", 0), ("q", 1)])
    genome.add_gate(
        depth=0.6, method_name="ry", qubits=[("q", 1)], parameters={"phi": second_angle}
    )
    return genome


def make_teacher_dataloader(
    teacher: CircuitGenome,
    n_samples: int = 16,
    batch_size: int = 4,
) -> tuple[DataLoader, torch.Tensor]:
    """Builds a dataloader whose targets are a teacher genome's float outputs.

    Args:
        teacher: An initialized genome used to label the synthetic inputs.
        n_samples: Number of samples to generate.
        batch_size: Batch size for the returned loader.

    Returns:
        A tuple of the dataloader and the float target tensor.
    """

    torch.manual_seed(0)
    inputs = torch.rand(n_samples, teacher.n_quantum_inputs(), dtype=torch.float32)
    with torch.no_grad():
        targets = teacher.forward(inputs).float()
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size), targets


class _TargetRecordingMetric(Metric):
    """Metric that records the dtype and shape of every target it receives."""

    def __init__(self) -> None:
        """Initializes the recorder."""
        self.seen: list[tuple[torch.dtype, tuple[int, ...]]] = []
        self.reset()

    def reset(self) -> None:
        """Clears the recorded targets for a new epoch."""
        self.seen = []

    def accumulate(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Records the target's dtype and shape.

        Args:
            output: The model prediction (unused).
            target: The target as handed over by the trainer.
        """
        self.seen.append((target.dtype, tuple(target.shape)))

    def calculate(self) -> int:
        """Returns how many targets were recorded.

        Returns:
            The number of accumulated targets.
        """
        return len(self.seen)


def test_float_vector_targets_train_and_improve() -> None:
    """A purely quantum genome learns to imitate another genome's outputs."""

    teacher = build_quantum_only_genome(0.9, 1.3)
    teacher.initialize_model()
    loader, targets = make_teacher_dataloader(teacher)

    assert targets.dtype == torch.float32
    assert targets.ndim == 2

    student = build_quantum_only_genome(0.1, 0.2)
    student.initialize_model()

    trainer = SupervisedTrainer(
        training_dataloader=loader,
        validation_dataloader=loader,
        training_loss_function=torch.nn.MSELoss(),
        validation_loss_function=torch.nn.MSELoss(),
        metrics={"fidelity": Fidelity()},
    )
    trainer.train(student)

    history = student.metadata["training_epoch_metrics"]
    assert len(history) >= 2

    # The student starts away from the teacher and must measurably close the gap.
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["fidelity"]["mean"] > history[0]["fidelity"]["mean"]
    assert history[-1]["fidelity"]["mean"] > 0.9


def test_targets_reach_metrics_unmodified() -> None:
    """Float target vectors arrive at metrics without being coerced to integers.

    Casting targets to ``long`` would truncate a probability distribution to all
    zeros, so this guards the exact regression that silently zeroed fidelity.
    """

    teacher = build_quantum_only_genome(0.9, 1.3, epochs=1)
    teacher.initialize_model()
    loader, targets = make_teacher_dataloader(teacher, n_samples=8)

    student = build_quantum_only_genome(0.1, 0.2, epochs=1)
    student.initialize_model()

    recorder = _TargetRecordingMetric()
    trainer = SupervisedTrainer(
        training_dataloader=loader,
        validation_dataloader=loader,
        training_loss_function=torch.nn.MSELoss(),
        validation_loss_function=torch.nn.MSELoss(),
        metrics={"recorded": recorder},
    )
    trainer.train(student)

    assert recorder.seen, "the metric should have received targets"
    for dtype, shape in recorder.seen:
        assert dtype == torch.float32
        assert shape == (targets.shape[1],)
