"""Tests for the teacher-imitation losses and the metrics that report them.

Each measure is written once in :mod:`src.metrics.teacher_losses` and backs both
a training loss and a metric in :mod:`src.metrics.teacher_metrics`, so these
tests check the maths once and then check that the two layers agree.

The properties pinned here are the ones that make the measures trustworthy:
identical distributions score perfectly, disjoint ones score worst, known
closed-form values come out right, gradients stay finite where a student
converges onto its teacher (the classic ``inf * 0 = nan`` hazard, since teacher
targets are frequently exactly zero), and every loss actually trains.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.circuits.teacher_circuits import build_teacher_genome
from src.datasets.teacher_loaders import get_teacher_dataloaders
from src.metrics.teacher_losses import (
    TEACHER_LOSS_NAMES,
    TEACHER_LOSSES,
    TEACHER_MEASURES,
    bures_angle,
    classical_fidelity,
    get_teacher_loss,
    kl_divergence,
    mean_squared_error,
)
from src.metrics.teacher_metrics import (
    TEACHER_METRICS,
    BuresAngle,
    Fidelity,
    KLDivergence,
    MeanSquaredError,
    build_teacher_metrics,
)
from src.trainer.supervised_trainer import SupervisedTrainer

#: A few distributions to exercise the measures with.
_DISTRIBUTIONS = torch.tensor(
    [
        [0.25, 0.25, 0.25, 0.25],
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ]
)


def test_identical_distributions_score_perfectly() -> None:
    """A perfect match scores 1 fidelity and 0 on every discrepancy."""

    predictions = _DISTRIBUTIONS
    targets = _DISTRIBUTIONS.clone()

    fidelity = classical_fidelity(predictions, targets)
    assert torch.allclose(fidelity, torch.ones(len(predictions)), atol=1e-6)

    assert torch.allclose(
        bures_angle(predictions, targets), torch.zeros(len(predictions)), atol=1e-3
    )
    assert torch.allclose(
        kl_divergence(predictions, targets), torch.zeros(len(predictions)), atol=1e-6
    )
    assert torch.allclose(
        mean_squared_error(predictions, targets),
        torch.zeros(len(predictions)),
        atol=1e-12,
    )


def test_fidelity_is_bounded_by_one() -> None:
    """Fidelity never exceeds 1, including for sparse distributions."""

    fidelity = classical_fidelity(_DISTRIBUTIONS, _DISTRIBUTIONS.clone())

    assert bool((fidelity <= 1.0 + 1e-6).all())
    assert bool((fidelity >= 0.0).all())


def test_disjoint_distributions_score_worst() -> None:
    """Distributions with no overlap give zero fidelity and a right angle."""

    predictions = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[0.0, 1.0]])

    assert float(classical_fidelity(predictions, targets)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(bures_angle(predictions, targets)) == pytest.approx(
        math.pi / 2, abs=1e-5
    )


def test_known_closed_form_values() -> None:
    """The measures match values worked out by hand."""

    predictions = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[0.5, 0.5]])

    # F = (sqrt(1 * 0.5) + sqrt(0 * 0.5))^2 = 0.5
    assert float(classical_fidelity(predictions, targets)) == pytest.approx(
        0.5, abs=1e-5
    )
    assert float(bures_angle(predictions, targets)) == pytest.approx(
        math.acos(math.sqrt(0.5)), abs=1e-5
    )

    # MSE over [1, 0] vs [0.5, 0.5] = mean(0.25, 0.25) = 0.25
    assert float(mean_squared_error(predictions, targets)) == pytest.approx(0.25)


def test_kl_divergence_is_asymmetric_and_non_negative() -> None:
    """KL is a non-negative, direction-dependent discrepancy."""

    first = torch.tensor([[0.7, 0.3]])
    second = torch.tensor([[0.4, 0.6]])

    forward = float(kl_divergence(first, second))
    backward = float(kl_divergence(second, first))

    assert forward > 0.0
    assert backward > 0.0
    assert forward != pytest.approx(backward)


def test_measures_accept_a_single_sample() -> None:
    """A 1-D sample is treated as a batch of one."""

    prediction = torch.tensor([0.5, 0.5])
    target = torch.tensor([0.5, 0.5])

    for measure in TEACHER_MEASURES.values():
        value = measure(prediction, target)
        assert value.shape == (1,)


@pytest.mark.parametrize("name", TEACHER_LOSS_NAMES)
def test_losses_are_scalar_and_differentiable(name: str) -> None:
    """Every loss returns a scalar with a usable gradient.

    Args:
        name: The loss under test.
    """

    predictions = torch.tensor([[0.4, 0.3, 0.2, 0.1]], requires_grad=True)
    targets = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    loss = TEACHER_LOSSES[name](predictions, targets)

    assert loss.ndim == 0
    loss.backward()

    assert predictions.grad is not None
    assert bool(torch.isfinite(predictions.grad).all())
    assert bool((predictions.grad != 0).any())


@pytest.mark.parametrize("name", TEACHER_LOSS_NAMES)
def test_gradients_stay_finite_at_a_perfect_match(name: str) -> None:
    """Gradients do not blow up or go NaN where the student matches the teacher.

    Teacher targets are frequently exactly zero, which is where a naive
    ``sqrt(p * q)`` hands autograd ``inf * 0 = nan``.

    Args:
        name: The loss under test.
    """

    predictions = torch.tensor([[0.5, 0.5, 0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.5, 0.5, 0.0, 0.0]])

    TEACHER_LOSSES[name](predictions, targets).backward()

    assert bool(torch.isfinite(predictions.grad).all())
    assert not bool(torch.isnan(predictions.grad).any())


def test_fidelity_loss_is_one_minus_fidelity() -> None:
    """The fidelity loss inverts the similarity so it can be minimized."""

    predictions = torch.tensor([[0.7, 0.3]])
    targets = torch.tensor([[0.4, 0.6]])

    fidelity = float(classical_fidelity(predictions, targets))
    loss = float(TEACHER_LOSSES["fidelity"](predictions, targets))

    assert loss == pytest.approx(1.0 - fidelity, abs=1e-6)


@pytest.mark.parametrize(
    "metric_class, measure",
    [
        (Fidelity, classical_fidelity),
        (BuresAngle, bures_angle),
        (KLDivergence, kl_divergence),
        (MeanSquaredError, mean_squared_error),
    ],
)
def test_metric_reports_the_mean_of_its_measure(metric_class, measure) -> None:
    """Each metric reports exactly the mean of the function it is built from.

    Args:
        metric_class: The metric under test.
        measure: The per-sample function it should average.
    """

    predictions = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])
    targets = torch.tensor([[0.4, 0.6], [0.3, 0.7], [0.5, 0.5]])

    metric = metric_class()
    for prediction, target in zip(predictions, targets):
        metric.accumulate(prediction, target)

    expected = float(measure(predictions, targets).mean())

    assert metric.calculate()["mean"] == pytest.approx(expected, abs=1e-6)


def test_metric_reset_clears_accumulation() -> None:
    """Resetting a metric discards the previous epoch's samples."""

    metric = Fidelity()
    metric.accumulate(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert metric.calculate()["mean"] == pytest.approx(0.0, abs=1e-6)

    metric.reset()
    assert metric.calculate() == {"mean": 0.0}

    metric.accumulate(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert metric.calculate()["mean"] == pytest.approx(1.0, abs=1e-6)


def test_metric_mean_is_batch_size_independent() -> None:
    """Accumulating in batches gives the same mean as one sample at a time."""

    predictions = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]])
    targets = torch.tensor([[0.4, 0.6], [0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])

    one_at_a_time = Fidelity()
    for prediction, target in zip(predictions, targets):
        one_at_a_time.accumulate(prediction, target)

    batched = Fidelity()
    batched.accumulate(predictions[:3], targets[:3])
    batched.accumulate(predictions[3:], targets[3:])

    assert batched.calculate()["mean"] == pytest.approx(
        one_at_a_time.calculate()["mean"], abs=1e-6
    )


def test_losses_and_metrics_cover_the_same_names() -> None:
    """Every selectable loss has a metric reporting the same measure."""

    assert set(TEACHER_LOSSES) == set(TEACHER_METRICS) == set(TEACHER_MEASURES)
    assert set(build_teacher_metrics()) == set(TEACHER_LOSS_NAMES)


def test_unknown_loss_is_rejected() -> None:
    """An unknown loss name is reported with the available choices."""

    with pytest.raises(ValueError, match="Unknown teacher loss"):
        get_teacher_loss("not_a_loss")


@pytest.mark.parametrize("name", TEACHER_LOSS_NAMES)
def test_each_loss_trains_a_student_onto_its_teacher(name: str) -> None:
    """Optimizing any of the losses closes the gap to the teacher.

    Args:
        name: The loss being optimized.
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
        for parameter_name in gate.parameters:
            gate.parameters[parameter_name] += 0.8
    student.hyperparameters.update(
        {"learning_rate": 0.25, "epochs": 20, "improvement_cutoff": 25}
    )

    loss = get_teacher_loss(name)
    SupervisedTrainer(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        training_loss_function=loss,
        validation_loss_function=loss,
        metrics=build_teacher_metrics(),
    ).train(student)

    history = student.metadata["validation_epoch_metrics"]

    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["fidelity"]["mean"] > history[0]["fidelity"]["mean"]
    assert history[-1]["fidelity"]["mean"] > 0.99

    # every measure is reported, whichever one was optimized
    for metric_name in TEACHER_LOSS_NAMES:
        assert metric_name in history[-1]
