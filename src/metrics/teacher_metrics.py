"""Metrics for the quantum-teacher imitation task.

Each metric reports the mean of one of the per-sample measures defined in
:mod:`src.metrics.teacher_losses` -- the very functions the training losses are
built from -- so a reported number can never drift away from the quantity being
optimized. The classes here hold only the accumulation bookkeeping; none of them
reimplements any of the maths.

Direction of improvement differs by measure: fidelity is a similarity in
``[0, 1]`` that improves as it rises toward 1, while the Bures angle, the KL
divergence and the mean squared error are discrepancies that improve as they
fall toward 0. Each class documents which it is.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from src.metrics.metric import Metric
from src.metrics.teacher_losses import (
    TEACHER_MEASURES,
    bures_angle,
    classical_fidelity,
    kl_divergence,
    mean_squared_error,
)


class PerSampleMeanMetric(Metric):
    """Accumulates the mean of a per-sample measure over an epoch.

    Subclasses supply the measure; this class owns only the running total, which
    is kept as a sample-weighted mean so it is correct regardless of how the
    data is batched.

    Attributes:
        measure: The per-sample function being averaged.
        total: Running sum of the measure over accumulated samples.
        n_samples: How many samples have been accumulated.
    """

    #: The per-sample measure this metric averages. Subclasses override it.
    measure: Callable[[Tensor, Tensor], Tensor] = staticmethod(classical_fidelity)

    def __init__(self) -> None:
        """Initializes the metric with an empty accumulator."""

        self.total = 0.0
        self.n_samples = 0
        self.reset()

    def reset(self) -> None:
        """Clears the accumulator for the beginning of a new epoch."""

        self.total = 0.0
        self.n_samples = 0

    def accumulate(self, output: Tensor, target: Tensor) -> None:
        """Accumulates the measure for one sample or a batch of samples.

        Args:
            output: The student's output, of shape ``[n_outputs]`` for a single
                sample or ``[batch_size, n_outputs]`` for a batch.
            target: The teacher's target, the same shape as ``output``.

        Returns:
            None. Updates the running total and sample count.
        """

        with torch.no_grad():
            values = type(self).measure(output.float(), target.float())

        self.total += float(values.sum().item())
        self.n_samples += int(values.shape[0])

    def calculate(self) -> dict[str, float]:
        """Computes the mean of the measure over the accumulated samples.

        Returns:
            A dict with a single ``"mean"`` entry, or ``0.0`` when nothing has
            been accumulated.
        """

        return {"mean": self.total / self.n_samples if self.n_samples else 0.0}


class Fidelity(PerSampleMeanMetric):
    """Mean classical fidelity between predicted and target distributions.

    A similarity in ``[0, 1]``: **higher is better**, and ``1`` means the
    student reproduces the teacher's distribution exactly.
    """

    measure = staticmethod(classical_fidelity)


class BuresAngle(PerSampleMeanMetric):
    """Mean Bures angle between predicted and target distributions.

    A distance in ``[0, pi/2]`` radians: **lower is better**, and ``0`` means an
    exact match.
    """

    measure = staticmethod(bures_angle)


class KLDivergence(PerSampleMeanMetric):
    """Mean KL divergence from the target distribution to the prediction.

    A discrepancy in nats: **lower is better**, and ``0`` means an exact match.
    """

    measure = staticmethod(kl_divergence)


class MeanSquaredError(PerSampleMeanMetric):
    """Mean squared error between predicted and target outputs.

    A discrepancy: **lower is better**, and ``0`` means an exact match. Unlike
    the distribution measures this makes no assumption about the outputs, so it
    is the one to use for the ``"expval"`` readout.
    """

    measure = staticmethod(mean_squared_error)


#: Every teacher metric, keyed by the same names the losses use.
TEACHER_METRICS: dict[str, type[PerSampleMeanMetric]] = {
    "fidelity": Fidelity,
    "angle": BuresAngle,
    "kl": KLDivergence,
    "mse": MeanSquaredError,
}

#: Metric names, in registry order.
TEACHER_METRIC_NAMES: tuple[str, ...] = tuple(TEACHER_METRICS)

# The metrics and the losses must stay in lockstep: every selectable loss needs
# a metric reporting the same measure.
assert set(TEACHER_METRICS) == set(TEACHER_MEASURES)


def build_teacher_metrics() -> dict[str, PerSampleMeanMetric]:
    """Builds a fresh instance of every teacher metric.

    Every measure is reported each epoch regardless of which one is being
    optimized, so a run can be compared against the others after the fact.

    Returns:
        A mapping from metric name to a new metric instance.
    """

    return {name: metric_class() for name, metric_class in TEACHER_METRICS.items()}
