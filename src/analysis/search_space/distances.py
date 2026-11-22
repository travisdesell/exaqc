"""Distances between genomes, for projecting a run's search space.

A :class:`DistanceMetric` turns each serialized genome into a *feature* (a set of
gate innovations, a unitary, a vector of outputs...) and then measures every pair
of features at once. Splitting the two steps is what makes the search-space view
cheap to keep current on a live run: features depend on one genome only, so they
are cached and only new genomes are featurized, while the pairwise step is a
single vectorized pass.

A metric may also depend on the run as a whole -- the unitary metric brings every
genome to the same set of qubits, and the behaviour metric feeds every genome the
same data -- which it declares through :meth:`DistanceMetric.context`: a hashable
summary of the genomes and the run's recorded information, and cached features
are reused only while it is unchanged.

New metrics are added by subclassing :class:`DistanceMetric` and decorating the
class with :func:`register_distance`; the dashboard lists every registered metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import lru_cache
from typing import Any, ClassVar

import numpy as np
from loguru import logger

from src.analysis.search_space.unitary import (
    MAX_UNITARY_QUBITS,
    circuit_unitary,
    genome_qubits,
)

#: Every registered distance metric, by name.
DISTANCE_METRICS: dict[str, type[DistanceMetric]] = {}

#: The most complex values the unitary metric stacks in double precision (512 MiB);
#: larger runs are compared in single precision, halving the memory.
_DOUBLE_PRECISION_LIMIT = 2**25


class UnsupportedGenome(ValueError):
    """Raised when a metric cannot place a genome, such as a circuit too large to simulate."""


class DistanceMetric(ABC):
    """Measures how far apart genomes are.

    Attributes:
        name: The metric's identifier, as the API and dashboard name it.
        label: A short human-readable name.
        description: One or two sentences on what the distance measures.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str]

    def context(
        self, genomes: list[dict[str, Any]], run_info: dict[str, Any] | None = None
    ) -> Hashable:
        """Summarizes what the metric needs to know about the run as a whole.

        Features computed under one context are reused only while the context
        stays the same, so a metric whose features depend on more than their own
        genome must return everything they depend on here.

        Args:
            genomes: Every serialized genome being placed.
            run_info: What the run recorded in its archive (``run_info``), such
                as its command-line ``arguments``; ``None`` when not known.

        Returns:
            A hashable summary; ``None`` (the default) when features depend on
            their own genome only.
        """

        return None

    @abstractmethod
    def featurize(self, genome: dict[str, Any], context: Hashable) -> Any:
        """Computes the feature one genome is compared by.

        Args:
            genome: A serialized genome (``CircuitGenome.to_dict``).
            context: What :meth:`context` returned for the run.

        Returns:
            The genome's feature, in whatever form :meth:`pairwise` takes.

        Raises:
            UnsupportedGenome: If the genome cannot be placed by this metric.
        """

    @abstractmethod
    def pairwise(self, features: list[Any]) -> np.ndarray:
        """Measures the distance between every pair of features.

        Args:
            features: One feature per genome, from :meth:`featurize`.

        Returns:
            The symmetric ``n x n`` distance matrix, with a zero diagonal.
        """

    @classmethod
    def describe(cls) -> dict[str, str]:
        """Describes the metric for the dashboard's picker.

        Returns:
            Its ``name``, ``label`` and ``description``.
        """

        return {"name": cls.name, "label": cls.label, "description": cls.description}


def register_distance(metric: type[DistanceMetric]) -> type[DistanceMetric]:
    """Registers a distance metric under its name (usable as a class decorator).

    Args:
        metric: The metric class.

    Returns:
        The same class, so this can decorate it.

    Raises:
        ValueError: If another metric is already registered under the name.
    """

    if metric.name in DISTANCE_METRICS and DISTANCE_METRICS[metric.name] is not metric:
        raise ValueError(
            f"A distance metric named {metric.name!r} is already registered."
        )
    DISTANCE_METRICS[metric.name] = metric
    return metric


def get_distance(name: str) -> DistanceMetric:
    """Creates a registered distance metric.

    Args:
        name: The metric's name.

    Returns:
        A new instance of the metric.

    Raises:
        ValueError: If no metric is registered under the name.
    """

    if name not in DISTANCE_METRICS:
        raise ValueError(
            f"Unknown distance metric {name!r}; choose from {', '.join(sorted(DISTANCE_METRICS))}."
        )
    return DISTANCE_METRICS[name]()


@register_distance
class FubiniStudyDistance(DistanceMetric):
    """The phase-invariant angle between two circuits' unitaries.

    ``theta(U, V) = arccos(|Tr(U^dagger V)| / d)``, with ``d = 2**n``: zero for
    circuits implementing the same operation up to a global phase, and at most
    ``pi / 2``. It is a true metric (the geodesic distance between unitaries once
    the global phase is factored out) and maps one to one onto the average gate
    fidelity, ``F = (d cos^2(theta) + 1) / (d + 1)``.

    Every genome is brought to the union of the run's qubits, with the identity
    on the qubits it does not use, so genomes on different qubit sets are
    comparable. It measures the whole operation: differences on qubits that are
    never measured, or that the input encoding never reaches, still count.
    """

    name = "fubini_study"
    label = "Unitary (Fubini–Study)"
    description = (
        "Angle between the circuits' unitaries, ignoring global phase: 0 when two "
        "circuits implement the same operation, at most π/2."
    )

    def context(
        self, genomes: list[dict[str, Any]], run_info: dict[str, Any] | None = None
    ) -> Hashable:
        """Collects the qubits every genome is brought to.

        Args:
            genomes: Every serialized genome being placed.
            run_info: Unused.

        Returns:
            The sorted union of the genomes' qubits, as a tuple.
        """

        qubits: set[tuple[str, int]] = set()
        for genome in genomes:
            qubits.update(genome_qubits(genome))
        return tuple(sorted(qubits))

    def featurize(self, genome: dict[str, Any], context: Hashable) -> np.ndarray:
        """Computes a genome's unitary over the run's qubits.

        Args:
            genome: A serialized genome.
            context: The run's qubits, from :meth:`context`.

        Returns:
            The unitary, flattened.

        Raises:
            UnsupportedGenome: If the circuit is too large or cannot be simulated.
        """

        qubits = list(context) if context else None
        if qubits is not None and len(qubits) > MAX_UNITARY_QUBITS:
            raise UnsupportedGenome(
                f"the run's circuits span {len(qubits)} qubits; unitaries are only "
                f"computed for up to {MAX_UNITARY_QUBITS}"
            )
        try:
            return circuit_unitary(genome, qubits).reshape(-1)
        except (KeyError, ValueError, TypeError) as error:
            raise UnsupportedGenome(str(error)) from error

    def pairwise(self, features: list[np.ndarray]) -> np.ndarray:
        """Measures the angle between every pair of unitaries.

        ``|Tr(U^dagger V)|`` is the magnitude of the inner product of the
        flattened matrices, so all pairs come from one matrix product.

        Args:
            features: Each genome's flattened unitary, all the same size.

        Returns:
            The angles, in radians.
        """

        if not features:
            return np.zeros((0, 0))
        size = features[0].size
        dimension = int(round(np.sqrt(size)))
        dtype = (
            np.complex128
            if len(features) * size <= _DOUBLE_PRECISION_LIMIT
            else np.complex64
        )
        stacked = np.stack(features).astype(dtype, copy=False)
        overlaps = np.abs(stacked.conj() @ stacked.T) / dimension
        distances = np.arccos(np.clip(overlaps, 0.0, 1.0)).astype(float)
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0.0)
        return distances


@register_distance
class JaccardDistance(DistanceMetric):
    """The share of two genomes' enabled gates they do not have in common.

    ``1 - |A & B| / |A | B|`` over the innovation numbers of each genome's enabled
    gates: the structural (genotype) distance of NEAT-style searches. Innovation
    numbers are shared within a run, so this compares genomes of the same run
    only. Gate parameters are ignored.
    """

    name = "jaccard"
    label = "Structure (Jaccard)"
    description = (
        "Share of the two genomes' enabled gates (by innovation number) that only "
        "one of them has: 0 for the same gates, 1 for none in common."
    )

    def featurize(self, genome: dict[str, Any], context: Hashable) -> frozenset[int]:
        """Collects a genome's enabled gate innovation numbers.

        Args:
            genome: A serialized genome.
            context: Unused.

        Returns:
            The innovation numbers.
        """

        return frozenset(
            int(gate["innovation_number"])
            for gate in genome.get("gates") or []
            if gate.get("enabled", True) and gate.get("innovation_number") is not None
        )

    def pairwise(self, features: list[frozenset[int]]) -> np.ndarray:
        """Measures the Jaccard distance between every pair of gate sets.

        Args:
            features: Each genome's enabled innovation numbers.

        Returns:
            The distances, in ``[0, 1]``; two empty circuits are at distance 0.
        """

        count = len(features)
        innovations = sorted(set().union(*features)) if features else []
        column = {innovation: i for i, innovation in enumerate(innovations)}
        membership = np.zeros((count, len(innovations)), dtype=np.float64)
        for row, feature in enumerate(features):
            membership[row, [column[innovation] for innovation in feature]] = 1.0

        shared = membership @ membership.T
        sizes = membership.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - shared
        with np.errstate(invalid="ignore", divide="ignore"):
            distances = np.where(union > 0, 1.0 - shared / union, 0.0)
        np.fill_diagonal(distances, 0.0)
        return distances


@register_distance
class ReadoutOperatorDistance(DistanceMetric):
    """How differently two circuits act on what the model actually measures.

    Compares the circuits in the Heisenberg picture: for each observable the
    model reads out, ``O -> U^dagger O U``. With ``probs`` readout the
    observables are the projectors onto each computational-basis outcome of the
    output qubits; with ``expval`` readout they are Pauli-Z on each output qubit.
    The distance is the Frobenius distance between the two circuits' stacks of
    evolved observables, scaled into ``[0, 1]``.

    Unlike the full-unitary distance it ignores everything the readout cannot
    see -- phases applied after the circuit, and anything on qubits that are not
    measured -- so it saturates much less, and it needs no data.
    """

    name = "readout"
    label = "Readout operators"
    description = (
        "Distance between the circuits' measured observables (U†OU for each readout "
        "outcome): ignores phases and unmeasured qubits; needs no data. 0 to 1."
    )

    def context(
        self, genomes: list[dict[str, Any]], run_info: dict[str, Any] | None = None
    ) -> Hashable:
        """Collects the run's qubits, measured qubits and readout mode.

        Args:
            genomes: Every serialized genome being placed.
            run_info: Unused.

        Returns:
            ``(qubits, output_qubits, output_mode)``: the sorted union of the
            genomes' qubits and of their output qubits, and the most common
            ``quantum_output_mode`` (``probs`` when none is recorded).
        """

        qubits: set[tuple[str, int]] = set()
        outputs: set[tuple[str, int]] = set()
        modes: dict[str, int] = {}
        for genome in genomes:
            qubits.update(genome_qubits(genome))
            outputs.update(tuple(qubit) for qubit in genome.get("output_qubits") or [])
            mode = (genome.get("hyperparameters") or {}).get("quantum_output_mode")
            if mode:
                modes[mode] = modes.get(mode, 0) + 1
        mode = max(modes, key=modes.get) if modes else "probs"
        return (tuple(sorted(qubits)), tuple(sorted(outputs)), mode)

    def featurize(self, genome: dict[str, Any], context: Hashable) -> np.ndarray:
        """Evolves each readout observable by the genome's circuit.

        Args:
            genome: A serialized genome.
            context: ``(qubits, output_qubits, output_mode)`` from :meth:`context`.

        Returns:
            The evolved observables, flattened and scaled so that the distance
            between any two genomes is at most 1.

        Raises:
            UnsupportedGenome: If the circuit is too large or cannot be simulated,
                or the run measures no qubits.
        """

        qubits, outputs, mode = context
        if not outputs:
            raise UnsupportedGenome("the run records no output qubits")
        if len(qubits) > MAX_UNITARY_QUBITS:
            raise UnsupportedGenome(
                f"the run's circuits span {len(qubits)} qubits; unitaries are only "
                f"computed for up to {MAX_UNITARY_QUBITS}"
            )
        try:
            unitary = circuit_unitary(genome, list(qubits))
        except (KeyError, ValueError, TypeError) as error:
            raise UnsupportedGenome(str(error)) from error

        n_qubits = len(qubits)
        dimension = 2**n_qubits
        states = np.arange(dimension)
        # the value of each measured qubit in every basis state (qubit 0 is the most significant bit)
        bits = [
            (states >> (n_qubits - 1 - qubits.index(qubit))) & 1 for qubit in outputs
        ]

        if mode == "expval":
            observables = [
                (unitary.conj().T * (1 - 2 * bit)[None, :]) @ unitary for bit in bits
            ]
            # two observables with eigenvalues +-1 are at most 2 sqrt(d) apart
            scale = np.sqrt(4 * dimension * len(bits))
        else:
            outcome = np.zeros(dimension, dtype=int)
            for bit in bits:
                outcome = (outcome << 1) | bit
            observables = []
            for value in range(2 ** len(bits)):
                rows = unitary[outcome == value]
                observables.append(rows.conj().T @ rows)
            # the projectors' squared distances add up to at most 2d
            scale = np.sqrt(2 * dimension)
        return (np.stack(observables).reshape(-1) / scale).astype(np.complex64)

    def pairwise(self, features: list[np.ndarray]) -> np.ndarray:
        """Measures the Frobenius distance between every pair of observable stacks.

        Args:
            features: Each genome's scaled, flattened observables.

        Returns:
            The distances, in ``[0, 1]``.
        """

        if not features:
            return np.zeros((0, 0))
        stacked = np.stack(features)
        gram = np.real(stacked @ stacked.conj().T).astype(float)
        norms = np.diag(gram)
        squared = norms[:, None] + norms[None, :] - 2 * gram
        distances = np.sqrt(np.clip(squared, 0.0, None))
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0.0)
        return distances


@register_distance
class BehaviourDistance(DistanceMetric):
    """How differently two classifiers label the run's own data.

    Every genome's whole model -- encoder, circuit and decoder, with its saved
    (trained) weights -- is run on every sample of the run's dataset, and its
    outputs are turned into class probabilities with the softmax the training
    loss applies. The distance is the square root of the Jensen–Shannon
    divergence (base 2) between the two models' class distributions, averaged
    over the samples: 0 for models that classify every sample identically, at
    most 1. It is a true metric.

    It measures only what the task sees, so the many circuits that implement
    the same classifier collapse together. The samples are the dataset the run
    was evolved on, rebuilt from the run's recorded ``dataset``,
    ``normalization`` and ``seed`` (its training and validation splits
    together). Only classification runs on tabular datasets are supported.
    """

    name = "behaviour"
    label = "Behaviour (Jensen–Shannon)"
    description = (
        "Square root of the Jensen–Shannon divergence between the two models' class "
        "probabilities on the run's dataset, averaged over samples: 0 when they "
        "classify alike, at most 1."
    )

    def context(
        self, genomes: list[dict[str, Any]], run_info: dict[str, Any] | None = None
    ) -> Hashable:
        """Identifies the data every genome is run on.

        Args:
            genomes: Every serialized genome being placed.
            run_info: What the run recorded; its ``task`` and ``arguments``
                (``dataset``, ``normalization`` and ``seed``) choose the data.

        Returns:
            ``("classification", dataset, normalization, seed)``, or
            ``("unsupported", reason)`` for a run whose data cannot be rebuilt.
        """

        run_info = run_info or {}
        arguments = run_info.get("arguments") or {}
        first = genomes[0] if genomes else {}
        task = run_info.get("task") or first.get("task")
        dataset = (
            arguments.get("dataset")
            or run_info.get("task_target")
            or first.get("task_target")
        )
        if task != "classification":
            return (
                "unsupported",
                f"behaviour is only measured for classification runs, not {task!r}",
            )

        from src.datasets.classification_loaders import UCI_DATASETS

        if dataset not in UCI_DATASETS:
            return (
                "unsupported",
                f"behaviour is only measured on tabular datasets, not {dataset!r}",
            )
        return (
            "classification",
            dataset,
            arguments.get("normalization") or "minmax",
            int(arguments.get("seed") or 0),
        )

    def featurize(self, genome: dict[str, Any], context: Hashable) -> np.ndarray:
        """Runs a genome's model on the run's data.

        Args:
            genome: A serialized genome.
            context: The data, from :meth:`context`.

        Returns:
            The class probabilities, one row per sample.

        Raises:
            UnsupportedGenome: If the run's data cannot be rebuilt, or the
                genome's model cannot be built or run on it.
        """

        if context[0] != "classification":
            raise UnsupportedGenome(context[1])
        _, dataset, normalization, seed = context

        import torch

        from src.circuits.circuit import CircuitGenome

        try:
            inputs = _classification_samples(dataset, normalization, seed)
        except (OSError, ValueError) as error:
            raise UnsupportedGenome(f"could not load {dataset!r}: {error}") from error

        # building a model logs its qubits and gates, once per genome of the run
        logger.disable("src.circuits")
        try:
            model = CircuitGenome.from_dict(genome)
            model.initialize_model()
            model.hybrid_model.eval()
            with torch.no_grad():
                outputs = model.forward(inputs)
        except (
            Exception
        ) as error:  # noqa: BLE001 -- any failure only leaves this genome out
            raise UnsupportedGenome(f"its model could not be run: {error}") from error
        finally:
            logger.enable("src.circuits")
        return torch.softmax(outputs.double(), dim=-1).numpy()

    def pairwise(self, features: list[np.ndarray]) -> np.ndarray:
        """Measures the mean Jensen–Shannon distance between every pair of models.

        Args:
            features: Each genome's class probabilities, all over the same samples.

        Returns:
            The distances, in ``[0, 1]``.
        """

        count = len(features)
        distances = np.zeros((count, count))
        if count < 2:
            return distances
        probabilities = np.stack(features)

        def entropy_terms(p: np.ndarray, m: np.ndarray) -> np.ndarray:
            """Sums ``p log2(p / m)`` over classes, taking ``0 log 0`` as 0."""
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1) / m), 0.0)
            return terms.sum(axis=-1)

        for i in range(count - 1):
            others = probabilities[i + 1 :]
            mixture = (probabilities[i][None] + others) / 2
            divergence = (
                entropy_terms(probabilities[i][None], mixture)
                + entropy_terms(others, mixture)
            ) / 2
            row = np.sqrt(np.clip(divergence, 0.0, 1.0)).mean(axis=1)
            distances[i, i + 1 :] = row
            distances[i + 1 :, i] = row
        return distances


@lru_cache(maxsize=4)
def _classification_samples(dataset: str, normalization: str, seed: int) -> Any:
    """Rebuilds a classification run's samples, as its training loaded them.

    Args:
        dataset: The tabular dataset's name.
        normalization: The run's ``--normalization``.
        seed: The run's ``--seed``, which chose its training/validation split.

    Returns:
        A float tensor of every sample: the training split, then the validation
        split, each in the order the run's loaders hold them.
    """

    import torch

    from src.datasets.classification_loaders import get_uci_dataloaders

    training, validation = get_uci_dataloaders(
        dataset, normalize=normalization, seed=seed
    )
    return torch.cat([training.dataset.x, validation.dataset.x])
