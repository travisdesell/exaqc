"""How EXAQC chooses the training hyperparameters of each child it generates.

Every genome carries its own ``hyperparameters`` dict, which the trainers read
at train time. :class:`EXAQC` asks a :class:`HyperparameterStrategy` for that
dict whenever it generates a child:

- :class:`FixedHyperparameters` (the default) gives every child a copy of the
  search's configured hyperparameters, so they never change during a run.
- :class:`SimplexHyperparameters` co-evolves a chosen subset of them with
  simplex hyperparameter optimization (SHO; Kini et al., "Co-evolving Recurrent
  Neural Networks and their Hyperparameters with Simplex Hyperparameter
  Optimization", GECCO '23 Companion). While the population is still being
  filled each tuned value is drawn uniformly from a narrow initial range; after
  that, each child's values are a random point on the line from the average of
  several randomly chosen genomes' values towards the best of them.
"""

from __future__ import annotations

import argparse
import math
import random

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.circuits.circuit import CircuitGenome

#: The ``--hyperparameter_strategy`` choices: ``fixed`` gives every child the
#: configured hyperparameters, ``simplex`` co-evolves the ``--sho_tune`` ones.
HYPERPARAMETER_STRATEGIES: tuple[str, ...] = ("fixed", "simplex")

#: How a tuned hyperparameter is stepped: ``linear`` on its value, ``log`` on
#: its base-10 logarithm (for values spanning orders of magnitude, such as a
#: learning rate), and ``int`` linearly and then rounded to an integer.
HYPERPARAMETER_SCALES: tuple[str, ...] = ("linear", "log", "int")


@dataclass(frozen=True)
class TunedHyperparameter:
    """One hyperparameter SHO tunes, and the ranges it may take.

    Attributes:
        name: The key in a genome's ``hyperparameters`` dict.
        scale: One of :data:`HYPERPARAMETER_SCALES`.
        initial_min: Lower bound of the burn-in range.
        initial_max: Upper bound of the burn-in range.
        min: Lower bound any generated value is clamped to.
        max: Upper bound any generated value is clamped to.
    """

    name: str
    scale: str
    initial_min: float
    initial_max: float
    min: float
    max: float

    @classmethod
    def parse(cls, spec: str) -> TunedHyperparameter:
        """Parses a ``name=scale:initial_min:initial_max[:min:max]`` specification.

        When ``min`` and ``max`` are left out, generated values are clamped to
        the burn-in range.

        Args:
            spec: The specification, e.g. ``learning_rate=log:1e-3:5e-2:1e-5:0.3``.

        Returns:
            The parsed hyperparameter.

        Raises:
            ValueError: If the specification is malformed, names an unknown
                scale, has non-numeric bounds, has a burn-in range outside the
                full range, or has non-positive bounds on a ``log`` scale.
        """

        name, separator, rest = spec.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(
                f"{spec!r} is not of the form name=scale:initial_min:initial_max[:min:max]."
            )

        fields = rest.split(":")
        if len(fields) not in (3, 5):
            raise ValueError(
                f"{spec!r} needs 3 or 5 ':'-separated fields after '=', got {len(fields)}."
            )

        scale = fields[0].strip()
        if scale not in HYPERPARAMETER_SCALES:
            raise ValueError(
                f"{spec!r} has unknown scale {scale!r}; choose one of {HYPERPARAMETER_SCALES}."
            )

        try:
            bounds = [float(field) for field in fields[1:]]
        except ValueError:
            raise ValueError(f"{spec!r} has a bound that is not a number.") from None

        initial_min, initial_max = bounds[0], bounds[1]
        full_min, full_max = (bounds[2], bounds[3]) if len(bounds) == 4 else bounds

        if not full_min <= initial_min <= initial_max <= full_max:
            raise ValueError(
                f"{spec!r} must satisfy min <= initial_min <= initial_max <= max."
            )

        if scale == "log" and full_min <= 0.0:
            raise ValueError(f"{spec!r} uses a log scale, so its bounds must be > 0.")

        return cls(
            name=name,
            scale=scale,
            initial_min=initial_min,
            initial_max=initial_max,
            min=full_min,
            max=full_max,
        )

    def to_search_space(self, value: float) -> float:
        """Maps a hyperparameter value into the space SHO steps in.

        Args:
            value: The hyperparameter's value; clamped to ``[min, max]`` first.

        Returns:
            ``log10(value)`` on a ``log`` scale, otherwise the clamped value.
        """

        value = min(max(float(value), self.min), self.max)
        return math.log10(value) if self.scale == "log" else value

    def from_search_space(self, position: float) -> float | int:
        """Maps a point of the space SHO steps in back to a hyperparameter value.

        Args:
            position: The point in search space.

        Returns:
            The value, clamped to ``[min, max]``: ``10 ** position`` on a ``log``
            scale, rounded to an ``int`` on an ``int`` scale.
        """

        value = 10.0**position if self.scale == "log" else position
        value = min(max(value, self.min), self.max)
        return int(round(value)) if self.scale == "int" else value

    def sample_initial(self, rng: random.Random) -> float | int:
        """Draws a burn-in value uniformly from the initial range.

        On a ``log`` scale the draw is uniform in the logarithm.

        Args:
            rng: The random number generator to draw with.

        Returns:
            The drawn value.
        """

        low = self.to_search_space(self.initial_min)
        high = self.to_search_space(self.initial_max)
        return self.from_search_space(rng.uniform(low, high))

    def as_dict(self) -> dict[str, Any]:
        """Describes this hyperparameter for the archive's ``run_info``.

        Returns:
            The name, scale and ranges, as JSON-serializable values.
        """

        return {
            "name": self.name,
            "scale": self.scale,
            "initial_range": [self.initial_min, self.initial_max],
            "range": [self.min, self.max],
        }


class HyperparameterStrategy(ABC):
    """Chooses the training hyperparameters of each child EXAQC generates.

    Attributes:
        min_population: How many genomes the strategy needs to learn from. When
            the child's own sub-population (e.g. a repopulating island) holds
            fewer, :class:`~src.evolution.exaqc.EXAQC` passes the whole
            population instead. A strategy with ``0`` does not learn from the
            population and is passed an empty list.
    """

    min_population: int = 0

    def tuned_names(self) -> list[str]:
        """Names the hyperparameters this strategy chooses per child.

        Returns:
            The tuned names; none by default, where every child gets the
            configured values.
        """

        return []

    @staticmethod
    def initialize_parser(
        parser: argparse.ArgumentParser,
        tunable: tuple[str, ...],
        default_tune: list[str],
    ) -> None:
        """Adds the hyperparameter-strategy command-line arguments to a parser.

        Args:
            parser: The parser to add the arguments to.
            tunable: The hyperparameter names ``--sho_tune`` may name, i.e. the
                ones the entry point's trainer reads per genome.
            default_tune: The ``--sho_tune`` specifications used when it is not
                given.

        Returns:
            None. Mutates ``parser`` by adding ``--hyperparameter_strategy`` and
            the SHO arguments (see :meth:`SimplexHyperparameters.initialize_parser`).
        """

        parser.add_argument(
            "--hyperparameter_strategy",
            choices=HYPERPARAMETER_STRATEGIES,
            default="fixed",
            help=(
                "How each genome's training hyperparameters are chosen: 'fixed' uses the "
                "configured values for every genome, 'simplex' co-evolves the --sho_tune "
                "ones with simplex hyperparameter optimization (SHO)."
            ),
        )
        SimplexHyperparameters.initialize_parser(parser, tunable, default_tune)

    @staticmethod
    def from_args(args: argparse.Namespace) -> HyperparameterStrategy:
        """Builds the strategy the arguments from :meth:`initialize_parser` choose.

        Args:
            args: The parsed arguments. A run recorded before these arguments
                existed has no ``hyperparameter_strategy``, and is fixed.

        Returns:
            A :class:`FixedHyperparameters` or :class:`SimplexHyperparameters`.

        Raises:
            ValueError: If the SHO arguments are invalid (see
                :meth:`SimplexHyperparameters.from_args`).
        """

        if getattr(args, "hyperparameter_strategy", "fixed") == "simplex":
            return SimplexHyperparameters.from_args(args)
        return FixedHyperparameters()

    @abstractmethod
    def generate(
        self,
        base: dict[str, Any],
        population: list[CircuitGenome],
        initializing: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns the hyperparameters for a new child.

        Args:
            base: The search's configured hyperparameters. Keys the strategy does
                not tune are copied from here unchanged.
            population: The evaluated genomes the child is generated among, best
                first (see
                :meth:`~src.evolution.population_strategy.PopulationStrategy.get_population_for_child`).
            initializing: Whether the population is still being filled.
            metadata: The child's metadata, which the strategy may record how
                the hyperparameters were chosen in, or ``None`` when there is no
                child (the search's seed genome).

        Returns:
            A new hyperparameters dict.
        """
        pass

    def run_info(self) -> dict[str, Any]:
        """Describes the strategy for the archive's ``run_info``.

        Returns:
            ``run_info`` entries to record, or an empty dict to record nothing.
        """

        return {}


class FixedHyperparameters(HyperparameterStrategy):
    """Gives every child a copy of the search's configured hyperparameters."""

    def generate(
        self,
        base: dict[str, Any],
        population: list[CircuitGenome],
        initializing: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns a copy of the configured hyperparameters.

        Args:
            base: The search's configured hyperparameters.
            population: Unused.
            initializing: Unused.
            metadata: Unused.

        Returns:
            A copy of ``base``.
        """

        return base.copy()


class SimplexHyperparameters(HyperparameterStrategy):
    """Co-evolves hyperparameters with simplex hyperparameter optimization (SHO).

    While the population is initializing (the burn-in), each tuned value is
    drawn uniformly from its initial range. Afterwards ``n_genomes`` distinct
    genomes are chosen at random from the child's population, independently of
    how its parents were chosen. With ``h_best`` the best one's values and
    ``h_avg`` the mean of the others', every tuned value is set to
    ``h_avg + r * (h_best - h_avg)`` for one ``r = U(0, 1) * l1 - l2`` shared by
    all of them, clamped to its full range. Steps are taken in each value's
    search space (see :meth:`TunedHyperparameter.to_search_space`).
    """

    @staticmethod
    def initialize_parser(
        parser: argparse.ArgumentParser,
        tunable: tuple[str, ...],
        default_tune: list[str],
    ) -> None:
        """Adds the SHO command-line arguments to a parser.

        They only take effect with ``--hyperparameter_strategy simplex`` (see
        :meth:`HyperparameterStrategy.initialize_parser`, which calls this).

        Args:
            parser: The parser to add the arguments to.
            tunable: The hyperparameter names ``--sho_tune`` may name.
            default_tune: The ``--sho_tune`` specifications used when it is not
                given.

        Returns:
            None. Mutates ``parser`` by adding ``--sho_tune``, ``--sho_genomes``,
            ``--sho_l1`` and ``--sho_l2``.
        """

        def tune_spec(spec: str) -> str:
            """Validates a ``--sho_tune`` specification for argparse.

            The string itself is kept (rather than the parsed object) so the
            arguments stay JSON-serializable for the archive and ``--restart``.

            Args:
                spec: The specification given on the command line.

            Returns:
                ``spec`` unchanged.

            Raises:
                argparse.ArgumentTypeError: If the specification is malformed or
                    names a hyperparameter that cannot be tuned.
            """

            try:
                parsed = TunedHyperparameter.parse(spec)
            except ValueError as error:
                raise argparse.ArgumentTypeError(str(error)) from None
            if parsed.name not in tunable:
                raise argparse.ArgumentTypeError(
                    f"{parsed.name!r} cannot be tuned; choose from {list(tunable)}."
                )
            return spec

        def genome_count(value: str) -> int:
            """Validates a ``--sho_genomes`` value for argparse.

            Args:
                value: The value given on the command line.

            Returns:
                The value as an ``int``.

            Raises:
                argparse.ArgumentTypeError: If it is not an integer of at least 2.
            """

            try:
                count = int(value)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"{value!r} is not an integer."
                ) from None
            if count < 2:
                raise argparse.ArgumentTypeError(
                    f"SHO needs at least 2 genomes per step, got {count}."
                )
            return count

        parser.add_argument(
            "--sho_tune",
            type=tune_spec,
            nargs="+",
            default=list(default_tune),
            metavar="NAME=SCALE:INITIAL_MIN:INITIAL_MAX[:MIN:MAX]",
            help=(
                "Hyperparameters to co-evolve with simplex hyperparameter optimization. "
                f"SCALE is one of {', '.join(HYPERPARAMETER_SCALES)}; values start in "
                "[INITIAL_MIN, INITIAL_MAX] and are clamped to [MIN, MAX] (the initial range "
                f"when omitted). Tunable: {', '.join(tunable)}."
            ),
        )

        parser.add_argument(
            "--sho_genomes",
            type=genome_count,
            default=4,
            help=(
                "How many genomes SHO picks at random to step from: the best of them "
                "against the average of the rest (at least 2)."
            ),
        )

        parser.add_argument(
            "--sho_l1",
            type=float,
            default=2.0,
            help="SHO step width: r = U(0, 1) * l1 - l2.",
        )

        parser.add_argument(
            "--sho_l2",
            type=float,
            default=0.5,
            help="SHO step offset: r = U(0, 1) * l1 - l2.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SimplexHyperparameters:
        """Builds the strategy from the arguments added by :meth:`initialize_parser`.

        Args:
            args: The parsed arguments.

        Returns:
            The configured strategy.

        Raises:
            ValueError: If the arguments do not describe a valid strategy (see
                :meth:`__init__`).
        """

        return cls(
            tuned=[TunedHyperparameter.parse(spec) for spec in args.sho_tune],
            n_genomes=args.sho_genomes,
            l1=args.sho_l1,
            l2=args.sho_l2,
        )

    def __init__(
        self,
        tuned: list[TunedHyperparameter],
        n_genomes: int = 4,
        l1: float = 2.0,
        l2: float = 0.5,
        rng: random.Random | None = None,
    ) -> None:
        """Creates the strategy.

        Args:
            tuned: The hyperparameters to co-evolve.
            n_genomes: How many genomes each step is taken from.
            l1: Step width, as in ``r = U(0, 1) * l1 - l2``.
            l2: Step offset, as in ``r = U(0, 1) * l1 - l2``.
            rng: The random number generator to draw with; the ``random``
                module's shared generator (seeded with the run) when ``None``.

        Raises:
            ValueError: If ``n_genomes`` is less than 2 or a hyperparameter is
                tuned twice.
        """

        if n_genomes < 2:
            raise ValueError(f"SHO needs at least 2 genomes per step, got {n_genomes}.")

        names = [hyperparameter.name for hyperparameter in tuned]
        if len(set(names)) != len(names):
            raise ValueError(f"a hyperparameter is tuned more than once: {names}.")

        self.tuned = list(tuned)
        self.n_genomes = n_genomes
        self.min_population = n_genomes
        self.l1 = l1
        self.l2 = l2
        self.rng: random.Random | Any = rng if rng is not None else random

    def tuned_names(self) -> list[str]:
        """Names the hyperparameters SHO co-evolves.

        Returns:
            The ``--sho_tune`` names, in the order given.
        """

        return [hyperparameter.name for hyperparameter in self.tuned]

    def run_info(self) -> dict[str, Any]:
        """Describes the strategy for the archive's ``run_info``.

        Returns:
            A ``hyperparameter_strategy`` entry holding the tuned
            hyperparameters and the step settings.
        """

        return {
            "hyperparameter_strategy": {
                "name": "simplex",
                "tuned": [hyperparameter.as_dict() for hyperparameter in self.tuned],
                "n_genomes": self.n_genomes,
                "l1": self.l1,
                "l2": self.l2,
            }
        }

    def generate(
        self,
        base: dict[str, Any],
        population: list[CircuitGenome],
        initializing: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns the hyperparameters for a new child, burn-in or SHO step.

        A burn-in draw is also used after initialization if fewer than
        ``n_genomes`` genomes are available to step from.

        Args:
            base: The search's configured hyperparameters; untuned keys are
                copied from here, as are tuned values a genome does not carry.
            population: The evaluated genomes the child is generated among, best
                first.
            initializing: Whether the population is still being filled.
            metadata: The child's metadata, or ``None``. When given, how the
                values were chosen is recorded in its
                ``hyperparameter_generation`` entry.

        Returns:
            A new hyperparameters dict.
        """

        hyperparameters = base.copy()

        if initializing or len(population) < self.n_genomes:
            for hyperparameter in self.tuned:
                hyperparameters[hyperparameter.name] = hyperparameter.sample_initial(
                    self.rng
                )
            record: dict[str, Any] = {"strategy": "simplex", "phase": "burn_in"}

        else:
            # the population is sorted best first, so the lowest chosen index is
            # the best of the chosen genomes
            chosen = sorted(self.rng.sample(range(len(population)), self.n_genomes))
            best = population[chosen[0]]
            others = [population[index] for index in chosen[1:]]
            r = self.rng.random() * self.l1 - self.l2

            for hyperparameter in self.tuned:

                def position(genome: CircuitGenome) -> float:
                    """Where a genome's value of this hyperparameter lies in search space.

                    Args:
                        genome: The genome whose value is read; the base value
                            stands in when it carries none.

                    Returns:
                        The value, mapped into search space.
                    """

                    value = (genome.hyperparameters or {}).get(
                        hyperparameter.name, base.get(hyperparameter.name)
                    )
                    return hyperparameter.to_search_space(value)

                best_position = position(best)
                average_position = sum(position(genome) for genome in others) / len(
                    others
                )
                hyperparameters[hyperparameter.name] = hyperparameter.from_search_space(
                    average_position + r * (best_position - average_position)
                )

            record = {
                "strategy": "simplex",
                "phase": "simplex",
                "best_genome": best.genome_number,
                "other_genomes": [genome.genome_number for genome in others],
                "r": r,
            }

        if metadata is not None:
            metadata["hyperparameter_generation"] = record

        return hyperparameters
