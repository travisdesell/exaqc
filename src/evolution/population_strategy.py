import argparse
import os

from abc import ABC, abstractmethod
from collections.abc import Callable

from src.circuits.circuit import CircuitGenome


class PopulationStrategy(ABC):

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds the population-strategy sub-commands to a parser.

        Every entry point offers the same choice of how genomes are managed, as
        a required sub-command that in turn owns its own flags. This registers
        the ``steady_state`` and ``islands`` sub-parsers and hands each to the
        concrete strategy's own ``initialize_parser`` (so each class defines its
        own constructor flags), keeping the entry points in sync.

        The concrete strategies are imported lazily because they subclass
        :class:`PopulationStrategy`; importing them at module load time would be
        circular.

        Args:
            parser: The parser to add the ``population_strategy`` sub-commands
                to.

        Returns:
            None. Mutates ``parser`` by adding a required ``population_strategy``
            sub-command with ``steady_state`` and ``islands`` choices, each
            carrying that strategy's own arguments.
        """

        from src.evolution.steady_state_islands import SteadyStateIslands
        from src.evolution.steady_state_population import SteadyStatePopulation

        populations = parser.add_subparsers(
            dest="population_strategy",
            required=True,
            help="Specify how genomes will be handled.",
        )
        # Each population strategy owns the flags for its own constructor.
        SteadyStatePopulation.initialize_parser(
            populations.add_parser(
                "steady_state", help="Use a single steady state population."
            )
        )
        SteadyStateIslands.initialize_parser(
            populations.add_parser(
                "islands", help="Use multiple islands of steady state populations."
            )
        )

    @staticmethod
    def from_args(
        args: argparse.Namespace,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
    ) -> "PopulationStrategy":
        """Builds the selected population strategy from parsed arguments.

        Constructs a :class:`~src.evolution.steady_state_population.SteadyStatePopulation`
        or :class:`~src.evolution.steady_state_islands.SteadyStateIslands` from
        the sub-command chosen by :meth:`initialize_parser` and its flags. The
        output directory the strategy writes genomes into is created here (the
        strategy is the component that owns ``--out_dir``).

        Args:
            args: Parsed arguments carrying ``population_strategy`` and the
                selected strategy's flags, along with ``--out_dir`` and
                ``--save_training_plot``.
            compare: Genome comparison used to order the population (task
                specific; each entry point defines its own).

        Returns:
            The constructed :class:`PopulationStrategy`.
        """

        # Imported lazily: the concrete strategies subclass this class, so a
        # module-level import here would be circular.
        from src.evolution.steady_state_islands import SteadyStateIslands
        from src.evolution.steady_state_population import SteadyStatePopulation

        # The strategy owns the output directory, so create it here.
        os.makedirs(args.out_dir, exist_ok=True)

        if args.population_strategy == "steady_state":
            return SteadyStatePopulation(
                max_population_size=args.max_population_size,
                compare=compare,
                out_dir=args.out_dir,
                save_training_plot=args.save_training_plot,
            )

        return SteadyStateIslands(
            n_islands=args.n_islands,
            max_island_size=args.max_island_size,
            genomes_before_extinction=args.genomes_before_extinction,
            genomes_for_next_extinction=args.genomes_for_next_extinction,
            islands_to_extinct=args.islands_to_extinct,
            primary_parent=args.primary_parent,
            intra_island_crossover_rate=args.intra_island_crossover_rate,
            compare=compare,
            topology=args.topology,
            out_dir=args.out_dir,
            save_training_plot=args.save_training_plot,
        )

    @abstractmethod
    def is_initializing(self) -> bool:
        """
        Used to determine if the strategy is still initializing so EXAQC
        can continue to generate genomes from the seed genome.

        Returns:
            True if the population strategy is still initializing (i.e., its
            population or populations are not all full).
        """
        pass

    @abstractmethod
    def get_best_genome(self) -> CircuitGenome:
        """
        Returns:
            The best genome in the strategy. Will return none if no genomes
            have been inserted yet (i.e., the very beginning of the search).
        """
        pass

    @abstractmethod
    def get_parent(self, **kwargs) -> tuple[CircuitGenome, dict[str, any]]:
        """
        Used to get a single to be used in mutation or
        other operations to generate children.

        Args:
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A single CircuitGenome from the population or None if the population
            is empty or it is not possible to get a parent. The second return
            value is a dictionary of metadata (which can be empty) for the child
            to be generated from these parents.
        """
        pass

    @abstractmethod
    def get_parents(
        self, n_parents: int = 2, **kwargs
    ) -> tuple[list[CircuitGenome], dict[str, any]]:
        """
        Used to get two or more parents to be used in crossover or
        other operations to generate children.

        Args:
            n_parents: specifies how many parents to return by the method.
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A list of unique (non-duplicate) CircuitGenomes. If the size of the population
            is less than n_parents, it will return None. The second return
            value is a dictionary of metadata (which can be empty) for the child
            to be generated from these parents.
        """
        pass

    @abstractmethod
    def insert_genome(self, genome: CircuitGenome, **kwargs) -> bool:
        """
        Inserts a genome back into the population.

        Args:
            genome: is the genome to insert into the population.
            **kwargs: is used to pass additional options to the method for
                inserting the genome, such as an island or species it came from.

        Returns:
            True if it was inserted into the population, False otherwise.
        """
        pass
