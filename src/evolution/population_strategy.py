from abc import ABC, abstractmethod

from src.circuits.circuit import CircuitGenome


class PopulationStrategy(ABC):

    @abstractmethod
    def get_parent(self, **kwargs) -> CircuitGenome:
        """
        Used to get a single to be used in mutation or
        other operations to generate children.

        Args:
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A single CircuitGenome from the population or None if the population
            is empty or it is not possible to get a parent.
        """
        pass

    @abstractmethod
    def get_parents(self, n_parents: int = 2, **kwargs) -> list[CircuitGenome]:
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
            is less than n_parents, it will return None.
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
