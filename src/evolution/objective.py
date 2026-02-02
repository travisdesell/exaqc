from abc import ABC

from src.circuits.circuit import CircuitGenome

class Objective(ABC):

    @abstractmethod
    def compare(self, genome1: CircuitGenome, genome2: CircuitGenome) -> int:
        """
        Used to sort genomes by fitness, even if there are multiple objectives, for population 
        management and crossover methods.

        Returns: 0 if the two genomes have equivalent fitnesses, a ngeative value if genome1 should be
            sorted before genome2, and a positive value if genome2 should be sorted before genome1
        """
        @pass


    @abstractmethod
    def __call__(self, genome: CircuitGenome):
        """
        Uses this objective function to evaluatate the provided genome. When completed this method
        should set the `fitness` attribute of the genome to a dictionary with key value pairs
        where the key is the name of the loss, and the value is the loss value.  This allows genomes
        to have multiple loss functions for multiple objectives.
        """
        pass
