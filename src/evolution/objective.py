from abc import ABC, abstractmethod

from src.circuits.circuit import CircuitGenome


class Objective(ABC):

    @abstractmethod
    def __call__(self, genome: CircuitGenome):
        """
        Uses this objective function to evaluatate the provided genome. When completed this method
        should set the `fitness` attribute of the genome to a dictionary with key value pairs
        where the key is the name of the loss, and the value is the loss value.  This allows genomes
        to have multiple loss functions for multiple objectives.
        """
        pass
