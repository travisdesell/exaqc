from abc import ABC, abstractmethod

from torch import Tensor


class Metric(ABC):
    """
    An abstract class to define how metrics can be generically calculated for varying batch
    sizes and dataloaders for both training and validation data.
    """

    @abstractmethod
    def reset(self):
        """
        Resets any values for the beginning of a new epoch.
        """
        pass

    @abstractmethod
    def accumulate(self, output: Tensor, target: Tensor):
        """
        Used to accumulate the metric given target and output tensors
        for individual passes or batches of results.

        Args:
            output: The produced output values from the circuit.
            target: The expected label values for each sample.
        """
        pass

    @abstractmethod
    def calculate(self) -> any:
        """
        At the end of an epoch, this is used to calculate the actual metric
        being evaluated.

        Returns:
            The metric or metrics calculated by this accumulator. Must be able to
            be written to JSON (e.g., a primitive, list or dict) for tracking and
            message passing of genome JSONs.
        """
        pass
