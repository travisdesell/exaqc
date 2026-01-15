from src.circuits.gate_specifications import GateSpecifications


class EXAQC:

    def __init__(
        self,
        gate_specifications: GateSpecifications,
        population: PopulationStrategy,
    ):
        """
        Creates an instance of Evolutionary Exploration of Augmenting Quantum Circuits given a
        particular population strategy, allowing the given gates (if specified).

        args:
            gate_specifications: is an object containing the allowed gates specifications for the search
                process, for either the pennylane or qiskit frameworks.
            population: is an instance of a subclass of the PopulationStrategy interface, utilized to get
                parents for mutation or crossover and insert children back into the population.
        """

        return
