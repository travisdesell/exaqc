from src.circuits.gate_specifications import all_gate_specifications


class EXAQC:

    def __init__(
        self,
        gate_specifications: GateSpecifications = all_gate_specifications,
        population: PopulationStrategy = None,
    ):
        """
        Creates an instance of Evolutionary Exploration of Augmenting Quantum Circuits given a
        particular population strategy, allowing the given gates (if specified).

        args:
            gate_specifications: is an object containing the allowed gates specifications for the search
                process. If it is none, it will default to the globally defined specifications for all
                gates.
            population: is an instance of a subclass of the PopulationStrategy interface, utilized to get
                parents for mutation or crossover and insert children back into the population.
        """

        return
