from src.circuits.circuit import CircuitGenome


def validate_innovation_numbers(circuit: CircuitGenome):
    """
    Performs a check to validate that all gate innovation numbers in the
    genome are unique.
    """

    seen = set()

    print(f"number gates: {len(circuit.gates)}")
    for gate in circuit.gates:
        print(f"\t{gate}")

    for gate in circuit.gates:
        print(f"\tgate innovation number: {gate.innovation_number}")
        assert gate.innovation_number not in seen
        seen.add(gate.innovation_number)
