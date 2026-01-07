from qiskit import QuantumCircuit

from src.circuits.gate_specifications import gate_specifications



class Gate:
    """
    Represents a gate within a quantum circuit, which could have multiple inputs
    and outputs, as well as potential paramaeters.  Contains appropriate information 
    to go from this representation to adding the correct qiskit method to an existing
    circuit.
    """

    def __init__(self, innovation_number: int, depth: float, method_name: str, qubits: list[tuple[str, int]], parameters: dict[str, float] = {}):
        """
        Initializes a gate element in an evolved quantum circuit.

        Args:
            innovation_number: is a unique number representing this gate across every evolved
                quantum circuit it appears in
            depth: is how deep in the circuit this gate appears, this will be a value
                greater than 0 and less than 1, as 0 represents the inputs and 1
                represents the outputs.
            method_name: is the method name used by qiskit to add the gate to a quantum circuit
            qubits: a list of qubits to form the arguments to the gate method name, each is a tuple with a string for the input
                register name and then the index. if the gate takes the whole register (and not individual qubits) then the second
                value will be None (e.g., for a Hadamard gate).
            parameters: is a dict where the keys are the parameter names and the values are the values
        """

        assert (depth > 0.0) and (depth < 1.0)
        self.depth = depth

        self.innovation_number = innovation_number

        self.method_name = method_name
        self.qubits = qubits
        self.parameters = parameters

        self.specs = gate_specifications[self.method_name]

        # the number of parameters and number of qubits provided need to be the
        # same as in the specifications
        assert len(self.qubits) == len(self.specs['qubits'])

        if 'parameters' not in self.specs:
            assert len(self.parameters) == 0
        else:
            assert len(self.parameters) == len(self.specs['parameters'])

        self.enabled = True

    def add_to_circuit(self, registers: dict[str, QuantumRegister], circuit: QuantumCircuit):
        """
        Adds this gate to the qiskit QuantumCircuit using reflection
        to specify the correct method along with the given parameters.

        Args:
            circuit: is the qiskit QuantumCircuit to add this gate to
        """

        print(f"adding gate {self.method_name}({self.qubits}, {self.parameters}")

        gate_method = getattr(circuit, self.method_name)

        qubit_args = {}

        for i, qubit in enumerate(self.qubits):
            qubit_name = qubit[0]
            qubit_index = qubit[1]
            argument_name = self.specs['qubits'][i]
            print(f"\tsetting argument '{argument_name}' = '{qubit_name}[{qubit_index}]'")

            # assign the values for the qubit arguments to the method
            # name
            qubit_args[argument_name] = registers[qubit_name][qubit_index]

        gate_method(**self.parameters, **qubit_args)


