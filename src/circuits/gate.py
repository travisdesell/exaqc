from qiskit import QuantumCircuit, QuantumRegister
import pennylane as qml

from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.evolution.innovation import innovation_number_generator


class Gate:
    """
    Represents a gate within a quantum circuit, which could have multiple inputs
    and outputs, as well as potential paramaeters.  Contains appropriate information
    to go from this representation to adding the correct qiskit method to an existing
    circuit.
    """

    def __init__(
        self,
        depth: float,
        method_name: str,
        qubits: list[tuple[str, int]],
        parameters: dict[str, float] = {},
        innovation_number: int = None,
    ):
        """
        Initializes a gate element in an evolved quantum circuit.

        Args:
            depth: is how deep in the circuit this gate appears, this will be a value
                greater than 0 and less than 1, with 0 being closest to the input and 1
                being closest to the output.
            method_name: is the method name used by qiskit to add the gate to a quantum circuit
            qubits: a list of qubits to form the arguments to the gate method name, each is a tuple with a string
                for the input register name and then the index. if the gate takes the whole register (and not
                individual qubits) then the second value will be None (e.g., for a Hadamard gate).
            parameters: is a dict where the keys are the parameter names and the values are the values
            innovation_number: is a unique number representing this gate across every evolved
                quantum circuit it appears in. if a value is not provided, a new innovation number will be generated
                for the gate.
        """

        assert (depth > 0.0) and (depth < 1.0)
        self.depth = depth

        if innovation_number is None:
            self.innovation_number = innovation_number_generator()
        else:
            self.innovation_number = innovation_number

        self.method_name = method_name
        self.qubits = qubits
        self.parameters = parameters

        self.specs = qiskit_gate_specifications[self.method_name]

        # the number of parameters and number of qubits provided need to be the
        # same as in the specifications
        assert len(self.qubits) == len(self.specs.qubits)

        assert len(self.parameters) == len(self.specs.parameters)

        self.enabled = True

    def copy(self, new_innovation_number: bool = False) -> Gate:
        """
        Creates a deep copy of this Gate.

        Args:
            new_innovation_number: if True, the copied gate will have a new innovation
                number instead of this gates innovation number (useful for mutating an
                existing gate into a new one).

        Returns:
            A deep copy of this gate, potentially with a new innovation number.
        """

        innovation_number = self.innovation_number
        if new_innovation_number:
            innovation_number = None

        return Gate(
            depth=self.depth,
            method_name=self.method_name,
            qubits=self.qubits.copy(),
            parameters=self.parameters.copy(),
            innovation_number=innovation_number,
        )

    def add_to_qiskit_circuit(
        self, registers: dict[str, QuantumRegister], circuit: QuantumCircuit
    ):
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
            argument_name = self.specs.qubits[i]
            print(
                f"\tsetting argument '{argument_name}' = '{qubit_name}[{qubit_index}]'"
            )

            # assign the values for the qubit arguments to the method
            # name
            qubit_args[argument_name] = registers[qubit_name][qubit_index]

        gate_method(**self.parameters, **qubit_args)


    # def add_to_pennylane_circuit(self, registers: dict[str, qml.Wires]):
    #     """
    #     Adds this gate to a PennyLane circuit using the provided wire registers.

    #     Args:
    #         registers: a dictionary mapping register names to PennyLane Wires objects
    #                 (e.g., created via `qml.registers({"q0": 3, "q1": 2})`).
    #     """
    #     if not self.enabled:
    #         print(f"Gate {self.method_name} is disabled; skipping.")
    #         return

    #     # Get the PennyLane gate class
    #     try:
    #         gate_cls = getattr(qml, self.method_name)
    #     except AttributeError:
    #         raise ValueError(f"PennyLane does not have a gate named '{self.method_name}'")

    #     # Build the wire list for this gate
    #     qubit_wires = []
    #     for i, qubit in enumerate(self.qubits):
    #         reg_name, index = qubit
    #         reg_wires = registers[reg_name]  # qml.Wires object

    #         if index is None:
    #             # Apply gate to whole register
    #             qubit_wires.extend(reg_wires)
    #         else:
    #             qubit_wires.append(reg_wires[index])

    #     # Apply the gate with parameters if any
    #     print(f"Adding {self.method_name} on wires {qubit_wires} with params {self.parameters}")
    #     if self.parameters:
    #         gate_cls(*self.parameters.values(), wires=qubit_wires)
    #     else:
    #         gate_cls(wires=qubit_wires)


    def add_to_pennylane_circuit(self, registers: dict[str, qml.Wires]):
        """
        Adds this gate to a PennyLane circuit using registers.
        Uses native PennyLane gates if available, otherwise falls back to decomposition.

        Args:
            registers: a dictionary mapping register names to PennyLane Wires objects
                    (e.g., created via `qml.registers({"q0": 3, "q1": 2})`).
        """
        if not self.enabled:
            print(f"Gate {self.method_name} is disabled; skipping.")
            return

        spec = pennylane_gate_specifications[self.method_name]

        # Build wire list
        qubit_wires = []
        for i, (reg_name, qubit_index) in enumerate(self.qubits):
            reg_wires = registers[reg_name]
            if qubit_index is None:
                qubit_wires.extend(reg_wires)
                print(f"\tGate argument '{spec.qubits[i]}' = whole register '{reg_name}'")
            else:
                qubit_wires.append(reg_wires[qubit_index])
                print(f"\tGate argument '{spec.qubits[i]}' = '{reg_name}[{qubit_index}]'")

        # Decide whether to use native gate or decomposition
        use_native = hasattr(qml, self.method_name) and not getattr(spec, "needs_validation", False)

        if use_native:
            gate_cls = getattr(qml, self.method_name)
            if self.parameters:
                gate_cls(*self.parameters.values(), wires=qubit_wires)
            else:
                gate_cls(wires=qubit_wires)
            print(f"Added native gate {self.method_name} on wires {qubit_wires}")
        else:
            # Use decomposition
            if self.method_name not in globals():
                raise ValueError(f"No decomposition function found for gate '{self.method_name}'")

            decomp_fn = globals()[self.method_name]

            # Handle multi-qubit gates
            # For example: mcx(controls=[...], target=...)
            # We'll automatically split controls vs target if needed
            # Simple heuristic: last wire is target, rest are controls
            wires = qubit_wires
            if self.method_name in ["mcx", "mcrx", "mcry", "mcrz", "mcp"]:
                # parametric multi-controlled gates
                param = list(self.parameters.values())[0] if self.parameters else 0.0
                *controls, target = wires
                decomp_fn(param, controls, target)
            elif self.method_name in ["cs", "csdg", "csx", "sx", "sxdg", "dcx", "ecr", "rccx", "rcccx", "cu", "rzx", "ms"]:
                # functions expect either (control, target) or individual qubits
                if self.parameters:
                    decomp_fn(*self.parameters.values(), *wires)
                else:
                    decomp_fn(*wires)
            else:
                # generic case: pass wires directly
                if self.parameters:
                    decomp_fn(*self.parameters.values(), wires=wires)
                else:
                    decomp_fn(*wires)

            print(f"Added decomposed gate {self.method_name} on wires {qubit_wires}")

