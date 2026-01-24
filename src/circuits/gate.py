from __future__ import annotations

from loguru import logger

from qiskit import QuantumCircuit, QuantumRegister
import pennylane as qml
import torch

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
        target: str = "qiskit",
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
            target: denotes whether you are adding qiskit or pennylane gates
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

        self.target = target
        if self.target == "qiskit":
            self.specs = qiskit_gate_specifications[self.method_name]
        else:
            self.specs = pennylane_gate_specifications[self.method_name]

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
            target=self.target,
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

        logger.debug(
            f"adding gate {self.method_name}(qubits={self.qubits}, params={self.parameters}"
        )

        gate_method = getattr(circuit, self.method_name)

        qubit_args = {}

        for i, qubit in enumerate(self.qubits):
            qubit_name = qubit[0]
            qubit_index = qubit[1]
            argument_name = self.specs.qubits[i]
            logger.debug(
                f"\tsetting argument '{argument_name}' = '{qubit_name}[{qubit_index}]'"
            )

            # assign the values for the qubit arguments to the method
            # name
            qubit_args[argument_name] = registers[qubit_name][qubit_index]

        gate_method(**self.parameters, **qubit_args)

    def add_to_pennylane_circuit(
        self, registers: dict[str, list], params: dict[str, torch.Tensor] = None
    ):
        """
        Adds this gate to a PennyLane circuit using the provided wire registers.

        Handles native PennyLane gates, decompositions for unsupported gates,
        multi-qubit / controlled gates, parametric gates, and adjoint (.adjoint) gates.

        Args:
            registers: a dictionary mapping register names to PennyLane wires (lists of ints).
            params: optional dictionary mapping "{innovation_number}:{param_name}" to
                    trainable torch.Tensor values. If None, uses self.parameters values.
        """
        if not self.enabled:
            # logger.debug(f"Gate {self.method_name} is disabled; skipping.")
            return

        spec = pennylane_gate_specifications[self.method_name]
        n_qubits = getattr(spec, "n_qubits", len(self.qubits))

        # Build qubit wire list
        qubit_wires = []
        for i in range(n_qubits):
            reg_name, qubit_index = self.qubits[i]
            reg_wires = registers[reg_name]
            if qubit_index is None:
                qubit_wires.extend(reg_wires)
            else:
                qubit_wires.append(reg_wires[qubit_index])

        # Resolve parameters
        if params is not None:
            param_values = [
                # params[f"{self.innovation_number}:{name}"]
                # for name in self.parameters
                params[f"{name}"]
                for name in self.parameters
            ]
        else:
            param_values = list(self.parameters.values())

        pennylane_op_name = getattr(spec, "pennylane_op", None)

        decomposition_module = __import__(
            "src.circuits.pennylane_decompositions", fromlist=["*"]
        )

        try:
            if pennylane_op_name is not None:
                # Handle adjoint gates
                if ".adjoint" in pennylane_op_name:
                    base_op_name = pennylane_op_name.split(".")[0]
                    gate_cls = getattr(qml, base_op_name)
                    # Apply adjoint to all wires
                    gate_cls(*param_values, wires=qubit_wires).adjoint()
                    # logger.debug(
                    #     f"Added adjoint gate {self.method_name} on wires {qubit_wires}"
                    # )
                    return

                # Regular gate
                gate_cls = getattr(qml, pennylane_op_name)
                gate_cls(*param_values, wires=qubit_wires)
                # logger.debug(
                #     f"Added native gate {self.method_name} ({pennylane_op_name}) on wires {qubit_wires}"
                # )

            else:
                # Use decomposition if native gate unavailable
                decomp_func = getattr(decomposition_module, self.method_name, None)
                if decomp_func is None:
                    raise ValueError(
                        f"No decomposition found for gate '{self.method_name}'"
                    )
                decomp_func(*param_values, *qubit_wires)
                # logger.debug(
                #     f"Added decomposed gate {self.method_name} on wires {qubit_wires}"
                # )

        except Exception as e:
            logger.error(f"Failed to add gate {self.method_name}: {e}")
            raise

    def describe_pennylane_circuit(
        self,
        registers: dict[str, list],
    ):
        """
        Print EXACTLY the same messages as add_to_pennylane_circuit(),
        but without executing any PennyLane operations.
        This should be called ONCE at circuit generation time.
        """
        if not self.enabled:
            logger.debug(f"Gate {self.method_name} is disabled; skipping.")
            return

        spec = pennylane_gate_specifications[self.method_name]
        n_qubits = getattr(spec, "n_qubits", len(self.qubits))

        # Build qubit wire list (IDENTICAL LOGIC)
        qubit_wires = []
        for i in range(n_qubits):
            reg_name, qubit_index = self.qubits[i]
            reg_wires = registers[reg_name]
            if qubit_index is None:
                qubit_wires.extend(reg_wires)
            else:
                qubit_wires.append(reg_wires[qubit_index])

        pennylane_op_name = getattr(spec, "pennylane_op", None)

        if pennylane_op_name is not None:
            if ".adjoint" in pennylane_op_name:
                logger.debug(
                    f"Added adjoint gate {self.method_name} on wires {qubit_wires}"
                )
                return

            logger.debug(
                f"Added native gate {self.method_name} ({pennylane_op_name}) on wires {qubit_wires}"
            )
        else:
            logger.debug(
                f"Added decomposed gate {self.method_name} on wires {qubit_wires}"
            )
