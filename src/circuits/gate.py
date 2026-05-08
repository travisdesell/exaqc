from __future__ import annotations

from loguru import logger

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter as QiskitParameter
import pennylane as qml
import torch

from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.evolution.innovation import innovation_number_generator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome


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

    def get_input_circuit_indexes(self, circuit: CircuitGenome) -> list[int]:
        """
        Determines which qubit indexes in the circuit genome this gate uses as inputs, given
        the gate specification and its qubit names.

        Args:
            circuit: is the CircuitGenome to determine which qubit indexes this gate
                uses.

        Returns:
            A list of ints, where each is an index of a qubit in the circuit.
        """

        input_indexes = []

        # logger.debug(f'getting input qubits from circuit with qubits: {circuit.qubits}')

        for gate_qubit_index in self.specs.input_qubit_indexes:
            # get the qubit touple (register name and index in the register)
            # for the gate
            gate_qubit = self.qubits[gate_qubit_index]

            # figure out which index this qubit has in the circuit
            qubit_index = circuit.qubits.index(gate_qubit)

            input_indexes.append(qubit_index)

        return input_indexes

    def get_output_circuit_indexes(self, circuit: CircuitGenome) -> list[int]:
        """
        Determines which qubit indexes in the circuit genome this gate uses as outputs, given
        the gate specification and its qubit names.

        Args:
            circuit: is the CircuitGenome to determine which qubit indexes this gate
                uses.

        Returns:
            A list of ints, where each is an index of a qubit in the circuit.
        """

        output_indexes = []

        # logger.debug(f'getting output qubits from circuit with qubits: {circuit.qubits}')

        for gate_qubit_index in self.specs.output_qubit_indexes:
            # get the qubit touple (register name and index in the register)
            # for the gate
            gate_qubit = self.qubits[gate_qubit_index]

            # figure out which index this qubit has in the circuit
            qubit_index = circuit.qubits.index(gate_qubit)

            output_indexes.append(qubit_index)

        return output_indexes

    def to_dict(self) -> dict[str, any]:
        """
        Creates a dict version of this gate that can be used for MPI serialization or
        saving the gate to a JSON file.  This can be used to reconstruct the gate using
        Gate.from_dict().
        """
        serialized = {}
        serialized["depth"] = self.depth
        serialized["method_name"] = self.method_name
        serialized["qubits"] = self.qubits.copy()
        serialized["parameters"] = self.parameters.copy()
        serialized["innovation_number"] = self.innovation_number
        serialized["target"] = self.target
        serialized["enabled"] = self.enabled
        return serialized

    @classmethod
    def from_dict(cls, serialized: dict[str, any]) -> Gate:
        """
        Args:
            serialized: is a serialized Gate dict created by the to_dict method.

        Returns:
            A deserialized Gate from the serialized dict.
        """
        new_gate = Gate(
            depth=serialized["depth"],
            method_name=serialized["method_name"],
            qubits=serialized["qubits"],
            parameters=serialized["parameters"],
            innovation_number=serialized["innovation_number"],
            target=serialized["target"],
        )

        new_gate.enabled = serialized["enabled"]

        return new_gate

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
        self,
        register_dict: dict[tuple[str, int], QuantumRegister],
        circuit: QuantumCircuit,
        parametric: bool = False,
    ):
        """
        Adds this gate to the qiskit QuantumCircuit using reflection
        to specify the correct method along with the given parameters.

        Args:
            register_dict: is a dict of qubit tuples to the appropriate quantum register
            circuit: is the qiskit QuantumCircuit to add this gate to
            parametric: if True, replace numeric parameter values with qiskit
                Parameter symbols (cached on this Gate) so the resulting
                QuantumCircuit can be used with autodiff via SamplerQNN /
                EstimatorQNN. The float values stay in self.parameters and
                are used as the initial values for the Parameters at bind time.
        """

        logger.debug(
            f"adding gate {self.method_name}(qubits={self.qubits}, params={self.parameters}"
        )

        gate_method = getattr(circuit, self.method_name)

        # Look up the qiskit spec at call time so a genome built with
        # target="pennylane" can still be rendered as a qiskit circuit.
        qiskit_specs = qiskit_gate_specifications[self.method_name]

        qubit_args = {}

        for i, qubit in enumerate(self.qubits):
            qubit_name = qubit[0]
            qubit_index = qubit[1]
            argument_name = qiskit_specs.qubits[i]
            logger.debug(
                f"\tsetting argument '{argument_name}' = '{qubit_name}[{qubit_index}]'"
            )

            # assign the values for the qubit arguments to the method
            # name
            qubit_args[argument_name] = register_dict[qubit]

        if parametric and self.parameters:
            param_args = self._get_qiskit_parameters()
        else:
            param_args = self.parameters

        gate_method(**param_args, **qubit_args)

    def _get_qiskit_parameters(self) -> dict[str, "QiskitParameter"]:
        """Return a dict of qiskit.circuit.Parameter objects for this gate.

        Allocated lazily and cached so the same Parameter objects are reused
        across multiple builds of the same parametric circuit (e.g. when the
        QNN is re-instantiated). The Parameter name encodes the gate's
        innovation number to keep names unique across the genome.

        Returns:
            Dict mapping param name -> qiskit Parameter symbol.
        """
        cache = getattr(self, "_qiskit_parameter_cache", None)
        if cache is None:
            cache = {
                pname: QiskitParameter(f"g{self.innovation_number}_{pname}")
                for pname in self.parameters.keys()
            }
            self._qiskit_parameter_cache = cache
        return cache

    def get_pennylane_wires(
        self,
        circuit_qubits: list[tuple[str, int]],
    ) -> list[int]:
        """Return the PennyLane wire indexes used by this gate.

        Args:
            circuit_qubits: Ordered list of all qubit tuples in the circuit.

        Returns:
            List of PennyLane wire indexes corresponding to the qubits acted on
            by this gate.
        """
        if not self.enabled:
            return []

        spec = pennylane_gate_specifications[self.method_name]
        n_qubits = getattr(spec, "n_qubits", len(self.qubits))

        qubit_wires = []
        for i in range(n_qubits):
            qubit_wires.append(circuit_qubits.index(self.qubits[i]))

        return qubit_wires
    

    def add_to_pennylane_circuit(
        self,
        circuit_qubits: list[tuple[str, int]],
        params: dict[str, torch.Tensor] = None,
    ):
        """
        Adds this gate to a PennyLane circuit using the provided wire registers.

        Handles native PennyLane gates, decompositions for unsupported gates,
        multi-qubit / controlled gates, parametric gates, and adjoint (.adjoint) gates.

        Args:
            circuit_qubits: the list of all qubits in the circuit so we can map the qubit names
                to wire indexes (ints).
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
            qubit_wires.append(circuit_qubits.index(self.qubits[i]))

        # Resolve parameters
        if params is not None:
            param_values = [
                params[f"{self.innovation_number}:{name}"] for name in self.parameters
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
        circuit_qubits: list[tuple[str, int]],
    ):
        """
        Print EXACTLY the same messages as add_to_pennylane_circuit(),
        but without executing any PennyLane operations.
        This should be called ONCE at circuit generation time.

        Args:
            circuit_qubits: is the list of qubit tuples so we can index the parameter
                qubit tuples to get the wire index (int).
        """
        if not self.enabled:
            logger.debug(f"Gate {self.method_name} is disabled; skipping.")
            return

        spec = pennylane_gate_specifications[self.method_name]
        n_qubits = getattr(spec, "n_qubits", len(self.qubits))

        pennylane_op_name = getattr(spec, "pennylane_op", None)

        # Build qubit wire list
        qubit_wires = []
        for i in range(n_qubits):
            qubit_wires.append(circuit_qubits.index(self.qubits[i]))

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
