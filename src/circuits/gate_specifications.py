"""
The following dict provides information about all the possible gates that can
be applied to a QuantumCircuit.

The key for an entry is the method name to be used on the QuantumCircuit object.
 * 'name' for an entry is a long form name for the gate.
 * if 'parameters' is present, it specifies the parameter names taken by the gate
    method. These need to be in the same order that the gate method accepts these
    arguments (which typically come before any qubit arguments).
 * 'qubits' specfies the qubit argument names. These need to be in the same order
    as the game method that accepts them (and they typically come after any
    qubit arguments).
"""

from __future__ import annotations
from loguru import logger


class GateSpecification:
    def __init__(
        self,
        name: str,
        qubits: list[str],
        parameters: list[str] = [],
        needs_validation: bool = False,
        pennylane_op: str | None = None,
    ):
        """
        Initializes a gate specification object which tracks the qiskit method name, formal name,
        input/control/target qubits, parameter names and if we need to further validate it before being used.

        Args:
            name: is a formal full name for the gate
            qubits: is a list of the qubit arguments to the qiskit method
            parameters: is a list of the parameter names for the non-qubit arguments to the qiskit method
            needs_validation: can be set to true for a gate method we know exists but we have not yet validated
                how to use it correctly, so it will be turned off for testing and the EXAQC algorithm.
            pennylane_op: method name in pennylane
        """

        self.name = name
        self.qubits = qubits
        self.parameters = parameters
        self.needs_validation = needs_validation
        self.pennylane_op = pennylane_op

        # this will be set when when the GateSpecification is added to a
        # GateSpecifications object in the __setitem__ method.
        self.method_name = None

        self.input_qubit_indexes = []
        self.output_qubit_indexes = []

        for i, qubit in enumerate(self.qubits):
            if "control_" in qubit:
                self.input_qubit_indexes.append(i)
            elif "target_" in qubit:
                # target (output) qubits are also inputs
                self.input_qubit_indexes.append(i)
                self.output_qubit_indexes.append(i)
            else:
                # qubit is both an input and output
                self.input_qubit_indexes.append(i)
                self.output_qubit_indexes.append(i)

    def __str__(self) -> str:
        """
        Returns:
            A human readable string of this gate
        """

        return f"'{self.name}' : {self.method_name}(qubits={self.qubits}, params={self.parameters})"

    @property
    def n_qubits(self) -> int:
        """Number of qubits this gate acts on (control + target)."""
        return len(self.qubits)

    @property
    def pl_qubits(self) -> list[str]:
        """
        For PennyLane, all gates take a single 'wires' argument at runtime.
        """
        return ["wires"]


class GateSpecifications:
    def __init__(self, target: str):
        """
        Constructs gate specifications for either qiskit or pennylane

        Args:
            target: should be either 'qiskit' or 'pennylane', specifying which
                target framework these are for.
        """

        if target not in ["qiskit", "pennylane"]:
            logger.error(
                f"ERROR: target framework '{target}' was neither 'qiskit' nor 'pennylane'"
            )
            exit(1)

        self.target = target

        self.specifications = {}

    def use_only(self, allowed_methods: list[str]) -> GateSpecifications:
        """
        Used to create a new GateSpecifications object which only has the
        provided gate methods.

        Args:
            allow_methods: are the gate method names for the gates to be
                kept and used.
        """

        new_specs = GateSpecifications(self.target)

        for method_name, specs in self.specifications.items():
            if method_name in allowed_methods:
                new_specs[method_name] = specs

        return new_specs

    def __setitem__(self, method_name: str, gate_specification: GateSpecification):
        """
        Adds a new gate specification to this dict of gate specifications.

        Args:
            method_name: is the qiskit method name for applying this gate to a QuantumCircuit
            gate_specification: is the GateSpecifcation object containing all the information.
        """

        gate_specification.method_name = method_name
        self.specifications[method_name] = gate_specification

    def __getitem__(self, method_name: str) -> GateSpecification:
        """
        Gets a gate specification from this dict of gate specifications.

        Args:
            method_name: is the qiskit method name for applying this gate to a QuantumCircuit

        Returns:
            The GateSpecification for the given qiskit method name
        """

        return self.specifications[method_name]

    def keys(self) -> list[str]:
        """
        Returns:
            All the gate method names (all the keys) of the specifications dict
        """

        return self.specifications.keys()

    def values(self) -> list[str]:
        """
        Returns:
            All the GateSpecifications (all the values) of the specifications dict
        """

        return self.specifications.values()

    def items(self) -> tuple[str, GateSpecification]:
        """
        Returns:
            The items listing for the dict of gate specifications.
        """

        return self.specifications.items()
