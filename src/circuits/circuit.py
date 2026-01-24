from __future__ import annotations
from loguru import logger
from typing import Dict, Optional
import bisect

from qiskit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
import pennylane as qml
import torch

from src.circuits.gate import Gate
from src.utils.helpers import register_wire_map


class CircuitGenome:

    def __init__(
        self,
        genome_number: int,
        target: int,
        registers: dict[str, int] = None,
        output_qubits: list[int] = None,
    ):
        """
        Initializes an empty quantum circuit.

        Args:
            genome_number: a unique identifier for this evolved circuit, which also represents the ordering
                that genomes have been generated, e.g., 0 is the first genome created, 1 is the next, etc.
            target: specifies if the circuit is for qiskit or pennylane
            registers: a dict of register names and sizes (the key is the qubit name, the value is its size)
            output_qubits: the list of indexes that corrspond to the output qubits
        """
        self.genome_number = genome_number

        # a dict of cubits where the key is the cubit name is the name of the
        # input quantum register and the value is an instantiated quantum register
        self.registers: dict[str, int] = registers

        # create a list of input qubits (which are tuples of register names and indexes)
        # so we can easily select random qubits to use for gate mutations
        self.qubits: list[str, int] = []

        for gate_name, gate_size in registers.items():
            for index in range(gate_size):
                self.qubits.append((gate_name, index))

        # a list of Gates sorted by depth represnting the gates in the quantum
        # circuit
        self.gates: list[Gate] = []

        self.target = target

        # if a genome has not yet been evaluated, its fitness is None
        self.fitness = None

        # the inherent circuit is set to None
        self.circuit = None

        # mark the indexes of the output qubits
        self.output_qubits = output_qubits
        if self.output_qubits is None:
            self.output_qubits = list(range(sum(registers.values())))

    def dominates(self, other: CircuitGenome) -> bool:
        """
        Determines if this genome dominates another genome. This method is needed because
        in the multi-objective case we can't just compare a single fitness value to determine
        if one genome is better than another.

        Args:
            other: is the other genome to compare to.

        Returns:
            True if this genome dominates another genome.
        """

        # TODO: update for multi objectives, but for now just use fidelity loss
        return self.fitness["loss"] < other.fitness["loss"]

    def copy(self, genome_number: int = None) -> CircuitGenome:
        """
        Creates a deep copy of this CircuitGenome, with potentially a new
        genome_number if it will be used as a child genome, e.g. for crossover
        or mutation.

        Args:
            genome_number: if this is specified, the copy will use this new genome
                number. This also means the fitness should be set to None as it will
                be modified via crossover or mutation.

        Returns:
            A copy of this genome, with potentially modified genome number and fitness.
        """

        fitness = self.fitness

        if genome_number is None:
            genome_number = self.genome_number
            fitness = None

        new_genome = CircuitGenome(
            genome_number=genome_number,
            target=self.target,
            registers=self.registers.copy(),
            output_qubits=self.output_qubits.copy(),
        )
        new_genome.fitness = fitness

        for gate in self.gates:
            new_genome.add_existing_gate(gate)

        return new_genome

    def add_existing_gate(self, gate: Gate):
        """
        Adds a new already created gate to this quantum circuit, keeping the
        gates in order sorted first by depth and then by innovation number to
        handle any gates with the same depth (which shouldn't usually happen).

        Args:
            gate: is the gate to add.
        """

        bisect.insort(self.gates, gate, key=lambda g: (g.depth, g.innovation_number))

    def add_gate(
        self,
        depth: float,
        method_name: str,
        qubits: list[tuple[str, int]] = [],
        parameters: dict[str, float] = {},
    ):
        """
        Adds a new already created gate to this quantum circuit, keeping the
        gates in order sorted first by depth and then by innovation number to
        handle any gates with the same depth (which shouldn't usually happen).

        Args:
            depth: a number between 0 and 1 representing the depth of the gate in the circuit.
            method_name: the name of the method to invoke this gate on a qiskit QuantumCircuit
            qubits: a list of qubits to form the arguments to the gate method name, each is a tuple
                with a string for the input register name and then the index.
            parameters: a dict where the key is the parameter name and the value is the parameter value
        """

        gate = Gate(
            depth=depth,
            method_name=method_name,
            qubits=qubits,
            parameters=parameters,
            target=self.target,
        )
        # make sure to add the gate in sorted order
        bisect.insort(self.gates, gate, key=lambda g: (g.depth, g.innovation_number))

    def sort_gates(self):
        """
        Sorts the gates in the circuit by their depth (useful if new gates are
        added or the circuit is mutated).

        Sort the gates first by depth then by innovation number (in case two gates
        somehow had the same depth).
        """
        self.gates.sort(key=lambda g: (g.depth, g.innovation_number))

    def get_possible_output_qubits(self, depth: float) -> list[int]:
        """
        Traces back the gates from the output to a given depth to determine which output
        qubits will effect any of the final output gates that are being measured.

        Args:
            depth: the depth a new gate will be added at, the results of this method will be
                used to determine which qubits can be used as output (target) parameters for
                the gate.

        Returns:
            A list of potential qubit indexes in this circuit that will effect the output
            qubits.
        """
        # determine which qubits this gate can be applied to so it will effect the output qubits
        reverse_gates = sorted(
            self.gates, key=lambda g: (g.depth, g.innovation_number), reverse=True
        )

        possible_output_indexes = set(self.output_qubits)

        for gate in reverse_gates:
            output_circuit_indexes = gate.get_output_circuit_indexes(self)
            input_circuit_indexes = gate.get_input_circuit_indexes(self)

            # if any of the output indexes for the gate are in the possible output
            # qubit indexes, then this gate effects the output and we can add its
            # inputs as effecting the output
            if not set(output_circuit_indexes).isdisjoint(possible_output_indexes):
                possible_output_indexes.update(input_circuit_indexes)

            if gate.depth <= depth:
                # we've gone through all gates ahead of the insertion
                # depth for this new gate.
                break

            if len(possible_output_indexes) == len(self.qubits):
                # all gates are possible so we can quit checking
                break

        return sorted(possible_output_indexes)

    def generate_qiskit_circuit(self) -> QuantumCircuit:
        """
        Converts this genome into a useable qiskit instationation.

        Returns:
            A qiskit QuantumCircuit instantiation of this circuit genome.
        """
        self.total_qubits = sum(self.registers.values())
        quantum_registers = {}
        classical_registers = {}
        for name, size in self.registers.items():
            quantum_registers[name] = QuantumRegister(size, name=name)
            classical_registers[name] = ClassicalRegister(size)

        circuit = QuantumCircuit(
            *quantum_registers.values(), *classical_registers.values()
        )

        # make sure we apply the gates in the correct ordering by depth
        self.sort_gates()
        for gate in self.gates:
            gate.add_to_qiskit_circuit(quantum_registers, circuit)

        for name, classical_register in classical_registers.items():
            circuit.measure(quantum_registers[name], classical_register)

        self.circuit = circuit

        return circuit

    # def _register_wire_map(self) -> dict:
    #     """Return a dict mapping register names to PennyLane wires."""
    #     wire_map = {}
    #     offset = 0
    #     for name, size in self.registers.items():
    #         wire_map[name] = list(range(offset, offset + size))
    #         offset += size
    #     return wire_map

    def generate_pennylane_circuit(
        self,
        device_name: str = "lightning.qubit",
        measure_registers: bool = True,
        shots: Optional[int] = None,
        input_mode: str = "basis",
        return_probs: bool = True,
    ):
        """
        Converts this genome into a PennyLane QNode-ready function.

        Args:
            device_name: Name of the PennyLane device to use.
            measure_registers: If True, return measurement results for all wires (like Qiskit classical registers).
            shots: Sample from circuit output,
            input_mode: choose encoding paradigm "basis" or "angle"
            return_probs: Return the actual probablity weights from the circuit

        Returns:
            A tuple (dev, qnode_fn), where `dev` is the PennyLane device and
            `qnode_fn` is a QNode function that implements this circuit genome.
        """
        # Create wire registers via qml.registers
        self.total_qubits = sum(self.registers.values())
        # registers = qml.registers(dict(self.registers.items()))
        # self.register_map = self._register_wire_map()

        self.register_map = register_wire_map(self.registers)

        registers = self.register_map.copy()
        logger.info(f"pennylane register: {registers}")

        # Instantiate PennyLane device
        dev = qml.device(
            device_name,
            wires=self.total_qubits,
            shots=shots,
        )

        # Define the QNode function
        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def qnode_fn(
            input_bits: torch.Tensor,
            params: Dict[str, torch.Tensor],
        ):

            # --- Input preparation ---
            if input_mode == "basis":
                # expects int tensor length == total_qubits
                qml.BasisState(input_bits, wires=range(self.total_qubits))
            elif input_mode == "angle":
                # expects float tensor on "input" register wires
                # encode x_i in [0,1] -> RY(pi*x_i) (common, stable)
                in_wires = registers["input"]
                for i, w in enumerate(in_wires):
                    qml.RY(torch.pi * input_bits[i], wires=w)
            else:
                raise ValueError(f"Unknown input_mode={input_mode}")

            # Apply all gates in depth order
            self.sort_gates()
            for gate in self.gates:
                gate.add_to_pennylane_circuit(registers, params=params)

            # 4️⃣ Measurement
            if return_probs:
                return qml.probs(
                    # wires=self.register_map["output"]
                    wires=self.output_qubits
                )  # shape = [2**len(out_wires)] (real)
            elif measure_registers:
                # fallback if you want expvals
                expvals = [
                    qml.expval(qml.PauliZ(w))
                    for w in self.output_qubits  # self.register_map["output"]
                ]
                return torch.stack(expvals)

            return qml.state()

        self.circuit = qnode_fn

        # 🔹 PRINT ONCE HERE
        self.sort_gates()
        for gate in self.gates:
            gate.describe_pennylane_circuit(registers)

        return dev, qnode_fn
