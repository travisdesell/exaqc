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
        registers: dict[str, int] = None,
        output_qubits: list[int] = None,
    ):
        """
        Initializes an empty quantum circuit.

        Args:
            genome_number: a unique identifier for this evolved circuit, which also represents the ordering
                that genomes have been generated, e.g., 0 is the first genome created, 1 is the next, etc.
            registers: a dict of register names and sizes (the key is the qubit name, the value is its size)
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

        gate = Gate(depth, method_name, qubits, parameters)
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
        device_name: str = "default.qubit",
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


"""
qc = CircuitGenome(genome_number=1, registers={"a" : 3, "b" : 5})

qc.add_gate(depth=0.05, method_name='x', qubits=[('a', 1)])
qc.add_gate(depth=0.10, method_name='x', qubits=[('b', 1)])
qc.add_gate(depth=0.15, method_name='x', qubits=[('b', 2)])
qc.add_gate(depth=0.20, method_name='x', qubits=[('b', 4)])

qc.add_gate(depth=0.25, method_name='h', qubits=[('a', 0)])
qc.add_gate(depth=0.30, method_name='h', qubits=[('b', 1)])

qc.add_gate(depth=0.31, method_name='rx', qubits=[('b', 1)], parameters={'theta': 0.2})

qc.add_gate(depth=0.35, method_name='cp', qubits=[('b', 3), ('a', 0)], parameters={'theta': 0.3})

qc.add_gate(depth=0.40, method_name='ccz', qubits=[('b', 0), ('b', 1), ('b',3)])
qc.add_gate(depth=0.40, method_name='cswap', qubits=[('b', 0), ('b', 1), ('b',2)])
qc.add_gate(depth=0.40, method_name='cswap', qubits=[('b', 0), ('b', 1), ('b',2)])
qc.add_gate(depth=0.45, method_name='cswap', qubits=[('b', 2), ('b', 3), ('b',4)])
qc.add_gate(depth=0.50, method_name='cswap', qubits=[('b', 3), ('b', 4), ('b',0)])

circuit = qc.generate_qiskit_circuit()

circuit.draw(output="mpl")

plt.show()


# Draw a new circuit with barriers and more registers
q_a = QuantumRegister(3, name="a")
q_b = QuantumRegister(5, name="b")
c_a = ClassicalRegister(3)
c_b = ClassicalRegister(5)

circuit = QuantumCircuit(q_a, q_b, c_a, c_b)
#circuit = QuantumCircuit(q_a, q_b)
circuit.x(q_a[1])
circuit.x(q_b[1])
circuit.x(q_b[2])
circuit.x(q_b[4])
circuit.barrier()
#circuit.h(q_b)
circuit.h([q_a[0], q_a[2], q_b[1], q_b[3], q_b[4]])
circuit.barrier(q_a)
circuit.h(q_b)
#circuit.mcrx(0.3, [q_a[0], q_b[1], q_b[3]], q_b[2])
circuit.ms(0.3, [q_a[0], q_b[1], q_b[3], q_b[2]])
circuit.cswap(q_b[0], q_b[1], q_b[2])
circuit.cswap(q_b[2], q_b[3], q_b[4])
circuit.cswap(q_b[3], q_b[4], q_b[0])
circuit.barrier(q_b)
circuit.measure(q_a, c_a)
circuit.measure(q_b, c_b);

circuit.draw(output="mpl")

plt.show()
"""
