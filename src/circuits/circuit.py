from __future__ import annotations
from typing import Dict, Optional
import bisect

from qiskit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
import pennylane as qml
import torch

from src.circuits.gate import Gate


class CircuitGenome:

    def __init__(self, genome_number: int, registers: dict[str, int] = None):
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
            genome_number=genome_number, registers=self.registers.copy()
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

        return circuit

    def _register_wire_map(self):
        """Return a dict mapping register names to PennyLane wires."""
        wire_map = {}
        offset = 0
        for name, size in self.registers.items():
            wire_map[name] = list(range(offset, offset + size))
            offset += size
        return wire_map


    def generate_pennylane_circuit(
        self,
        device_name: str = "default.qubit",
        measure_registers: bool = True,
        shots: Optional[int] = None,
    ):
        """
        Converts this genome into a PennyLane QNode-ready function.

        Args:
            device_name: Name of the PennyLane device to use.
            measure_registers: If True, return measurement results for all wires (like Qiskit classical registers).

        Returns:
            A tuple (dev, qnode_fn), where `dev` is the PennyLane device and
            `qnode_fn` is a QNode function that implements this circuit genome.
        """
        # 1️⃣ Create wire registers via qml.registers
        total_qubits = sum(self.registers.values())
        # registers = qml.registers(dict(self.registers.items()))
        registers = self._register_wire_map()
        print(f"pennylane register: {registers}")

        # 2️⃣ Instantiate PennyLane device
        dev = qml.device(device_name, wires=total_qubits, shots=shots,)

        # 3️⃣ Define the QNode function
        @qml.qnode(dev, interface="torch")
        def qnode_fn(
            input_bits: torch.Tensor,
            params: Dict[str, torch.Tensor],
            ):
            # Basis state preparation
            qml.BasisState(input_bits, wires=range(total_qubits))

            # Apply all gates in depth order
            self.sort_gates()
            for gate in self.gates:
                gate.add_to_pennylane_circuit(registers, params=params)

            # Optional: measure all wires in computational basis
            # For PennyLane, measuring in classical bits is optional; return state
            # return qml.state()

            # 4️⃣ Measurement
            # if measure_registers:
            #     # Return computational basis samples for each register
            #     # Flatten all register wires for measurement
            #     all_wires = []
            #     for reg_wires in registers.values():
            #         all_wires.extend(reg_wires)
            #     return qml.sample(wires=all_wires)
            # else:
            #     # Return full state vector
            #     return qml.state()

            return qml.state()

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
