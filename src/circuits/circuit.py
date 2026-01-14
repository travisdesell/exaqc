import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister

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
        # a dict of cubits where the key is the cubit name is the name of the
        # input quantum register and the value is an instantiated quantum register
        self.registers : dict[str, int] = registers

        # create a list of input qubits (which are tuples of register names and indexes)
        # so we can easily select random qubits to use for gate mutations
        self.qubits: list[str, int] = []

        for gate_name, gate_size in registers.items():
            for index in range(gate_size):
                self.qubits.append((gate_name, index))

        # a list of Gates sorted by depth represnting the gates in the quantum
        #circuit
        self.gates : list[Gate] = []

    def add_existing_gate(self, gate: Gate):
        """
        Adds a new already created gate to this quantum circuit.

        Args:
            gate: is the gate to add.
        """

        self.gates.append(gate)
        self.sort_gates()

    def add_gate(self, depth: float, method_name: str, qubits: list[tuple[str,int]] = [], parameters: dict[str,float] = {}):
        """
        Adds a new gate to this quantum circuit at the given depth.

        Args:
            depth: a number between 0 and 1 representing the depth of the gate in the circuit.
            method_name: the name of the method to invoke this gate on a qiskit QuantumCircuit
            qubits: a list of qubits to form the arguments to the gate method name, each is a tuple with a string for the input
                register name and then the index. if the gate takes the whole register (and not individual qubits) then the second
                value will be None (e.g., for a Hadamard gate).
            parameters: a dict where the key is the parameter name and the value is the parameter value
        """

        self.gates.append(Gate(depth, method_name, qubits, parameters))
        self.sort_gates()

    def sort_gates(self):
        '''
        Sorts the gates in the circuit by their depth (useful if new gates are
        added or the circuit is mutated).

        Sort the gates first by depth then by innovation number (in case two gates
        somehow had the same depth).
        '''
        self.gates.sort(key = lambda g : (g.depth, g.innovation_number))

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

        circuit = QuantumCircuit(*quantum_registers.values(), *classical_registers.values())

        # make sure we apply the gates in the correct ordering by depth
        self.sort_gates()
        for gate in self.gates:
            gate.add_to_circuit(quantum_registers, circuit)

        for name, classical_register in classical_registers.items():
            circuit.measure(quantum_registers[name], classical_register)

        return circuit


'''
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
'''
