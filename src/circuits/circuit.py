from __future__ import annotations
from loguru import logger
from typing import Dict, Optional
import bisect
import os
import json

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
import pennylane as qml
import torch

from src.circuits.gate import Gate
from src.objectives.genome_objectives import (
    genome_to_torch_params,
)


class CircuitGenome:

    def __init__(
        self,
        genome_number: int,
        target: str,
        input_qubits: list[tuple[str, int]],
        output_qubits: list[tuple[str, int]] = None,
        metadata: dict[str, any] = {},
    ):
        """
        Initializes an empty quantum circuit.

        Args:
            genome_number: a unique identifier for this evolved circuit, which also represents the ordering
                that genomes have been generated, e.g., 0 is the first genome created, 1 is the next, etc.
            target: specifies if the circuit is for qiskit or pennylane
            input_qubits: a list of qubit names and indexes (e.g., (a, 0)).
            output_qubits: a list of qubit names and indexes (e.g., (a, 0)), if None then output_qubits are
                the same as the input qubits.
            metadata: is metadata about the genome used by things like the population strategy, etc. or to
                track other information about the genome.
        """
        self.genome_number = genome_number
        self.metadata = metadata

        # these should be specified by EXAQC

        # create a list of input qubits (which are tuples of register names and indexes)
        # so we can easily select random qubits to use for gate mutations
        self.qubits: list[tuple[str, int]] = []

        self.input_qubits: list[tuple[str, int]] = input_qubits
        for qubit in self.input_qubits:
            self.qubits.append(qubit)

        if output_qubits is None:
            output_qubits = input_qubits.copy()
        self.output_qubits: list[tuple[str, int]] = output_qubits
        for qubit in self.output_qubits:
            if qubit not in self.qubits:
                self.qubits.append(qubit)

        # make sure they are all sorted
        self.input_qubits.sort()
        self.output_qubits.sort()
        self.qubits.sort()

        # get indexes for input and output qubits in the full qubit list
        self.input_indexes = []
        for qubit in self.input_qubits:
            self.input_indexes.append(self.qubits.index(qubit))

        self.output_indexes = []
        for qubit in self.output_qubits:
            self.output_indexes.append(self.qubits.index(qubit))

        # a list of Gates sorted by depth represnting the gates in the quantum
        # circuit
        self.gates: list[Gate] = []

        self.target = target

        # if a genome has not yet been evaluated, its fitness is None
        self.fitness = None

        # the inherent circuit is set to None
        self.circuit = None

    def is_valid(self) -> bool:
        """
        Returns:
            True if there is at least a path from the input qubits to the
            output qubits through the gates (i.e., the inputs can effect
            the outputs).  False, otherwise.
        """

        # determine which qubits this gate can be applied to so it will effect the output qubits
        self.sort_gates()

        reached_indexes = set(self.input_indexes)
        logger.debug(f"inital reached indexes now: {reached_indexes}")

        for gate in self.gates:
            if not gate.enabled:
                continue

            output_circuit_indexes = gate.get_output_circuit_indexes(self)
            input_circuit_indexes = gate.get_input_circuit_indexes(self)

            # if any of the input indexes for the gate are in the reached
            # qubit indexes, then this gate is effected by the input and we can add
            # its outputs as additional possible inputs

            if not set(input_circuit_indexes).isdisjoint(reached_indexes):
                reached_indexes.update(output_circuit_indexes)

            logger.debug(f"\treached indexes now: {reached_indexes}")

        valid = not reached_indexes.isdisjoint(self.output_indexes)
        logger.debug(
            f"output indexes are: {self.output_indexes}, circuit valid? {valid}"
        )

        return valid

    def dominates(self, other: CircuitGenome, loss: str = "loss") -> bool:
        """
        Determines if this genome dominates another genome. This method is needed because
        in the multi-objective case we can't just compare a single fitness value to determine
        if one genome is better than another.

        Args:
            other: is the other genome to compare to.

        Returns:
            True if this genome dominates another genome.
        """

        # TODO: update for multi objectives, but for now just use the given
        # loss key
        return self.fitness[loss] < other.fitness[loss]

    def get_gate_innovations(self) -> list[int]:
        """
        Returns:
            A sorted list of all the enabled gate innovation numbers in this
            genome.
        """
        gates = [gate.innovation_number for gate in self.gates if gate.enabled]
        gates.sort()
        return gates

    def has_same_gates(self, other: CircuitGenome) -> bool:
        """
        This checks to see if this genome has the exact same enabled
        gates as the other genome.

        Returns:
            True if both genomes have the same enabled gates innovation nubmers (but
            gates can potentially have different trained parameters).
        """

        self_gates = self.get_gate_innovations()
        other_gates = other.get_gate_innovations()

        logger.info(
            f"comparing self gates {self_gates} to other gates {other_gates}, equal? {self_gates == other_gates}"
        )

        return self.get_gate_innovations() == other.get_gate_innovations()

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
            input_qubits=self.input_qubits.copy(),
            output_qubits=self.output_qubits.copy(),
        )
        new_genome.metadata = self.metadata.copy()
        new_genome.fitness = fitness
        new_genome.hyperparameters = self.hyperparameters.copy()

        for gate in self.gates:
            new_genome.add_existing_gate(gate)

        return new_genome

    def to_dict(self) -> dict(str, any):
        """
        Creates a dict representation of the circuit genome that can be converted to JSON
        or used for MPI serialization. This won't contain any of the qiskit or pennylane
        internals which will need to be recreated when it is loaded back with the
        CircuitGenome.from_dict method.

        Returns:
            A simple dict representation of this CircuitGenome.
        """

        serialized = {}
        serialized["fitness"] = self.fitness
        serialized["genome_number"] = self.genome_number
        serialized["metadata"] = self.metadata
        serialized["target"] = self.target
        serialized["input_qubits"] = self.input_qubits.copy()
        serialized["output_qubits"] = self.output_qubits.copy()
        serialized["hyperparameters"] = self.hyperparameters.copy()
        serialized["gates"] = []

        for gate in self.gates:
            serialized["gates"].append(gate.to_dict())

        return serialized

    @classmethod
    def from_dict(cls, serialized: dict[str, any]) -> CircuitGenome:
        """
        Args:
            serialized: is a serialized version of a CircuitGenome created
                by the to_dict method.

        Returns:
            A circuit genome created from a serialized dict of a circuit genome.
        """
        new_genome = CircuitGenome(
            genome_number=serialized["genome_number"],
            target=serialized["target"],
            input_qubits=serialized["input_qubits"],
            output_qubits=serialized["output_qubits"],
            metadata=serialized["metadata"],
        )
        new_genome.fitness = serialized["fitness"]
        new_genome.hyperparameters = serialized["hyperparameters"]

        for serialized_gate in serialized["gates"]:
            gate = Gate.from_dict(serialized_gate)
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
        innovation_number: int = None,
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

    def get_possible_input_qubits(self, depth: float) -> list[int]:
        """
        Traces back the gates from the input to a given depth to determine which input
        qubits will effect any of the final output gates that are being measured.

        Args:
            depth: the depth a new gate will be added at, the results of this method will be
                used to determine which qubits can be used as input (control) parameters for
                the gate.

        Returns:
            A list of potential qubit indexes in this circuit that will effect the output
            qubits.
        """
        # determine which qubits this gate can be applied to so it will effect the output qubits
        self.sort_gates()

        possible_input_indexes = set(self.input_indexes)

        for gate in self.gates:
            if not gate.enabled:
                continue

            output_circuit_indexes = gate.get_output_circuit_indexes(self)
            input_circuit_indexes = gate.get_input_circuit_indexes(self)

            # if any of the input indexes for the gate are in the possible input
            # qubit indexes, then this gate is effected by the input and we can add
            # its outputs as additional possible inputs

            if not set(input_circuit_indexes).isdisjoint(possible_input_indexes):
                possible_input_indexes.update(output_circuit_indexes)

            if gate.depth >= depth:
                # we've gone through all gates ahead of the insertion
                # depth for this new gate.
                break

            if len(possible_input_indexes) == len(self.qubits):
                # all gates are possible so we can quit checking
                break

        return sorted(possible_input_indexes)

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

        possible_output_indexes = set(self.output_indexes)

        for gate in reverse_gates:
            if not gate.enabled:
                continue

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
        quantum_registers = []
        classical_registers = []
        register_dict = {}
        for qubit_name, qubit_index in self.qubits:
            quantum_register = QuantumRegister(1, name=f"{qubit_name}-{qubit_index}")
            quantum_registers.append(quantum_register)
            register_dict[(qubit_name, qubit_index)] = quantum_register

        for qubit_name, qubit_index in self.output_qubits:
            classical_registers.append(ClassicalRegister(1))

        circuit = QuantumCircuit(*quantum_registers, *classical_registers)

        # make sure we apply the gates in the correct ordering by depth
        self.sort_gates()
        for gate in self.gates:
            gate.add_to_qiskit_circuit(register_dict, circuit)

        for output_index, input_index in enumerate(self.output_indexes):
            circuit.measure(
                quantum_registers[input_index], classical_registers[output_index]
            )

        self.circuit = circuit

        return circuit

    def _has_input_u3_layer(self) -> bool:
        """Check whether the genome already contains a trainable input U3 layer.

        The input U3 layer is identified by enabled ``"u"`` gates that contain
        the ``"input_u3_layer"`` metadata flag. These gates are inserted at the
        beginning of the circuit and act as trainable preprocessing layers
        applied after classical input encoding.

        Returns:
            bool: ``True`` if the genome already contains at least one enabled
            input U3 layer gate, otherwise ``False``.
        """
        return any(
            gate.enabled
            and gate.method_name == "u"
            and getattr(gate, "metadata", {}).get("input_u3_layer", False)
            for gate in self.gates
        )


    def add_input_u3_layer(
        self,
        base_depth: float = 1e-6,
        depth_eps: float = 1e-9,
        init_scale: float = 0.01,
    ) -> None:
        """Add a trainable U3 preprocessing layer to all input qubits.

        This method inserts one parametric ``U3`` gate per input qubit near the
        beginning of the circuit. The added gates are standard innovation-tracked
        genome gates, meaning their parameters participate naturally in mutation,
        crossover, serialization, and gradient-based optimization.

        Each inserted gate contains three trainable parameters:

        - ``theta``
        - ``phi``
        - ``delta``

        These parameters are automatically exposed through the genome parameter
        helpers using keys of the form:

            <innovation_number>:theta
            <innovation_number>:phi
            <innovation_number>:delta

        The layer is inserted only once. If the genome already contains an input
        U3 layer, the method returns immediately without modification.

        Args:
            base_depth (float, optional): Base circuit depth used for the first
                inserted U3 gate. This should be close to zero so the layer is
                applied immediately after input encoding. Defaults to ``1e-6``.

            depth_eps (float, optional): Small depth offset added between adjacent
                U3 gates to preserve deterministic ordering during sorting.
                Defaults to ``1e-9``.

            init_scale (float, optional): Initial parameter value assigned to all
                U3 gate parameters. Defaults to ``0.01``.

        Returns:
            None
        """

        if self._has_input_u3_layer():
            return

        for i, qubit in enumerate(self.input_qubits):
            self.add_gate(
                depth=base_depth + i * depth_eps,
                method_name="u",
                qubits=[qubit],
                parameters={
                    "theta": init_scale,
                    "phi": init_scale,
                    "delta": init_scale,
                },
            )

            self.gates[-1].metadata = {
                "input_u3_layer": True,
                "input_index": i,
            }

        self.sort_gates()

    def generate_pennylane_circuit(
        self,
        device_name: str = "default.qubit",
        measure_registers: bool = False,
        shots: Optional[int] = None,
        input_mode: str = "basis",
        return_probs: bool = False,
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
        self.total_qubits = len(self.qubits)

        logger.info(
            f"input indexes: {self.input_indexes}, output_indexes: {self.output_indexes}"
        )

        # Instantiate PennyLane device
        dev = qml.device(
            device_name,
            wires=self.total_qubits,
            shots=shots,
        )

        # Define the QNode function
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode_fn(
            input_bits: torch.Tensor,
            params: Dict[str, torch.Tensor],
        ):

            # --- Input preparation ---
            if input_mode == "basis":
                # expects int tensor length == total_qubits
                qml.BasisState(input_bits, wires=self.input_indexes)

            elif input_mode == "angle":
                # expects float tensor on "input" register wires
                # encode x_i in [0,1] -> RY(pi*x_i) (common, stable)
                for i, w in enumerate(self.input_indexes):
                    qml.RY(torch.pi * input_bits[i], wires=w)

            elif input_mode == "amplitude":
                # expects float tensor of length 2**len(in_wires)
                qml.AmplitudeEmbedding(
                    features=input_bits,
                    wires=self.input_indexes,
                    normalize=False,
                    pad_with=0.0,
                )

            elif input_mode == "u3":
                expected_dim = 3 * len(self.input_indexes)

                input_bits = input_bits.flatten()

                if input_bits.shape[0] != expected_dim:
                    raise ValueError(
                        f"learned_u3 encoding expects {expected_dim} values "
                        f"for {len(self.input_indexes)} input qubits, "
                        f"but got {input_bits.shape[0]}."
                    )

                # Map raw encoder outputs into valid angle range (-pi, pi).
                # angles = torch.pi * torch.tanh(input_bits)
                angles = torch.pi * input_bits

                for i, w in enumerate(self.input_indexes):
                    theta = angles[3 * i + 0]
                    phi = angles[3 * i + 1]
                    delta = angles[3 * i + 2]

                    qml.U3(theta, phi, delta, wires=w)

            else:
                raise ValueError(f"Unknown input_mode={input_mode}")

            # Apply all gates in depth order
            self.sort_gates()
            for gate in self.gates:
                gate.add_to_pennylane_circuit(self.qubits, params=params)

            # 4️⃣ Measurement
            if return_probs:
                return qml.probs(wires=self.output_indexes)
            elif measure_registers:
                # fallback if you want expvals
                expvals = [
                    qml.expval(qml.PauliZ(w))
                    for w in self.output_indexes  # self.register_map["output"]
                ]
                return expvals

            return qml.state()

        self.circuit = qnode_fn

        # 🔹 PRINT ONCE HERE
        self.sort_gates()
        for gate in self.gates:
            gate.describe_pennylane_circuit(self.qubits)

        return dev, qnode_fn

    def save_circuit(
        self,
        insert_type: str,
        out_dir: str = "artifacts/",
    ):
        """
        Saves this genome into the specified output directory in JSON and
        txt format.

        Args:
            insert_type: a tag to put at the beginning of the filename, e.g.
                'best' for global_best genomes.
            out_dir: where to write the genome files.
        """
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, f"genome_{self.genome_number}.json")
        logger.info(f"writing NEW BEST gnome to {json_path}")
        with open(json_path, "w") as fp:
            json.dump(self.to_dict(), fp, ensure_ascii=False, indent=4)

        # --- Text gate list ---
        txt_path = os.path.join(out_dir, f"genome_{self.genome_number}.txt")
        with open(txt_path, "w") as f:
            self.sort_gates()
            f.write(f"Genome {self.genome_number}\n")
            f.write(f"Qubits: {self.qubits}\n\n")
            for g in self.gates:
                if getattr(g, "enabled", True):
                    f.write(
                        f"{g.depth:.3f}  {g.method_name}  {g.qubits}  {g.parameters}\n"
                    )

        # --- PennyLane draw ---
        self.generate_pennylane_circuit(return_probs=True, input_mode="angle")
        # logger.info(f"genome circuit: {self.circuit}")

        test_metric = self.fitness.get("test_acc", None)
        if test_metric is None:
            test_metric = self.fitness.get("test_fidelity")

        try:
            tag = (
                f"trainloss_{self.fitness['train_loss']:.4f}_testloss_"
                f"{self.fitness['test_loss']:.4f}_testacc_{test_metric:.3f}"
            )
        except Exception:
            tag = (
                f"best_ep_return_{self.fitness['best_episode_return']:.4f}_"
                f"eval_return_mean_{self.fitness['eval_return_mean']:.4f}"
            )

        try:
            params = genome_to_torch_params(self)
            x0 = torch.zeros(len(self.input_indexes))
            fig, ax = qml.draw_mpl(self.circuit)(x0, params)
            ax.set_title(f"Genome {self.genome_number}")
            path = os.path.join(
                out_dir, f"{insert_type}_genome_{self.genome_number}_{tag}.png"
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not draw circuit: {e}")
