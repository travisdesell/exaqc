import json
import pytest
import torch

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import initialize_decoder
from src.circuits.encoder import initialize_encoder
from src.circuits.registers import expand_registers

from tests.supervised_trainer_test_utils import build_classification_genome


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_all_disabled(target: str):
    """
    Creates a circuit genome with 3 gates which are all disabled.
    The mutation should return False.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    metadata = {
        "a": 30,
        "b": {"C": 10, "d": [1, 2, 3]},
    }

    qc = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"i1": 3, "i1": 3}),
        output_qubits=expand_registers({"o1": 3, "o2": 3}),
        target=target,
        metadata=metadata,
    )
    qc.hyperparameters = {
        "steps": 30,
        "learning_rate": 0.005,
        "log_every": 15,
        "batch_size": 12,
        "quantum_input_mode": "ry",
        "quantum_output_mode": "probs",
    }

    # create a linear encoder which also needs to serialized weights
    n_qubits = len(qc.input_indexes)
    qc.encoder = initialize_encoder(
        target=target, encoding_str="linear", n_inputs=n_qubits, n_outputs=n_qubits
    )

    # create a linear decoder which also needs to serialized weights
    qc.decoder = initialize_decoder(
        target=target, decoding_str="linear", n_inputs=2**n_qubits, n_outputs=n_qubits
    )

    # cswap is one control two target
    qc.add_gate(
        depth=0.30, method_name="cswap", qubits=[("i2", 0), ("i1", 1), ("i2", 2)]
    )
    # this does not connect an input to an output so the circuit is now valid

    # ccz is two control one target
    qc.add_gate(depth=0.40, method_name="ccz", qubits=[("i1", 2), ("o2", 1), ("o1", 2)])

    # qubit is input and output
    qc.add_gate(
        depth=0.10, method_name="p", qubits=[("i1", 1)], parameters={"theta": 0.1}
    )

    # first two are control, third is target
    qc.add_gate(depth=0.30, method_name="ccz", qubits=[("i1", 0), ("i2", 0), ("o2", 1)])

    # both are inputs and outputs
    qc.add_gate(depth=0.50, method_name="iswap", qubits=[("o2", 1), ("o2", 0)])

    # first is control, second is target
    qc.add_gate(depth=0.7, method_name="ch", qubits=[("i2", 1), ("o1", 0)])

    # first is control, second and third are target
    qc.add_gate(
        depth=0.90, method_name="cswap", qubits=[("o1", 0), ("i2", 0), ("o1", 1)]
    )

    circuit_dict = qc.to_dict()
    print("circuit_dict:")
    print(json.dumps(circuit_dict, indent=4, sort_keys=True))

    qc2 = CircuitGenome.from_dict(circuit_dict)

    assert qc.genome_number == qc2.genome_number
    assert qc.fitness == qc2.fitness
    assert qc.input_qubits == qc2.input_qubits
    assert qc.output_qubits == qc2.output_qubits
    assert qc.qubits == qc2.qubits
    assert qc.input_indexes == qc2.input_indexes
    assert qc.output_indexes == qc2.output_indexes
    assert qc.hyperparameters == qc2.hyperparameters
    assert qc.metadata == qc2.metadata

    assert len(qc.gates) == len(qc2.gates)

    for i, gate in enumerate(qc.gates):
        assert gate.depth == qc.gates[i].depth
        assert gate.method_name == qc.gates[i].method_name
        assert gate.qubits == qc.gates[i].qubits
        assert gate.parameters == qc.gates[i].parameters
        assert gate.target == qc.gates[i].target
        assert gate.specs == qc.gates[i].specs
        assert gate.enabled == qc.gates[i].enabled


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_json_round_trip_restores_tuple_qubits_and_initializes(target: str):
    """A JSON-serialized genome reloads with tuple qubits and can initialize.

    Regression test for a ``from_dict`` bug: JSON turns each ``(name, index)``
    qubit tuple into a list, and the qiskit circuit path uses qubits as dict
    keys. Before the fix, ``from_dict`` of a JSON-loaded genome left them as
    lists, so ``initialize_model()`` raised ``TypeError: unhashable type:
    'list'`` for the qiskit target. Pennylane tolerated lists but is covered
    here as well.

    Args:
        target: The circuit framework (``"qiskit"`` or ``"pennylane"``).
    """

    genome, _ = build_classification_genome(
        genome_number=3,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        include_parametric=True,
    )

    # A real JSON round-trip turns the qubit tuples into lists.
    serialized = json.loads(json.dumps(genome.to_dict()))
    assert isinstance(serialized["input_qubits"][0], list)
    assert isinstance(serialized["gates"][0]["qubits"][0], list)

    restored = CircuitGenome.from_dict(serialized)

    # from_dict must restore qubits to hashable tuples everywhere.
    assert all(isinstance(qubit, tuple) for qubit in restored.input_qubits)
    assert all(isinstance(qubit, tuple) for qubit in restored.output_qubits)
    assert all(isinstance(qubit, tuple) for qubit in restored.qubits)
    assert all(
        isinstance(qubit, tuple) for gate in restored.gates for qubit in gate.qubits
    )

    # The step that previously failed for qiskit; also exercise a forward pass.
    restored.initialize_model()
    output = restored.forward(torch.zeros(restored.encoder.n_inputs))
    assert output.shape[-1] == restored.decoder.n_outputs
