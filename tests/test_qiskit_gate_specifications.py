import inspect
import pytest
import warnings

from qiskit import QuantumCircuit, QuantumRegister

from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications


@pytest.mark.parametrize("gate_method_name", qiskit_gate_specifications.keys())
def test_gate_specification(gate_method_name: str):
    '''
    This uses python reflection to ensure that the gate specfications provided
    (gate method names, qubit and parameter arguments) are correct.

    Args:
        gate_method_name: is the qiskit gate method name that can be applied
            to the QuantumCircuit object.
    '''
    print(f"testing gate: {gate_method_name}, type: {type(gate_method_name)}")

    specification = qiskit_gate_specifications[gate_method_name]

    print(f"gate specification: {specification}")

    if specification.needs_validation:
        # skip these for now
        warnings.warn(f"skipping gate {gate_method_name} ({specification.name}) that needs validation")
        return

    # get the parameter args (if any) otherwise set to an empty list
    parameter_args = specification.parameters

    # create a register large enough for the gates input and
    # output qubits
    qubit_args = specification.qubits
    n_qubits = len(qubit_args)
    register = QuantumRegister(n_qubits, name='test')

    # create a quantum circuit with that register
    circuit = QuantumCircuit(register)

    gate_method = getattr(circuit, gate_method_name)

    # determine what the required argument names are. these should be the
    # names of any non-qubit parameters (if there are any) and the qubit
    # parameter names
    signature = inspect.getfullargspec(gate_method)

    print(f"method args: {signature.args}")
    print(f"method defaults: {signature.defaults}")

    required_args = []
    if signature.defaults is not None:
        n_defaults = len(signature.defaults)
        # remove the self arg, and any defaults at the end
        required_args = signature.args[1:-n_defaults]
    else:
        required_args = signature.args[1:]

    print(f"method required args: {required_args}")
    print(f"spec qubit args: {qubit_args}")
    print(f"spec param args: {parameter_args}")

    spec_args = parameter_args + qubit_args
    print(f"spec args: {spec_args}")

    # make sure the specification arguments match the required
    # method arguments for number of qubits and other parameters
    assert (required_args == spec_args)

    print()

    
