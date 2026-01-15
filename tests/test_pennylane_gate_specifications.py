import inspect
import pytest
import warnings
import pennylane as qml

from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications


@pytest.mark.parametrize("gate_method_name", pennylane_gate_specifications.keys())
def test_gate_specification_pennylane(gate_method_name: str):
    """
    Uses Python reflection to ensure that the PennyLane gate specifications
    (gate method names, qubit and parameter arguments) are correct.

    Args:
        gate_method_name: the PennyLane gate name from the specification dict.
    """
    print(f"testing gate: {gate_method_name}, type: {type(gate_method_name)}")

    specification = pennylane_gate_specifications[gate_method_name]
    print(f"gate specification: {specification}")

    if getattr(specification, "needs_validation", False):
        warnings.warn(
            f"skipping gate {gate_method_name} ({specification.name}) that needs validation"
        )
        return

    # get the parameter args (if any) otherwise set to an empty list
    parameter_args = specification.parameters or []

    # get the qubit/wire args
    qubit_args = specification.qubits or []

    # try to get the PennyLane gate
    if not hasattr(qml, gate_method_name):
        warnings.warn(f"Gate '{gate_method_name}' not found natively in PennyLane. Skipping reflection check.")
        return

    gate_cls = getattr(qml, gate_method_name)

    # determine the signature (use __init__ if class, otherwise the function)
    if inspect.isclass(gate_cls):
        sig = inspect.getfullargspec(gate_cls.__init__)
    else:
        sig = inspect.getfullargspec(gate_cls)

    print(f"method args: {sig.args}")
    print(f"method defaults: {sig.defaults}")

    # remove 'self' if present and any defaults at the end
    if sig.defaults:
        n_defaults = len(sig.defaults)
        required_args = sig.args[1:-n_defaults]  # skip self, remove defaults
    else:
        required_args = sig.args[1:] if 'self' in sig.args else sig.args

    print(f"method required args: {required_args}")
    print(f"spec qubit args: {qubit_args}")
    print(f"spec param args: {parameter_args}")

    # combine spec parameter args and qubit args
    spec_args = parameter_args + qubit_args
    print(f"spec args: {spec_args}")

    # ensure specification matches the gate's required arguments
    assert required_args == spec_args

    print()
