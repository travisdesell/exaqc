import inspect
import pytest
import warnings
import pennylane as qml
from pennylane.operation import Operation

from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications


@pytest.mark.parametrize("gate_method_name", list(pennylane_gate_specifications.keys()))
def test_gate_specification_pennylane(gate_method_name: str):
    """
    Uses Python reflection to ensure that the PennyLane gate specifications
    (qubits and parameter arguments) match the expected gate signature.

    Args:
        gate_method_name: the PennyLane gate name from the specification dict.
    """
    spec = pennylane_gate_specifications[gate_method_name]

    # Skip gates that need validation
    if getattr(spec, "needs_validation", False):
        warnings.warn(
            f"Skipping gate {gate_method_name} ({spec.name}) that needs validation"
        )
        pytest.skip(
            f"Skipping gate {gate_method_name} ({spec.name}) that needs validation"
        )

    print(f"\nTesting gate: {gate_method_name}")
    print(f"Specification: {spec}")

    # For PennyLane, all gates use 'wires' at runtime
    expected_qubits = spec.pl_qubits  # ["wires"]
    expected_params = spec.parameters or []

    # Try to get the PennyLane operation
    if not spec.pennylane_op or not hasattr(qml, spec.pennylane_op):
        warnings.warn(
            f"Gate '{gate_method_name}' not found in PennyLane. Skipping reflection check."
        )
        pytest.skip(
            f"Gate '{gate_method_name}' not found in PennyLane. Skipping reflection check."
        )

    pl_gate_cls = getattr(qml, spec.pennylane_op)

    # Get the signature (constructor for class, function for others)
    if inspect.isclass(pl_gate_cls):
        sig = inspect.getfullargspec(pl_gate_cls.__init__)
    else:
        sig = inspect.getfullargspec(pl_gate_cls)

    # Positional args (skip self)
    positional_args = sig.args[1:] if "self" in sig.args else sig.args

    # Remove args that have default values
    if sig.defaults:
        positional_args = positional_args[: len(positional_args) - len(sig.defaults)]

    # Keyword-only args that have no defaults
    kwonly_args = []
    if sig.kwonlyargs:
        if sig.kwonlydefaults:
            for arg in sig.kwonlyargs:
                if arg not in sig.kwonlydefaults:
                    kwonly_args.append(arg)
        else:
            kwonly_args = list(sig.kwonlyargs)

    # Combine required args
    required_args = positional_args + kwonly_args

    # PennyLane Operations subclass of Operation must have 'wires'
    if inspect.isclass(pl_gate_cls) and issubclass(pl_gate_cls, Operation):
        if "wires" not in required_args:
            required_args.append("wires")

    # Combine spec parameters and qubit args
    spec_args = expected_params + expected_qubits

    print(f"Required args from signature: {required_args}")
    print(f"Expected spec args: {spec_args}")

    # Assert signature matches specification
    assert required_args == spec_args, (
        f"Mismatch in gate '{gate_method_name}': "
        f"expected {spec_args}, found {required_args}"
    )
