"""Minimal reproduction of a qiskit-aer parameter_binds mis-binding bug.

Observed with qiskit 2.4.1 / qiskit-aer 0.17.2 (Windows and Linux, CPU
statevector method): running a parameterized circuit through
``AerSimulator.run(..., parameter_binds=[...])`` produces different (wrong)
probabilities than binding the identical values into the identical circuit
with ``assign_parameters`` first.

Two known triggers, both exact via assign_parameters:
  * a single ``cry`` gate (shown below; ~1.9e-01 probability error)
  * ``ry``/``cp`` layers interleaved twice (the exaqc champion-ansatz shape)

Not caused by transpilation (reproduces without transpile) and not fixed by
``fusion_enable=False``, so the defect is in Aer's parameterization
application, not the Python conversion in AerBackend._convert_binds.

exaqc itself is unaffected -- nothing in the repo uses parameter_binds --
but anything that adopts it for batched execution (e.g. a faster SamplerQNN
runtime) silently gets wrong results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

theta = Parameter("theta")
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cry(theta, 0, 1)

value = 1.234

reference = Statevector.from_instruction(
    qc.assign_parameters({theta: value})
).probabilities()

run_qc = qc.copy()
run_qc.save_probabilities_dict(qubits=[0, 1], label="p")
backend = AerSimulator(method="statevector")

raw = (
    backend.run(run_qc, shots=1, parameter_binds=[{theta: [value]}])
    .result()
    .data(0)["p"]
)
via_binds = np.zeros(4)
for key, v in raw.items():
    idx = int(key, 16) if isinstance(key, str) else int(key)
    via_binds[idx] = float(v)

print("reference (assign_parameters):", np.round(reference, 6))
print("via parameter_binds:          ", np.round(via_binds, 6))
print("max abs error:                ", np.abs(reference - via_binds).max())
assert np.allclose(reference, via_binds, atol=1e-9), "parameter_binds mis-binds cry"
