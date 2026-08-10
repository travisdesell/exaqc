"""A drop-in replacement for qiskit-machine-learning's reference QMLSampler
that computes the same exact probabilities with qiskit-aer's C++ statevector
simulator, executing all parameter bindings of a PUB in a single batched job.

The reference sampler evolves every bound circuit gate-by-gate in pure Python
(`Statevector.from_instruction`), which dominates SamplerQNN training time.
Everything else (ExactProbArray packaging, gradient bookkeeping) is inherited
unchanged, so results stay bit-for-bit comparable up to float precision.
"""

from __future__ import annotations

import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_machine_learning.primitives.sampler import (
    ExactProbArray,
    ExactProbNDArray,
    QMLSampler,
    _ExactSamplerPubResult,
    _preprocess_circuit,
)
from qiskit.primitives import DataBin
from qiskit.primitives.containers.sampler_pub import SamplerPub


class AerExactSampler(QMLSampler):
    """QMLSampler with the per-circuit Python statevector loop replaced by a
    single batched AerSimulator job using parameter_binds."""

    def __init__(self, *, shots: int | None = None, **kwargs):
        super().__init__(shots=shots, **kwargs)
        self._backend = AerSimulator(method="statevector")
        # cache: id(pub.circuit) -> (transpiled circuit w/ save instr, qargs, meas_info)
        self._circuit_cache: dict[int, tuple] = {}

    def _prepare(self, circuit):
        key = id(circuit)
        cached = self._circuit_cache.get(key)
        if cached is not None:
            return cached

        unitary_circ, qargs, meas_info = _preprocess_circuit(circuit)
        run_circ = unitary_circ.copy()
        if qargs:
            run_circ.save_probabilities_dict(qubits=qargs, label="joint")
        transpiled = transpile(run_circ, self._backend, optimization_level=0)
        prepared = (transpiled, qargs, meas_info)
        self._circuit_cache[key] = prepared
        return prepared

    def _run_pub_exact(self, pub: SamplerPub) -> _ExactSamplerPubResult:
        transpiled, qargs, meas_info = self._prepare(pub.circuit)

        values = pub.parameter_values
        shape = values.shape
        n_bindings = int(np.prod(shape)) if shape else 1
        params = list(pub.circuit.parameters)

        if qargs:
            if params and n_bindings:
                # NOTE: do NOT use Aer's parameter_binds here -- it matches
                # parameters by lexicographically sorted name, so vectors with
                # >=10 elements bind wrong (weights[10] sorts before
                # weights[2]). Explicit assign_parameters is exact, and all
                # bound circuits still run as a single batched Aer job.
                flat = values.as_array(params).reshape(n_bindings, len(params))
                circuits = [
                    transpiled.assign_parameters(dict(zip(params, row)))
                    for row in flat
                ]
                job = self._backend.run(circuits, shots=1)
            else:
                job = self._backend.run(transpiled, shots=1)
            result = job.result()

            n_qargs = len(qargs)
            joint_probs = []
            for i in range(n_bindings):
                raw = result.data(i)["joint"]
                # Aer keys the dict by int (or hex string) over the saved
                # qubits; the reference sampler keys are bitstrings.
                joint = {}
                for k, v in raw.items():
                    if isinstance(k, str):
                        k = int(k, 16) if k.startswith("0x") else int(k, 2)
                    joint[format(int(k), f"0{n_qargs}b")] = float(v)
                joint_probs.append(joint)
        else:
            joint_probs = [{"": 1.0} for _ in range(n_bindings)]

        joint_probs_per_index = np.empty(shape, dtype=object)
        for i, index in enumerate(np.ndindex(shape if shape else ())):
            joint_probs_per_index[index] = joint_probs[i]

        data_fields = {}
        names = []
        for item in meas_info:
            names.append(item.creg_name)
            arr = np.empty(shape, dtype=object)
            for index, joint in np.ndenumerate(joint_probs_per_index):
                arr[index] = ExactProbArray(
                    joint_probs=joint,
                    mask=list(item.qreg_indices),
                    num_bits=item.num_bits,
                    shape=(),
                )
            data_fields[item.creg_name] = (
                arr.item() if arr.shape == () else ExactProbNDArray(arr)
            )

        data_bin = DataBin(**data_fields, shape=shape)
        return _ExactSamplerPubResult(
            data_bin,
            metadata={
                "shots": None,
                "exact": True,
                "names": names,
                "circuit_metadata": getattr(pub, "metadata", {}),
            },
        )
