# intent for classic VQC

want to minimize <psi|H|psi>

## h2 hamiltonian ( @ is the tensor product, the number in the parentheses is the qubit index)

```
 -0.09963387941370971 * I(0)
 + 0.17110545123720233 * Z(0)
 + 0.17110545123720225 * Z(1)
 + 0.16859349595532533 * (Z(0) @ Z(1))
 + 0.04533062254573469 * (Y(0) @ X(1) @ X(2) @ Y(3))
 + -0.04533062254573469 * (Y(0) @ Y(1) @ X(2) @ X(3))
 + -0.04533062254573469 * (X(0) @ X(1) @ Y(2) @ Y(3))
 + 0.04533062254573469 * (X(0) @ Y(1) @ Y(2) @ X(3))
 + -0.22250914236600539 * Z(2)
 + 0.12051027989546245 * (Z(0) @ Z(2))
 + -0.22250914236600539 * Z(3)
 + 0.16584090244119712 * (Z(0) @ Z(3))
 + 0.16584090244119712 * (Z(1) @ Z(2))
 + 0.12051027989546245 * (Z(1) @ Z(3))
 + 0.1743207725924201 * (Z(2) @ Z(3))
```

- target is min energy

- initial state is either |0000> or |1100> with the second being hartree fock state whatever that is (see wolfram alpha citation)

- qc is going to be 4 qubits, layers of roation gates and cnot entangles

- objective function is <psi|H|psi> psi is state of system after qc is applied to initial state

- differentiation method is backpropagation

- classical optimization method is gradient descent

- stopping condition is... unknown right now

compose VQC runs with increasingly more layers of rotational and CNOT gates. plot layers vs objective distance

## installs, these are required for pennlane qp.data.load

- h5py
- fsspec
- aiohttp

# intention for exacqc implementation

- want to compare how quickly it converges to the ground state compared to the layers
- should i do the same thing but allow it to change the gates? im not sure how that works

## citations

- pennylane.ai/datasets/h2-molecule
  - get the citation from here, used for hamiltonian of H2
- get wolfram alpha h2 citation later
