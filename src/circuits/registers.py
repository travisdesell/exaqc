def expand_registers(registers: dict[str, int]) -> list[tuple[str, int]]:
    """
    Expands a dictionary of register names and sizes into a list of qubit
    tuples (name, register_index).

    Args:
        registers: is a dictionary of register names and sizes.

    Returns:
        A list of qubit tuples (name, register_index). THe list will be the
        same length as the sum of all the sizes of the registers.
    """

    qubits = []
    for register_name, register_size in registers.items():
        for index in range(register_size):
            qubit = (register_name, index)
            qubits.append(qubit)

    return qubits
