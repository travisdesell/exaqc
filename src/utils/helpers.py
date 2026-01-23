def register_wire_map(registers: dict[str, int]) -> dict:
    """Return a dict mapping register names to PennyLane wires."""
    wire_map = {}
    offset = 0
    for name, size in registers.items():
        wire_map[name] = list(range(offset, offset + size))
        offset += size
    return wire_map
