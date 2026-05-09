"""Stage B placeholder (not implemented).

Stage B keeps the same N sweep as Stage A but lets the *structure* of
the encoder mutate, not just its angles. The encoder becomes its own
CircuitGenome whose gates are evolved alongside the downstream ansatz.
Run Stage A first; this is the next round.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "Stage B is not implemented. Run src.Ryan_cookin.stage_a first."
    )


if __name__ == "__main__":
    raise SystemExit(main())
