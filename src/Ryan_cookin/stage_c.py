"""Stage C placeholder (not implemented).

Stage C extends Stage B by also evolving the encoder qubit count N
itself rather than sweeping it. Requires CircuitGenome to support
register growth and shrinkage as mutation operators. Build A and B
first.
"""
from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "Stage C is not implemented. Stages A and B come first."
    )


if __name__ == "__main__":
    raise SystemExit(main())
