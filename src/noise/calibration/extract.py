"""Reading per-qubit and per-coupler values out of a calibration.

These encode the rules learnt from IBM Torino's 2025-02-26 calibration and
confirmed on Fez, Kingston and Marrakesh: only gates in ``configuration().basis_gates``
count (the chip also
lists ``rzz`` with error 1.0 on every pair), an error of exactly 1.0 is IBM's
flag for an unusable gate and not a rate, and zero-duration gates such as
``rz`` are virtual frame changes with no physical error.
"""

from __future__ import annotations

UNUSABLE_ERROR = 1.0
"""IBM reports this gate error for a coupler or gate that is not calibrated."""
