"""Fetching IBM calibration data for the noise models and the drift study.

Everything that talks to IBM lives here, so the rest of the noise package only
ever reads snapshot files. All data comes from real IBM backends through the
saved Qiskit account. Two modules:

- :mod:`src.noise.calibration.backends`: get a backend object and its
  calibration, current or as it stood on a past date
  (``properties(datetime=...)``).
- :mod:`src.noise.calibration.drift`: the same snapshot taken on many dates,
  stored as one file per date.
"""
