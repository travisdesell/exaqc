"""Fetching IBM calibration data for the noise models and the drift study.

Everything that talks to IBM lives here, so the rest of the noise package only
ever reads snapshot files. All data comes from real IBM backends through the
user's saved account. Three modules:

- :mod:`src.noise.calibration.backends`: get a backend object and its
  calibration, current or as it stood on a past date
  (``properties(datetime=...)``).
- :mod:`src.noise.calibration.extract`: read one qubit's or one coupler's
  values out of a ``BackendProperties`` object (native gates only; an error
  of exactly 1.0 marks an uncalibrated gate; ``rz`` is virtual).
- :mod:`src.noise.calibration.drift`: the same snapshot taken on many dates,
  stored as one file per date, plus the summaries the drift study needs.
"""
