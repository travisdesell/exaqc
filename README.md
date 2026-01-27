To run EXAQC, first create a python3.12 virtual environment (currently the newest
version which will have the appropriate torch, qiskit and pennylane dependencies):

```
python3.12 -m venv </path/to/exaqc/environment/>
```

Then dependencies can be installed with (from the EXAQC project root directory):

```
python3 -m pip install -e .
```

list of possible quantum gates and their inputs, outputs and if they have weights:

https://docs.google.com/spreadsheets/d/1Z6MjrHlESEH4S-SbZLGtgomzQ4ZeNv39t3BgJ3i0xFY/edit?gid=0#gid=0
