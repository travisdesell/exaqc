To run EXAQC, first create a python3.12 virtual environment (currently the newest
version which will have the appropriate torch, qiskit and pennylane dependencies):

```
python3.12 -m venv </path/to/exaqc/environment/>
```

Then dependencies can be installed with (from the EXAQC project root directory):

```
python3 -m pip install -e .
```

Example classification tasks can be run using MPI, e.g.:

```
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset iris --number_genomes 500 --input_qubits 4 --out_qubits 2 --out_dir ./iris_results 
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset wine --number_genomes 500 --input_qubits 6 --out_qubits 2 --out_dir ./wine_results 
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset seeds --number_genomes 500 --input_qubits 7 --out_qubits 2 --out_dir ./breast_cancer_results 
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 500 --input_qubits 8 --out_qubits 1 --out_dir ./breast_cancer_results 
```

Example teacher circuits can be run using MPI, e.g.:

```
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset iris --number_genomes 500 --input_qubits 4 --out_qubits 2 --out_dir ./iris_results 
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset wine --number_genomes 500 --input_qubits 6 --out_qubits 2 --out_dir ./wine_results 
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset seeds --number_genomes 500 --input_qubits 7 --out_qubits 2 --out_dir ./breast_cancer_results 
mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 500 --input_qubits 8 --out_qubits 1 --out_dir ./breast_cancer_results 
```



