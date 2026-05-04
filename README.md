# Installation

To run EXAQC, first create a python3.12 virtual environment (currently the newest
version which will have the appropriate torch, qiskit and pennylane dependencies):

```
python3.12 -m venv </path/to/exaqc/environment/>
```

Then load that environment:

```
source </path/to/exaqc/environment/bin/activate/>
```

Then dependencies can be installed with (from the EXAQC project root directory):

```
python3 -m pip install -e .
```

Please note you will need to have some version of MPI installed (probably openmpi).  If you are on OSX you can install with:

```
brew install openmpi
```

Or on linux with `apt` (replace with your favorite application manager):

```
sudo apt-get install openmpi
```

# PPSN Result Reproduction

The classification benchmarks (breast cancer, iris, seeds and wine) can be run to reproduce results with the following commands:

```
mpiexec -n 8 python3 -m src.examples.pl_classification --logging_level INFO --dataset breast_cancer --number_genomes 1000 --input_qubits 8 --batch_size 3 --loss per_class -ms uniform 1 3 --out_dir ./2026_gptp_exaqc/breast_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset iris --number_genomes 1000 --input_qubits 4 --batch_size 3 --loss per_class -ms uniform 1 3 --out_dir ./2026_ppsn_exaqc/iris_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset seeds --number_genomes 1000 --input_qubits 6 --batch_size 3 --loss per_class -ms uniform 1 3 --out_dir ./2026_ppsn_exaqc/seeds_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.pl_classification --logging_level INFO --dataset wine --number_genomes 1000 --input_qubits 6 --batch_size 3 --loss per_class -ms uniform 1 3 --out_dir ./2026_ppsn_exaqc/wine_i30_per_class_1 steady_state --max_population_size 30
```

These can be run for repeated experiments using the scripts provided in the [./scripts](./scripts) directory (the following will create 10 repeats for each):

```
sh scripts/run_iris.sh 1 10 per_class ./2026_ppsn_exaqc
sh scripts/run_seeds.sh 1 10 per_class ./2026_ppsn_exaqc
sh scripts/run_breast_cancer.sh 1 10 per_class ./2026_ppsn_exaqc
sh scripts/run_wine.sh 1 10 per_class ./2026_ppsn_exaqc
```

The results of these can then be processed to generate the table of mutation and crossover rates as well as statistics on the best found genomes:

```
python3 -m src.analysis.analyze_genome_generation --input_directories ./2026_ppsn_exaqc/* --groups iris seeds wine breast_cancer --metric test_acc
```


### Docstring Formatting

We use Google format for docstrings. See https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html

> If you use an editor like `PyCharm` you can enable auto doc string comments by going to
> Settings -> Tools -> Python Integrated Tools -> Docstrings -> Select Google
