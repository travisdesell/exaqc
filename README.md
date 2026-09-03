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

The classification benchmarks (breast cancer, iris, seeds and wine) can be run to reproduce results with the following commands. They use amplitude encoding for the quantum inputs, the `probs` output mode with a `clipped` decoder, and an identity encoder:

```
mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset breast_cancer --number_genomes 1000 --input_qubits 8 --output_qubits 1 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/breast_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset iris --number_genomes 1000 --input_qubits 4 --output_qubits 2 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/iris_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset seeds --number_genomes 1000 --input_qubits 6 --output_qubits 2 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/seeds_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset wine --number_genomes 1000 --input_qubits 4 --output_qubits 2 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/wine_i30_per_class_1 steady_state --max_population_size 30
```

These can be run for repeated experiments using the scripts provided in the [./scripts](./scripts) directory (the following will create 10 repeats for each):

```
sh scripts/run_iris.sh 1 10 per_class ./2026_ppsn_exaqc/classification
sh scripts/run_seeds.sh 1 10 per_class ./2026_ppsn_exaqc/classification
sh scripts/run_breast_cancer.sh 1 10 per_class ./2026_ppsn_exaqc/classification
sh scripts/run_wine.sh 1 10 per_class ./2026_ppsn_exaqc/classification
```

And reinforcement learning experiments can be run with:
```
sh scripts/run_cartpole.sh 1 10 per_class ./2026_ppsn_exaqc/rl
sh scripts/run_frozenlake.sh 1 10 per_class ./2026_ppsn_exaqc/rl
sh scripts/run_walker2d.sh 1 10 per_class ./2026_ppsn_exaqc/rl
sh scripts/run_mountaincar_continuous.sh 1 10 per_class ./2026_ppsn_exaqc/rl
```

# Quantum Teacher Imitation

Circuits can also be evolved to reproduce the outputs of a known reference
("teacher") circuit. The teacher is itself a circuit genome, so it runs on either
the `pennylane` or `qiskit` backend, and the evolved circuits carry **no encoder
and no decoder** -- there is nothing classical to learn, so the search is over
the circuit alone (hence no `--encoding`/`--decoding` options). The dataset is
generated: random input angles are drawn and the teacher's outputs for them
become the targets.

```
mpiexec -n 12 python3 -m src.examples.teacher --logging_level INFO --teacher half_adder --input_qubits 2 --output_qubits 2 --number_genomes 1000 --loss fidelity --epochs 30 --batch_size 8 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim ry -qom probs --out_dir ./2026_ppsn_exaqc/teacher_half_adder_1 steady_state --max_population_size 30
```

The input wires are the first `--input_qubits` wires and the readout wires are
the `--output_qubits` wires after them, so a teacher spans
`--input_qubits + --output_qubits` wires and the two sets never overlap. The
available teachers are `identity`, `x_out4`, `bell_out`, `copy_in_to_out`,
`parity012_to_out4`, `input_controlled_bell`, `2layer_out_block`, `grover` and
`half_adder`; each reports clearly if the requested wires cannot express it.

`--loss` selects the optimized measure from `fidelity`, `angle`, `kl` and `mse`.
The first three compare probability distributions and therefore require
`-qom probs`; `mse` also works with `-qom expval`. All four are reported every
epoch regardless of which one is optimized.

Repeated experiments can be run with:

```
sh scripts/run_teacher.sh 1 10 half_adder ./2026_ppsn_exaqc/teacher
```

# Refining a Single Genome

Every genome the search evaluates is written to JSON, and each one records the
`task` it was evolved for (`classification`, `teacher` or
`reinforcement_learning`) along with the `task_target` it ran against (the
dataset, teacher circuit or environment). A saved genome is therefore
self-describing and can be reloaded and trained further -- useful for giving the
best genome of a search a longer run than the search itself could afford:

```
python3 -m src.examples.refine_genome --genome ./2026_ppsn_exaqc/teacher/all_genomes/genome_11.json
```

The hyperparameters stored in the file are reused unchanged. Any of them can be
overridden with `--set`, which may be repeated:

```
python3 -m src.examples.refine_genome --genome best_genome.json --out_dir ./refined --set epochs=200 --set learning_rate=0.01
```

The refined genome is written back out as JSON (still self-describing, so it can
be refined again), alongside its architecture diagram and, with
`--save_training_plot`, its training history.

The results of these can then be processed to generate the table of mutation and crossover rates as well as statistics on the best found genomes:

```
python3 -m src.analysis.analyze_genome_generation --input_directories ./2026_ppsn_exaqc/classification/* --groups iris seeds wine breast_cancer --metric target_metric
python3 -m src.analysis.analyze_genome_generation --input_directories ./2026_ppsn_exaqc/rl/* --groups cartpole frozenlake walker2d mountaincar_continuous --metric target_metric
```


### Docstring Formatting

We use Google format for docstrings. See https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html

> If you use an editor like `PyCharm` you can enable auto doc string comments by going to
> Settings -> Tools -> Python Integrated Tools -> Docstrings -> Select Google
