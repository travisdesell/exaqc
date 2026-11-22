# Contributing to EXAQC

Thanks for your interest in EXAQC. This guide covers how to set up a
development environment, the standards code must meet, and how changes get
reviewed and merged. The [README](./README.md) is the project's tutorial and
describes what each part of the code does; this file is about how to change it.

By participating you agree to follow the [Code of Conduct](./CODE_OF_CONDUCT.md).
Contributions are released under the project's
[Creative Commons Attribution 4.0 license](./LICENSE.md).

## Table of Contents

- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)
- [Setting up a development environment](#setting-up-a-development-environment)
- [Making a change](#making-a-change)
- [Code standards](#code-standards)
  - [Type hints](#type-hints)
  - [Docstrings](#docstrings)
  - [Formatting and linting](#formatting-and-linting)
  - [Dependencies](#dependencies)
- [Tests](#tests)
- [Changes that affect entry points](#changes-that-affect-entry-points)
- [Run outputs and restarts](#run-outputs-and-restarts)
- [Pull requests](#pull-requests)

## Reporting bugs and requesting features

Open an issue on [GitHub](https://github.com/travisdesell/exaqc/issues). For a
bug, include:

- The exact command you ran (the entry point, its flags and the `mpiexec -n`
  you used).
- Your platform, Python version and MPI implementation.
- The full traceback, or what you expected to happen and what happened instead.
- For a problem with a finished or stopped run, which archive files it involves
  (`genomes.sqlar`, `best_*`) rather than the files themselves, unless asked.

## Setting up a development environment

EXAQC requires Python 3.11 or newer (CI runs on Python 3.12) and an MPI
implementation for `mpi4py`.

1. Install MPI. On macOS: `brew install openmpi`. On Debian/Ubuntu:
   `sudo apt-get install libopenmpi-dev openmpi-bin`. On Windows, use WSL.
2. Create and activate a virtual environment:

   ```
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. Install EXAQC in editable mode from the repository root. This also installs
   the development tools (`pytest`, `black`, `flake8`), which are listed as
   regular dependencies in [`pyproject.toml`](./pyproject.toml):

   ```
   python3 -m pip install -e .
   ```

4. Check the install by running the test suite (see [Tests](#tests)).

Intel (x86-64) Macs resolve to an older NumPy 1.x / Qiskit 1.x stack, because the
last PyTorch wheel for that platform does not support NumPy 2. The environment
markers in `pyproject.toml` handle this automatically; if you change a version
bound there, keep each Intel-mac / everywhere-else pair in sync.

## Making a change

1. Branch from an up-to-date `main`. Name branches `<your-name>/<short-topic>`
   (for example `travis/script_cleanup`).
2. Keep each branch to one logical change. A refactor and a behavior change are
   easier to review as separate pull requests.
3. Bring any function, method or class you touch up to the
   [code standards](#code-standards), even if the surrounding legacy code
   predates them.
4. Add or update tests for the behavior you changed.
5. Run `black`, `flake8` and `pytest` locally before opening a pull request.

## Code standards

### Type hints

Every function and method you add or modify must annotate every parameter and
its return type, including `-> None` and nested functions. Use modern typing:
`X | None`, `list[...]`, `dict[str, Any]`, `tuple[...]`, `Callable[..., ...]`
(modules already use `from __future__ import annotations`). Never use the
builtins `any`, `list` or `dict` in place of `Any`, `list[...]` or `dict[...]`.

### Docstrings

Use [Google-style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html),
matching the existing code, on public and private functions, methods and
classes alike:

- A one-line summary sentence.
- `Args:` with one entry per parameter (omit `self`/`cls`).
- `Returns:` describing the return value. When a method returns `None` and
  mutates state instead, say what it sets.
- `Raises:` listing every exception raised directly, with its condition.

```python
def scale_parameters(parameters: list[float], factor: float) -> list[float]:
    """Scales every circuit parameter by a constant factor.

    Args:
        parameters: The rotation angles to scale.
        factor: The multiplier applied to each angle.

    Returns:
        A new list with each angle multiplied by ``factor``.

    Raises:
        ValueError: If ``factor`` is not finite.
    """
```

PyCharm can generate this layout for you: Settings > Tools > Python Integrated
Tools > Docstrings > Google.

### Formatting and linting

Code is formatted with `black` and must pass `flake8` (configured in
[`.flake8`](./.flake8) with a 200-character line limit). CI runs `flake8 .` on
every pull request to `main`.

```
black --check .
flake8 .
```

`src/analysis/plot_gptp_histogram.py` is intentionally excluded from both tools.
Leave it as it is.

### Dependencies

If you add an import of a new third-party package, add the package to the
`dependencies` list in `pyproject.toml` in the same change. If an import fails
for a package that is already declared, install it into your environment rather
than removing the code that uses it.

## Tests

Tests live in [`tests/`](./tests) and run with `pytest`. CI runs the whole
suite on every push and pull request:

```
python3 -m pytest . --durations=0
```

While working, run just the files you affect, for example
`python3 -m pytest tests/test_crossover.py`.

- Name test files `test_<area>.py`. Shared helpers without tests go in modules
  without the `test_` prefix (such as `supervised_trainer_test_utils.py`).
- Tests that cover both backends should be parametrized over
  `target` in `["qiskit", "pennylane"]`, as the existing circuit tests are.
- The MPI restart tests launch real `mpiexec` processes and skip themselves,
  saying why, when no working launcher is available.
- A new MCP tool must be added to `EXPECTED_TOOLS` in
  `tests/test_exaqc_mcp.py` (or `ANNOTATION_WRITE_TOOLS`, for one that writes).
- Write any scratch output under a temporary directory (pytest's `tmp_path`),
  never inside the repository.

## Changes that affect entry points

The runnable entry points live in [`src/examples/`](./src/examples), and most
are wrapped by a script in [`scripts/`](./scripts):

| Entry point | Wrapper scripts |
|---|---|
| `python3 -m src.examples.classification` | `scripts/run_classification.sh`, `scripts/run_ppsn2026_classification.sh` |
| `python3 -m src.examples.reinforcement_learning` | `scripts/run_rl.sh`, `scripts/run_ppsn2026_rl.sh`, `scripts/submit_exaqc_rl_jobs.sh` / `scripts/exaqc_rl_job.sh` |
| `python3 -m src.examples.teacher` | none |
| `refine_genome`, `evaluate`, `visualize_rl`, `exaqc_dashboard`, `exaqc_mcp`, `backfill_archive` and the rest | none |

`README.md` documents every entry point in its own section under
[Examples](./README.md#examples), with an example command and a table of its
arguments. If your change alters how an entry point behaves (anywhere in the
code it reaches, not only in its own file), update the documentation and
scripts in the same pull request:

- Adding, removing or renaming an argument, or changing its default, choices or
  format: update its row in the README table and any README prose or script
  that uses it. Write the table from the parser, not from memory. Entry points
  that expose `build_parser()` can be introspected directly:

  ```
  python3 -c "
  import src.examples.classification as m
  for a in m.build_parser()._actions: print(a.dest, a.default, a.choices)"
  ```

- A new environment variable for `run_classification.sh` or `run_rl.sh` needs a
  row in the README's [Local run scripts](./README.md#local-run-scripts) table
  or the entry point's own Script table.
- A new entry point needs a `build_parser()` and a `main()`, its own `###`
  section in the README, and a link in the README's table of contents.
- Changing the keys of a genome's `fitness` dictionary affects the analysis
  scripts' `--metric` option and the README's analysis commands.

The `run_ppsn2026_*.sh` scripts reproduce published experiments. Keep the
commands they build unchanged, and treat the hyperparameters in documented
commands (qubit counts, crossover rates, encodings) as deliberate experiment
settings. If your change forces one of them to change, call it out in the pull
request description.

Before asking for review, run each affected entry point with `--help`, and,
where you can, a short live run with a small `--number_genomes` (or
`--episodes`) and an `--out_dir` in a temporary directory. A live run needs at
least two MPI processes (`mpiexec -n 2` or more), since rank 0 only coordinates:

```
python3 -m src.examples.classification --help
```

## Run outputs and restarts

A search's outputs (`genomes.sqlar` and the `best_*` files) are owned by
`src.utils.genome_archive.GenomeArchive`, and a stopped run is continued with
`--restart`, which rebuilds the search from what the archive recorded. Changes
here reach every search entry point, the single-genome tools, the analysis
scripts and the dashboard, so take extra care:

- Do not drop an archive field that restarting depends on.
- A new population strategy must implement `restore()`.
- A change to the archive's summary columns or a task's fitness formula must
  also be reflected in `src.utils.archive_backfill`, which brings older archives
  up to date.
- The dashboard and MCP tools only ever read archives. The one write path is
  annotations, kept in a separate `annotations.sqlite`.

## Pull requests

- Open pull requests against `main`.
- Describe what changed and why, and how you tested it (the tests you ran and
  any live runs).
- List any change to an entry point's interface or to the archive layout, and
  the README and script updates that go with it.
- CI must pass: the flake8 check and the full pytest suite on Python 3.12.
- A maintainer reviews every pull request before it is merged.
