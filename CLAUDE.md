# EXAQC — code conventions

## Type hints and docstrings (required)

Any code you **add or modify** must be fully type-hinted and documented. When
you touch a function, method, or class, bring it up to this standard even if
the surrounding legacy code predates it.

### Type hints
- Annotate **every** parameter and the **return type** of every function and
  method — including `-> None` when nothing is returned, and nested/inner
  functions and locally-defined classes.
- Use modern typing: `X | None` (not `Optional`-in-prose or a bare `= None`
  without the `| None`), `list[...]`, `dict[str, Any]`, `tuple[...]`,
  `Iterator[...]`, `Callable[..., ...]`. Import names from `typing` /
  `collections.abc` as needed; `from __future__ import annotations` is already
  used, so annotations are lazy strings.
- Never use the builtin `any`/`list`/`dict` **as a type** in place of `Any` /
  `list[...]` / `dict[...]` (e.g. `dict[str, Any]`, never `dict(str, any)`).

### Docstrings
Use **Google-style** docstrings (matching the existing codebase) with all of
the applicable sections:
- A one-line summary sentence.
- `Args:` — one entry per parameter (omit `self`/`cls`), describing each.
- `Returns:` — what is returned; state explicitly when the function returns
  `None` and instead mutates state (say what it sets).
- `Raises:` — every exception the function raises directly, with the condition.
- Keep docstrings **accurate**: if a method sets `self.x` rather than returning
  a value, document that; don't leave stale or placeholder argument lines.

Document public and private methods alike, plus nested functions and classes
that carry real logic. Trivial one-line lambdas/closures whose behavior is
fully covered by their enclosing function's docstring may be left undocumented.

### Style / tooling
- Format with `black` and keep `flake8 --max-line-length=200` clean.
- Match the conventions of the file you are editing.

### Parked files (leave alone)

Some files are knowingly excluded from the standards above. Do not lint,
reformat, refactor, "fix" or delete them, and do not report them as findings
unless explicitly asked about them:

- `src/analysis/plot_gptp_histogram.py` — not reachable from any entry point and
  runs all of its work at import time (it has no `__main__` guard, so merely
  importing it executes everything). Parked pending a decision on its future.
  Already excluded in `.flake8` and in `[tool.black]` in `pyproject.toml`; keep
  those exclusions in sync with this list.

When auditing for dead or stale code, treat this list as expected and skip it.

## Keep the README tutorial and entry-point scripts in sync (required)

`README.md` is the project's **tutorial**. It has a linked table of contents and
is organized as: shared sections for `EXAQC` (the search), the population
strategies, and the trainers; then **one section per entry point** in
`src/examples/`; then analysis and reproduction. Every runnable entry point is
documented there, and most are also wrapped by a script in
[`scripts/`](scripts):

- **Classification:** `python3 -m src.examples.classification`, wrapped by
  `scripts/run_iris.sh`, `scripts/run_seeds.sh`, `scripts/run_breast_cancer.sh`,
  and `scripts/run_wine.sh`.
- **Reinforcement learning:** `python3 -m src.examples.reinforcement_learning`,
  wrapped by `scripts/run_cartpole.sh`, `scripts/run_frozenlake.sh`,
  `scripts/run_walker2d.sh`, and `scripts/run_mountaincar_continuous.sh`.
- **Quantum teacher imitation:** `python3 -m src.examples.teacher`, wrapped by
  `scripts/run_teacher.sh`. Unlike the other two this evolves *purely quantum*
  genomes (no encoder or decoder), so it has no `--encoding`/`--decoding`
  options and its `--teacher` / `--loss` choices come from
  `src.circuits.teacher_circuits.TEACHER_NAMES` and
  `src.metrics.teacher_losses.TEACHER_LOSS_NAMES`.
- **Genome refinement:** `python3 -m src.examples.refine_genome`, which reloads
  a single saved genome and trains it further. It has no wrapper script and
  takes no task options: every genome records the `task` and `task_target` it
  was evolved for (stamped by `EXAQC`), so changing those names, or the
  hyperparameter keys a task records, changes what refinement can reload.
- **Analysis:** `python3 -m src.analysis.analyze_genome_generation`.

Whenever an edit would change **how any of these documented entry points
operate** — not just edits to those files, but edits anywhere in the code they
reach — you must **stop and prompt the user before finalizing**, explain the
impact, and propose the concrete edits that keep the scripts and the README
example commands runnable and in sync. This includes (non-exhaustively):

- Adding, removing, renaming, or making-required any command-line argument, or
  changing an argument's default, choices, or accepted format.
- Renaming or moving an entry-point module, or a module/function it imports.
- Changing accepted values or their meaning for `--dataset`, `--env`, `--algo`,
  the population sub-commands, or the encoder/decoder/quantum-mode options.
- Changing the keys written into a genome's `fitness` dict (the analysis
  `--metric` and the README analysis commands depend on these), or the layout
  of the output directories/files the scripts and analysis read.
- Any change to sizing logic (e.g. `input_qubits`/`output_qubits` derivation,
  encoder/decoder input/output counts) that would make a currently-documented
  invocation crash or behave differently.

When you prompt, be specific: name each affected script and README command,
state what breaks or changes, and give the exact updated flags/values (and, if
a run is feasible in the environment, verify the updated command actually runs
before considering the change complete). Do not silently update the scripts to
match a behavioral change, and do not leave the scripts/README describing an
interface the code no longer supports — surface the divergence and let the user
decide how to reconcile it.

### Verify before done

After any change that touches a documented entry point, confirm it still works
before considering the task complete: at minimum run the module with `--help`
(exercising the parser and imports), and when the environment allows, do a short
live run of each affected command with a small `--number_genomes` / `--episodes`
and an out-dir under a scratch/temp location. A change is not done until an
affected invocation has been shown to run.

### README commands and wrapper scripts must match

Each documented entry point appears twice: as a standalone example command in
`README.md` and inside a `scripts/run_*.sh` wrapper. They must express the same
interface — same flags, same choice values, same defaults. When you update one,
update the other in the same change, and keep per-dataset/per-env values (qubit
counts, output qubits, crossover rates, encodings) consistent between them.

### The README tutorial must track each entry point

Every file in `src/examples/` has its own `###` section under **Entry points**
in `README.md`, containing a one-line description of what it does, a runnable
example command, and a table of its command-line arguments with defaults. When
you change an entry point, update its section **in the same change**:

- **Adding, removing or renaming an argument** — add, remove or rename its table
  row. Keep the documented default, choices and required-ness identical to the
  parser.
- **Changing a default, choice list, or how a value is derived** — update the
  table cell *and* any guidance prose that quotes it. A default that is computed
  rather than fixed (for example the reinforcement-learning `--output_qubits`,
  derived from the environment's action space) must be described by what it
  actually computes, not by a plausible-sounding rule.
- **Adding a new entry point** — add a `### <module name>` section under
  **Entry points** *and* a matching link in the table of contents.
- **Removing or renaming one** — remove or rename both its section and its TOC
  link.
- **Changing shared behavior** — the search arguments, the population strategies
  and the trainers are documented once in their own sections, not repeated per
  entry point. Update those sections instead, and check whether any per-entry
  hyperparameter guidance still holds.

Write the documentation **from the parser, not from memory.** Introspect the
real defaults rather than recalling them:

```
python3 -c "
import src.examples.<module> as m
for a in m.build_parser()._actions: print(a.dest, a.default, a.choices)"
```

`classification`, `teacher`, `reinforcement_learning`, `refine_genome` and
`classical_image_classification` expose `build_parser()` alongside a `main()`,
which is the pattern to follow for any new entry point. The rest
(`reinforcement_learning_fixed`, `evaluate`, `visualize_rl`) still build their
parser inline under `if __name__ == "__main__":`, so read the source or run the
module with `--help` instead.

### Verify the tutorial after changing it

These checks are cheap and catch the mistakes that matter. Run them after
editing `README.md`:

- **Every table-of-contents anchor resolves.** Collect the `## `/`### ` headings,
  slugify them (lowercase, punctuation stripped, spaces to hyphens) and confirm
  every `](#...)` link matches one.
- **Every documented flag exists.** Collect every `` `--flag` `` mentioned in
  `README.md` and confirm each appears in some entry point's parser (including
  sub-parsers) or in `src/analysis/`.
- **Every documented default matches the parser.** Compare the default column of
  each argument table against the parser's actual `default=`.
- **Every example command still parses.** Feed each README command's arguments to
  its `build_parser().parse_args(...)`; for the inline-parser entry points, run
  the module and confirm argparse does not reject the arguments.
- **Every external link resolves.** `curl -s -o /dev/null -w "%{http_code}" -L
  <url>` for each `](http...)` link; treat anything other than 2xx/3xx as broken
  and replace it.

### Dependency and environment sync

Run all project Python through the project venv (`~/Environments/exaqc` or other
venv location as specified by the user, if this venv is not found please prompt
the user to see if they have one present that you should use instead). If you
add an import, add the package to `pyproject.toml`. If an import error traces to
a package already declared in `pyproject.toml`, install it into the venv rather
than removing the import. Do not work around a missing declared dependency by
deleting code that uses it.

### Documented values must track the code

Choice lists and defaults quoted in `README.md` (datasets, envs, algorithms,
encoder/decoder/quantum modes, population sub-commands) must match the code's
`choices=` and module-level constants, and argparse `help=` text should agree
with the README wording. When you change a constant, default, or choice, grep
`README.md` and `scripts/` for the old value and update every occurrence. Treat
the hyperparameters in the documented commands (qubit counts, crossover rates,
encodings) as intentional experiment settings — never change them incidentally;
if a code change forces a change, flag it per the sync section above.
