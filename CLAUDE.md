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
