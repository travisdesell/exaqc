"""Keeps every program in the repo logging at INFO unless asked for more.

Runs once wrote DEBUG messages until they filled the disk: ``refine_genome``
added its ``refine.log`` sink without a level, and loguru's default for such a
sink is DEBUG. These tests pin the three things that keep that from happening
again:

* importing ``src`` replaces loguru's DEBUG default handler with an INFO one,
  so code that never configures logging still logs at INFO;
* every ``logger.add`` in ``src/`` and ``scripts/`` gives an explicit level, so
  no sink silently falls back to loguru's DEBUG default;
* no ``--logging_level`` option defaults to anything more verbose than INFO.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

#: The repository root, holding ``src/`` and ``scripts/``.
ROOT = Path(__file__).resolve().parents[1]

#: Files knowingly excluded from the repo's standards (see CLAUDE.md).
PARKED = {ROOT / "src" / "analysis" / "plot_gptp_histogram.py"}

#: Levels more verbose than INFO, which must never be a default.
TOO_VERBOSE = {"TRACE", "DEBUG"}


def python_files() -> list[Path]:
    """Lists the repo's own Python source, minus parked files.

    Returns:
        Every ``.py`` file under ``src/`` and ``scripts/`` that is not parked.
    """

    files = [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]
    return sorted(path for path in files if path not in PARKED)


def calls(path: Path) -> list[ast.Call]:
    """Parses a file and returns every call expression in it.

    Args:
        path: The Python file to parse.

    Returns:
        Every :class:`ast.Call` node in the file.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def is_method_call(call: ast.Call, owner: str, method: str) -> bool:
    """Reports whether a call is ``<owner>.<method>(...)``.

    Args:
        call: The call to inspect.
        owner: The name the method is called on, e.g. ``"logger"``.
        method: The method name, e.g. ``"add"``.

    Returns:
        True when the call has exactly that form.
    """

    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == method
        and isinstance(function.value, ast.Name)
        and function.value.id == owner
    )


def test_importing_src_logs_at_info() -> None:
    """Code that never configures logging drops DEBUG and keeps INFO."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src\n"
            "from loguru import logger\n"
            "logger.debug('debug-marker')\n"
            "logger.info('info-marker')\n",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "info-marker" in result.stderr
    assert "debug-marker" not in result.stderr


def test_configured_logging_is_left_alone() -> None:
    """A program that configured logging before importing ``src`` keeps it."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from loguru import logger\n"
            "logger.remove()\n"
            "logger.add(sys.stderr, level='DEBUG')\n"
            "import src\n"
            "logger.debug('debug-marker')\n",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stderr.count("debug-marker") == 1


@pytest.mark.parametrize(
    "path", python_files(), ids=lambda path: str(path.relative_to(ROOT))
)
def test_every_log_sink_has_an_explicit_level(path: Path) -> None:
    """``logger.add`` never relies on loguru's DEBUG default.

    Args:
        path: A Python file from ``src/`` or ``scripts/``.
    """

    unleveled = [
        call.lineno
        for call in calls(path)
        if is_method_call(call, "logger", "add")
        and not any(keyword.arg == "level" for keyword in call.keywords)
    ]

    assert not unleveled, (
        f"logger.add without level= at {path.relative_to(ROOT)} lines {unleveled}; "
        "loguru would log DEBUG to that sink"
    )


@pytest.mark.parametrize(
    "path", python_files(), ids=lambda path: str(path.relative_to(ROOT))
)
def test_no_logging_level_defaults_to_debug(path: Path) -> None:
    """No ``--logging_level`` option defaults to DEBUG or TRACE.

    Args:
        path: A Python file from ``src/`` or ``scripts/``.
    """

    for call in calls(path):
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "add_argument":
            continue
        flags = [
            argument.value
            for argument in call.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ]
        if "--logging_level" not in flags:
            continue
        for keyword in call.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                assert str(keyword.value.value).upper() not in TOO_VERBOSE, (
                    f"--logging_level defaults to {keyword.value.value} at "
                    f"{path.relative_to(ROOT)} line {call.lineno}"
                )
