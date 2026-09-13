"""Serve EXAQC runs to an agent over MCP, on standard input and output.

This is the same read-only analysis interface the dashboard serves at ``/mcp``,
run as a standalone process so an MCP client (Claude Code, Claude Desktop) can
start it directly, with no web server involved::

    python3 -m src.examples.exaqc_mcp --runs ./artifacts/iris
    python3 -m src.examples.exaqc_mcp --directory ./2026_ppsn_exaqc --groups iris wine

Runs are given exactly as they are to ``exaqc_dashboard``: a list of run output
directories, or a directory to watch. Archives are opened read-only, and no tool
writes anything.

Logging goes to standard error, because standard output carries the protocol.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from src.utils.artifact_viewer.mcp_app import build_mcp_server
from src.utils.artifact_viewer.server import RunRegistry


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the MCP server.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description="Serve EXAQC runs to an agent over MCP (read-only) on stdio."
    )

    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Run output directories (or their genomes.sqlar files) to serve. A run whose "
            "search has not written its archive yet appears once it has."
        ),
    )
    sources.add_argument(
        "--directory",
        type=str,
        default=None,
        help=(
            "A directory to watch instead: every run below it, at any depth, is served, "
            "including runs started later."
        ),
    )

    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Substrings grouping runs for comparison: a run joins every group whose "
            "substring appears in its path (as with the analysis scripts' --groups)."
        ),
    )

    parser.add_argument(
        "--dashboard_url",
        type=str,
        default="http://127.0.0.1:8000",
        help=(
            "Base URL of the dashboard serving these runs, used for the deep links in "
            "tool results."
        ),
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="WARNING",
        help="""One of the 5 default logging levels, written to stderr. Pick DEBUG to show everything.""",
    )

    return parser


def main() -> None:
    """Serves the MCP interface on stdio until the client disconnects.

    Returns:
        None.
    """

    parser = build_parser()
    args = parser.parse_args()

    # stdout carries the MCP protocol, so every log line goes to stderr.
    logger.remove()
    logger.add(sys.stderr, level=args.logging_level)

    try:
        registry = RunRegistry(
            run_directories=args.runs,
            watch_directory=args.directory,
            groups=args.groups,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    logger.info("Serving {} run(s) over MCP on stdio.", len(registry.runs))
    build_mcp_server(registry, base_url=args.dashboard_url).run(transport="stdio")


if __name__ == "__main__":
    main()
