"""A web dashboard for EXAQC runs -- genomes, ancestry, insertion rates and search progress.

Starts a local web server over runs' ``genomes.sqlar`` archives and serves a
dashboard for them. Give the runs as run output directories, or give a
directory to watch, where every run below it is shown -- including runs started
after the dashboard. Runs can be browsed while their searches are still running,
and a genome's architecture diagram and training plot are drawn on demand, so
the search itself never has to draw them::

    python3 -m src.examples.exaqc_dashboard --runs ./artifacts/iris
    python3 -m src.examples.exaqc_dashboard --directory ./2026_ppsn_exaqc/classification --groups iris seeds wine

Then open the printed address (http://127.0.0.1:8000/ by default). To look at
runs on a remote machine, start the dashboard there and forward its port, e.g.
``ssh -L 8000:localhost:8000 cluster``.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from src.utils.artifact_viewer.server import serve


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the EXAQC dashboard.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description="A dashboard for EXAQC runs -- their genomes, ancestry, insertion rates and search progress -- in a local web page."
    )

    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Run output directories (or their genomes.sqlar files) to show. A run whose search has "
            "not written its archive yet (or not even created its directory) appears once it has."
        ),
    )
    sources.add_argument(
        "--directory",
        type=str,
        default=None,
        help=(
            "A directory to watch instead: every run below it, at any depth, is shown, including "
            "runs started while the dashboard is running."
        ),
    )

    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Substrings grouping runs for comparison: a run joins every group whose substring "
            "appears in its path (as with the analysis scripts' --groups)."
        ),
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Address to serve on. The default only accepts connections from this machine; view "
            "remote runs by forwarding the port over SSH."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (0 picks a free port).",
    )

    parser.add_argument(
        "--open_browser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the dashboard in a web browser once it is running.",
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    return parser


def main() -> None:
    """Starts the EXAQC dashboard and serves it until interrupted.

    Returns:
        None.
    """

    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    try:
        serve(
            runs=args.runs,
            directory=args.directory,
            groups=args.groups,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
