from __future__ import annotations

import argparse
from argparse import Namespace
from loguru import logger
import sys
import os

from src.utils.profiler import EXAQCProfiler


def plot(args: Namespace) -> None:
    plotter = EXAQCProfiler(out_dir=args.input_dir)
    logger.info("Loaded Profiler")

    plotter.aggregate_and_plot(
        csv_glob=os.path.join(args.input_dir, "*", "*.csv"),
        out_path=args.out_path,
        metric=args.metric,
        conf="std",
    )
    plotter.aggregate_and_plot_complexity(
        csv_glob=os.path.join(args.input_dir, "*", "*.csv"),
        out_path=os.path.join(args.out_path, "complexity_summary.png"),
        conf="std",
        title="EXAQC Gates and Parameters",
    )
    logger.info("Done Plotting")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        type=str,
        default="artifacts",
        help="Input directory to retrieve results from runs",
    )
    p.add_argument(
        "--out_path",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["std", "95ci"],
        help="How to fill confidence bounds",
    )
    p.add_argument(
        "--metric",
        type=str,
        choices=["topk", "best", "pop", "all"],
        help="How to fill confidence bounds",
    )
    p.add_argument(
        "--logging_level",
        type=str,
        required=False,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    args = p.parse_args()
    if args.metric == "all":
        args.metric = None

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    plot(args)
