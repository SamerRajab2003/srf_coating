"""Standalone command-line interface for single-cavity statistical analysis.

Reads raw simulation artifacts (all_samples_raw.csv and extracted_fields.npz)
from a specified run directory and executes the single-cavity reduction battery,
including Quality Factor statistics, frequency detuning, quench factors,
loaded parameters, and invariant field metrics.

Usage:
    python analyze.py <path_to_run_directory>
"""

import logging
from pathlib import Path
import sys

from src.analysis import analyze_run
from src.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Parses command-line arguments and triggers single-run analysis."""
    setup_logging()

    if len(sys.argv) != 2:
        logger.error("Usage: python analyze.py <path_to_run_directory>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        logger.error("Specified run directory does not exist: %s", run_dir)
        sys.exit(1)

    analyze_run(run_dir)


if __name__ == "__main__":
    main()
