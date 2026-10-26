"""Logging: INFO to console, DEBUG to <run_dir>/run.log."""
import logging
import sys
from pathlib import Path

CONSOLE_FMT = "%(levelname)-8s %(message)s"
FILE_FMT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"


def setup_logging(verbose: bool = False) -> None:
    """Call once at startup, before anything else logs."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FMT))
    root.addHandler(console)
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def attach_file_handler(log_dir: Path) -> Path:
    """Start mirroring DEBUG-level logs into <run_dir>/run.log."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "run.log"
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(FILE_FMT))
    logging.getLogger().addHandler(fh)
    logging.getLogger(__name__).info("Detailed log: %s", path)
    return path