"""
Standalone script to build a TESLA cavity and configure the CST Eigenmode solver.
Usage: python setup_cavity.py
"""
import sys
import logging
from src.logging_setup import setup_logging
from src.cavity_setup import run_cavity_setup

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        run_cavity_setup()
    except Exception as e:
        logger.error("Setup failed: %s", e)
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()