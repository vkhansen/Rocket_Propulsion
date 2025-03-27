"""Logging utilities for optimization solvers."""
import logging
import os
import sys
from typing import Optional

def setup_solver_logger(solver_name: str, log_dir: str = "logs") -> logging.Logger:
    """Set up a dedicated logger for a specific solver with console output only.
    
    Args:
        solver_name: Name of the solver (e.g., 'PSO', 'DE')
        log_dir: Directory to store log files (not used with console-only logging)
        
    Returns:
        Logger instance configured for the solver
    """
    # Create logger
    logger = logging.getLogger(f"solver.{solver_name}")
    logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO for performance
    
    # Remove any existing handlers to avoid duplicate logging
    logger.handlers = []
    
    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Create formatter - simplified for console output
    formatter = logging.Formatter(f'[{solver_name}] %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    return logger
