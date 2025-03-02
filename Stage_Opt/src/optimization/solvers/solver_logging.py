"""Logging utilities for optimization solvers."""
import logging
import os
from typing import Optional

def setup_solver_logger(solver_name: str, log_dir: str = "logs") -> logging.Logger:
    """Set up a dedicated logger for a specific solver.
    
    Args:
        solver_name: Name of the solver (e.g., 'PSO', 'DE')
        log_dir: Directory to store log files
        
    Returns:
        Logger instance configured for the solver
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"solver.{solver_name}")
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicate logging
    logger.handlers = []
    
    # Create file handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{solver_name.lower()}_solver.log"), mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(fh)
    
    return logger
