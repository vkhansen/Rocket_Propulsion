"""Parallel solver implementation for rocket stage optimization."""
import time
import logging
import numpy as np
from typing import List, Dict, Any
from ..utils.config import logger

class ParallelSolver:
    """Manages execution of multiple optimization solvers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize parallel solver with configuration."""
        self.config = config
        self.max_workers = config.get('max_workers', 1)
        self.timeout = config.get('timeout', 3600)  # 1 hour total timeout
        self.solver_timeout = config.get('solver_timeout', 600)  # 10 minutes per solver
        
    def solve(self, solvers: List, initial_guess, bounds) -> Dict[str, Any]:
        """Run multiple solvers sequentially (temporarily avoiding parallel execution).
        
        Args:
            solvers: List of solver instances
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            Dictionary with results from all solvers
        """
        try:
            logger.info(f"Starting optimization with {len(solvers)} solvers")
            start_time = time.time()
            results = {}
            
            # Separate GA solvers from other solvers
            ga_solvers = []
            other_solvers = []
            
            for solver in solvers:
                if 'GA' in solver.__class__.__name__ or 'Genetic' in solver.__class__.__name__:
                    ga_solvers.append(solver)
                else:
                    other_solvers.append(solver)
                    
            logger.info(f"Found {len(ga_solvers)} GA solvers and {len(other_solvers)} other solvers")
            
            # Run non-GA solvers first
            for solver in other_solvers:
                solver_name = solver.__class__.__name__
                try:
                    logger.info(f"Running {solver_name}...")
                    result = solver.solve(initial_guess, bounds)
                    if result and 'x' in result and result.get('success', False):
                        results[solver_name] = {
                            'solver_name': solver_name,
                            'solution': result['x'],
                            'fitness': result['fun'],
                            'success': True,
                            'raw_result': result
                        }
                        logger.info(f"{solver_name} completed successfully")
                    else:
                        logger.warning(f"{solver_name} failed to find valid solution")
                except Exception as e:
                    logger.error(f"Error in {solver_name}: {str(e)}")
            
            # Run GA solvers with results from other solvers
            other_solver_results = []
            for solver_name, result in results.items():
                if result.get('success', False) and result.get('solution') is not None:
                    other_solver_results.append({
                        'solver_name': solver_name,
                        'solution': result['solution'],
                        'fitness': result.get('fitness', float('inf'))
                    })
            
            for solver in ga_solvers:
                solver_name = solver.__class__.__name__
                try:
                    logger.info(f"Running {solver_name} with {len(other_solver_results)} bootstrapped solutions...")
                    result = solver.solve(initial_guess, bounds, other_solver_results=other_solver_results)
                    if result and 'x' in result and result.get('success', False):
                        results[solver_name] = {
                            'solver_name': solver_name,
                            'solution': result['x'],
                            'fitness': result['fun'],
                            'success': True,
                            'raw_result': result
                        }
                        logger.info(f"{solver_name} completed successfully")
                    else:
                        logger.warning(f"{solver_name} failed to find valid solution")
                except Exception as e:
                    logger.error(f"Error in {solver_name}: {str(e)}")
            
            # Log final summary
            elapsed = time.time() - start_time
            logger.info(f"Optimization completed in {elapsed:.2f}s")
            logger.info(f"Successful solvers: {list(results.keys())}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return {}
