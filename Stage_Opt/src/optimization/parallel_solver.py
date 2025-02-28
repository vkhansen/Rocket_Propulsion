"""Parallel solver implementation."""
import concurrent.futures
import numpy as np
from ..utils.config import logger

class ParallelSolver:
    """Class for running multiple solvers in parallel."""
    
    def __init__(self, solvers, config):
        """Initialize parallel solver.
        
        Args:
            solvers: List of solver instances
            config: Configuration dictionary
        """
        self.solvers = solvers
        self.config = config
        self.parallel_config = config.get('optimization', {}).get('parallel', {})
        self.max_workers = self.parallel_config.get('max_workers', None)
        
    def _run_solver(self, solver, initial_guess, bounds):
        """Run a single solver instance.
        
        Args:
            solver: Solver instance
            initial_guess: Initial solution
            bounds: Solution bounds
            
        Returns:
            tuple: (solver_name, results)
        """
        try:
            solver_name = solver.__class__.__name__
            logger.info(f"Starting solver: {solver_name}")
            results = solver.solve(initial_guess, bounds)
            logger.info(f"Completed solver: {solver_name}")
            return solver_name, results
        except Exception as e:
            logger.error(f"Error in solver {solver_name}: {str(e)}")
            return solver_name, {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'stages': []
            }
            
    def solve(self, initial_guess, bounds):
        """Run all solvers in parallel.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Results from all solvers, keyed by solver name
        """
        all_results = {}
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all solver jobs
            future_to_solver = {
                executor.submit(self._run_solver, solver, initial_guess.copy(), bounds): solver
                for solver in self.solvers
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_solver):
                solver_name, results = future.result()
                all_results[solver_name] = results
                
        # Find best solution
        best_payload = 0.0
        best_solver = None
        best_result = None
        
        for solver_name, result in all_results.items():
            if result['success'] and result['payload_fraction'] > best_payload:
                best_payload = result['payload_fraction']
                best_solver = solver_name
                best_result = result
                
        return {
            'all_results': all_results,
            'best_solver': best_solver,
            'best_result': best_result
        }
