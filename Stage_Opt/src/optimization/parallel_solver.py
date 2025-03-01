"""Parallel solver implementation."""
import concurrent.futures
import multiprocessing
import numpy as np
import time
from ..utils.config import logger

class ParallelSolver:
    """Class for running multiple solvers in parallel."""
    
    def __init__(self, config):
        """Initialize parallel solver.
        
        Args:
            config: Configuration dictionary for parallel execution
        """
        self.config = config
        
        # Set number of workers based on available CPUs
        n_cpus = multiprocessing.cpu_count()
        requested_workers = self.config.get('max_workers', n_cpus)
        
        # If max_workers is None, use all available CPUs
        if requested_workers is None:
            self.max_workers = n_cpus
        else:
            self.max_workers = min(int(requested_workers), n_cpus)
        
        # Get timeout settings
        self.timeout = float(self.config.get('timeout', 3600))  # Default 1 hour
        self.solver_timeout = float(self.config.get('solver_timeout', 600))  # Default 10 minutes
        
        logger.info(f"Initialized parallel solver with {self.max_workers} workers")
        
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
            start_time = time.time()
            
            logger.info(f"Starting solver: {solver_name}")
            results = solver.solve(initial_guess, bounds)
            execution_time = time.time() - start_time
            
            # Add execution time if not present
            if isinstance(results, dict):
                if 'execution_metrics' not in results:
                    results['execution_metrics'] = {}
                results['execution_metrics']['execution_time'] = execution_time
                
            logger.info(f"Completed solver: {solver_name} in {execution_time:.2f}s")
            return solver_name, results
            
        except Exception as e:
            logger.error(f"Error in solver {solver_name}: {str(e)}")
            return solver_name, {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'constraint_violation': float('inf'),
                'execution_metrics': {
                    'execution_time': 0.0,
                    'iterations': 0,
                    'function_evaluations': 0
                },
                'stages': []
            }
            
    def solve(self, solvers, initial_guess, bounds):
        """Run all solvers in parallel.
        
        Args:
            solvers: List of solver instances
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Results from all solvers, keyed by solver name
        """
        all_results = {}
        start_time = time.time()
        
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all solver jobs
                future_to_solver = {}
                for solver in solvers:
                    # Create a copy of initial guess for each solver
                    solver_guess = np.array(initial_guess, dtype=float)
                    future = executor.submit(self._run_solver, solver, solver_guess, bounds)
                    future_to_solver[future] = solver
                
                # Collect results as they complete with timeout
                remaining_time = self.timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    raise TimeoutError("Global timeout exceeded")
                    
                # Wait for all futures to complete or timeout
                done, not_done = concurrent.futures.wait(
                    future_to_solver,
                    timeout=remaining_time,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
                
                # Process completed solvers
                for future in done:
                    try:
                        solver_name, results = future.result(timeout=1)  # Short timeout for result collection
                        all_results[solver_name] = {
                            'method': solver_name,  # Add method name for plotting
                            'success': results.get('success', False),
                            'message': results.get('message', ''),
                            'payload_fraction': float(results.get('payload_fraction', 0.0)),
                            'constraint_violation': float(results.get('constraint_violation', float('inf'))),
                            'execution_time': float(results.get('execution_metrics', {}).get('execution_time', 0.0)),
                            'n_iterations': int(results.get('execution_metrics', {}).get('iterations', 0)),
                            'n_function_evals': int(results.get('execution_metrics', {}).get('function_evaluations', 0)),
                            'stages': results.get('stages', [])
                        }
                    except Exception as e:
                        solver = future_to_solver[future]
                        solver_name = solver.__class__.__name__
                        logger.error(f"Error collecting results from {solver_name}: {str(e)}")
                        all_results[solver_name] = {
                            'method': solver_name,
                            'success': False,
                            'message': f"Error collecting results: {str(e)}",
                            'payload_fraction': 0.0,
                            'constraint_violation': float('inf'),
                            'execution_time': 0.0,
                            'n_iterations': 0,
                            'n_function_evals': 0,
                            'stages': []
                        }
                
                # Cancel any remaining solvers
                for future in not_done:
                    future.cancel()
                    solver = future_to_solver[future]
                    solver_name = solver.__class__.__name__
                    all_results[solver_name] = {
                        'method': solver_name,
                        'success': False,
                        'message': "Solver timeout",
                        'payload_fraction': 0.0,
                        'constraint_violation': float('inf'),
                        'execution_time': 0.0,
                        'n_iterations': 0,
                        'n_function_evals': 0,
                        'stages': []
                    }
                    
        except Exception as e:
            logger.error(f"Error in parallel execution: {str(e)}")
            # Ensure all solvers have entries even if parallel execution failed
            for solver in solvers:
                solver_name = solver.__class__.__name__
                if solver_name not in all_results:
                    all_results[solver_name] = {
                        'method': solver_name,
                        'success': False,
                        'message': f"Parallel execution error: {str(e)}",
                        'payload_fraction': 0.0,
                        'constraint_violation': float('inf'),
                        'execution_time': 0.0,
                        'n_iterations': 0,
                        'n_function_evals': 0,
                        'stages': []
                    }
                    
        # Find best solution considering both payload fraction and constraint violation
        best_payload = 0.0
        best_solver = None
        best_result = None
        
        for solver_name, result in all_results.items():
            if result['success']:
                payload = result['payload_fraction']
                violation = result['constraint_violation']
                
                # Only consider solutions with acceptable constraint violations
                if violation < 1e-3 and payload > best_payload:
                    best_payload = payload
                    best_solver = solver_name
                    best_result = result
                    
        total_time = time.time() - start_time
        logger.info(f"Parallel optimization completed in {total_time:.2f}s")
        if best_solver:
            logger.info(f"Best solution found by {best_solver} with payload fraction {best_payload:.4f}")
        else:
            logger.warning("No valid solution found by any solver")
                
        return all_results  # Return just the results dictionary to match expected format
