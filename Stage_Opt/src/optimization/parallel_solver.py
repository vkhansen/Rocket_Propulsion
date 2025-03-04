"""Parallel solver implementation for rocket stage optimization."""
import concurrent.futures
import time
import psutil
import signal
import numpy as np
from typing import List, Dict, Any
from ..utils.config import logger

class ParallelSolver:
    """Manages parallel execution of multiple optimization solvers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize parallel solver.
        
        Args:
            config: Configuration dictionary with keys:
                - max_workers: Maximum number of parallel workers (default: CPU count)
                - timeout: Total timeout in seconds (default: 3600)
                - solver_timeout: Per-solver timeout in seconds (default: 600)
        """
        self.config = config
        self.max_workers = config.get('max_workers', psutil.cpu_count())
        self.timeout = config.get('timeout', 3600)  # 1 hour total timeout
        self.solver_timeout = config.get('solver_timeout', 600)  # 10 minutes per solver
        self.results = {}
        self.results_lock = concurrent.futures.thread.ThreadPoolExecutor(max_workers=1).__enter__()
        
    def solve(self, solvers: List, initial_guess, bounds) -> Dict[str, Any]:
        """Run multiple solvers in parallel.
        
        Args:
            solvers: List of solver instances
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Results from all solvers that completed successfully
        """
        try:
            logger.info(f"Starting parallel optimization with {len(solvers)} solvers")
            start_time = time.time()
            
            # Separate GA solvers from other solvers
            ga_solvers = []
            other_solvers = []
            
            for solver in solvers:
                if 'GA' in solver.__class__.__name__:
                    ga_solvers.append(solver)
                else:
                    other_solvers.append(solver)
                    
            logger.info(f"Found {len(ga_solvers)} GA solvers and {len(other_solvers)} other solvers")
            
            # Run non-GA solvers first
            if other_solvers:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    try:
                        # Submit all non-GA tasks
                        future_to_solver = {
                            executor.submit(self._run_solver, solver.__class__.__name__, solver): solver.__class__.__name__
                            for solver in other_solvers
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_solver.keys(), timeout=self.timeout):
                            solver_name = future_to_solver[future]
                            try:
                                result = future.result(timeout=0)  # Non-blocking since future is done
                                if result is not None:
                                    logger.info(f"{solver_name} completed successfully")
                                else:
                                    logger.warning(f"{solver_name} failed to find valid solution")
                            except concurrent.futures.TimeoutError:
                                logger.warning(f"{solver_name} timed out")
                                future.cancel()
                            except Exception as e:
                                logger.error(f"Error in {solver_name}: {str(e)}")
                            finally:
                                # Ensure process is terminated
                                if not future.done():
                                    future.cancel()
                                    
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Global timeout reached after {self.timeout}s")
                    except Exception as e:
                        logger.error(f"Error during parallel execution: {str(e)}")
                    finally:
                        # Force shutdown of executor
                        executor.shutdown(wait=False)
            
            # Run GA solvers with results from other solvers
            if ga_solvers and self.results:
                logger.info(f"Running GA solvers with bootstrapped solutions from {len(self.results)} other solvers")
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    try:
                        # Submit all GA tasks with other solver results
                        future_to_solver = {
                            executor.submit(self._run_solver, solver.__class__.__name__, solver): solver.__class__.__name__
                            for solver in ga_solvers
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_solver.keys(), timeout=self.timeout):
                            solver_name = future_to_solver[future]
                            try:
                                result = future.result(timeout=0)  # Non-blocking since future is done
                                if result is not None:
                                    logger.info(f"{solver_name} completed successfully")
                                else:
                                    logger.warning(f"{solver_name} failed to find valid solution")
                            except concurrent.futures.TimeoutError:
                                logger.warning(f"{solver_name} timed out")
                                future.cancel()
                            except Exception as e:
                                logger.error(f"Error in {solver_name}: {str(e)}")
                            finally:
                                # Ensure process is terminated
                                if not future.done():
                                    future.cancel()
                                    
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Global timeout reached after {self.timeout}s")
                    except Exception as e:
                        logger.error(f"Error during parallel execution: {str(e)}")
                    finally:
                        # Force shutdown of executor
                        executor.shutdown(wait=False)
            elif ga_solvers:
                # Run GA solvers without other solver results
                logger.info("Running GA solvers without bootstrapped solutions")
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    try:
                        # Submit all GA tasks without other solver results
                        future_to_solver = {
                            executor.submit(self._run_solver, solver.__class__.__name__, solver): solver.__class__.__name__
                            for solver in ga_solvers
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_solver.keys(), timeout=self.timeout):
                            solver_name = future_to_solver[future]
                            try:
                                result = future.result(timeout=0)  # Non-blocking since future is done
                                if result is not None:
                                    logger.info(f"{solver_name} completed successfully")
                                else:
                                    logger.warning(f"{solver_name} failed to find valid solution")
                            except concurrent.futures.TimeoutError:
                                logger.warning(f"{solver_name} timed out")
                                future.cancel()
                            except Exception as e:
                                logger.error(f"Error in {solver_name}: {str(e)}")
                            finally:
                                # Ensure process is terminated
                                if not future.done():
                                    future.cancel()
                                    
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Global timeout reached after {self.timeout}s")
                    except Exception as e:
                        logger.error(f"Error during parallel execution: {str(e)}")
                    finally:
                        # Force shutdown of executor
                        executor.shutdown(wait=False)
                    
            # Log final summary
            elapsed = time.time() - start_time
            logger.info(f"Parallel optimization completed in {elapsed:.2f}s")
            logger.info(f"Successful solvers: {list(self.results.keys())}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error in parallel optimization: {str(e)}")
            return {}
            
    def _run_solver(self, solver_name, solver_instance):
        """Run a single solver and return the result.
        
        Args:
            solver_name: Name of the solver
            solver_instance: Instance of the solver
            
        Returns:
            Dictionary with solver results
        """
        try:
            logger.info(f"Starting solver: {solver_name}")
            
            # Get results from other solvers that have completed
            other_solver_results = self._get_other_solver_results()
            
            # Run the solver with bootstrapped results
            start_time = time.time()
            result = solver_instance.solve(initial_guess, bounds, other_solver_results=other_solver_results)
            end_time = time.time()
            
            # Calculate solve time
            solve_time = end_time - start_time
            
            # Extract solution and objective value
            solution = result.get('x', None)
            objective = result.get('fun', float('inf'))
            success = result.get('success', False)
            
            # Check if solution is valid
            is_valid = True
            if solution is None or not np.all(np.isfinite(solution)):
                is_valid = False
                logger.warning(f"Solver {solver_name} returned invalid solution: {solution}")
            
            # Prepare result dictionary
            solver_result = {
                'solver_name': solver_name,
                'solution': solution,
                'fitness': objective,
                'success': success and is_valid,
                'solve_time': solve_time,
                'raw_result': result
            }
            
            # Add to results dictionary
            with self.results_lock:
                self.results[solver_name] = solver_result
            
            logger.info(f"Solver {solver_name} completed in {solve_time:.2f} seconds with objective {objective:.6f}")
            return solver_result
            
        except Exception as e:
            logger.error(f"Error running solver {solver_name}: {str(e)}")
            # Add failed result to results dictionary
            with self.results_lock:
                self.results[solver_name] = {
                    'solver_name': solver_name,
                    'solution': None,
                    'fitness': float('inf'),
                    'success': False,
                    'solve_time': 0.0,
                    'error': str(e)
                }
            return None
            
    def _get_other_solver_results(self):
        """Get results from other solvers that have completed.
        
        Returns:
            List of solver results
        """
        other_results = []
        
        with self.results_lock:
            for solver_name, result in self.results.items():
                # Only include successful results with valid solutions
                if result.get('success', False) and result.get('solution') is not None:
                    # Create a simplified result dictionary
                    simplified_result = {
                        'solver_name': solver_name,
                        'solution': result['solution'],
                        'fitness': result.get('fitness', float('inf'))
                    }
                    other_results.append(simplified_result)
        
        return other_results
