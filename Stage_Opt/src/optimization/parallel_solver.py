"""Parallel solver implementation for rocket stage optimization."""
import concurrent.futures
import time
import psutil
import signal
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
            results = {}
            
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
                            executor.submit(self._run_solver, solver, initial_guess, bounds): solver.__class__.__name__
                            for solver in other_solvers
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_solver.keys(), timeout=self.timeout):
                            solver_name = future_to_solver[future]
                            try:
                                result = future.result(timeout=0)  # Non-blocking since future is done
                                if result is not None:
                                    results[solver_name] = result
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
            if ga_solvers and results:
                logger.info(f"Running GA solvers with bootstrapped solutions from {len(results)} other solvers")
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    try:
                        # Submit all GA tasks with other solver results
                        future_to_solver = {
                            executor.submit(self._run_solver_with_results, solver, initial_guess, bounds, results): solver.__class__.__name__
                            for solver in ga_solvers
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_solver.keys(), timeout=self.timeout):
                            solver_name = future_to_solver[future]
                            try:
                                result = future.result(timeout=0)  # Non-blocking since future is done
                                if result is not None:
                                    results[solver_name] = result
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
                            executor.submit(self._run_solver, solver, initial_guess, bounds): solver.__class__.__name__
                            for solver in ga_solvers
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_solver.keys(), timeout=self.timeout):
                            solver_name = future_to_solver[future]
                            try:
                                result = future.result(timeout=0)  # Non-blocking since future is done
                                if result is not None:
                                    results[solver_name] = result
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
            logger.info(f"Successful solvers: {list(results.keys())}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel optimization: {str(e)}")
            return {}
            
    def _run_solver(self, solver, initial_guess, bounds):
        """Run a single solver with proper error handling.
        
        Args:
            solver: Solver instance
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Solver results if successful, None otherwise
        """
        try:
            # Set up process signal handlers
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            
            # Run solver and return results
            return solver.solve(initial_guess, bounds)
        except Exception as e:
            logger.error(f"Error in solver {solver.__class__.__name__}: {str(e)}")
            return None
            
    def _run_solver_with_results(self, solver, initial_guess, bounds, other_solver_results):
        """Run a solver with results from other solvers.
        
        Args:
            solver: Solver instance
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            other_solver_results: Dictionary of results from other solvers
            
        Returns:
            dict: Solver results if successful, None otherwise
        """
        try:
            # Set up process signal handlers
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            
            # Log detailed information about bootstrapped solutions
            logger.info(f"Bootstrapping {solver.__class__.__name__} with solutions from {len(other_solver_results)} other solvers")
            for solver_name, result in other_solver_results.items():
                if 'x' in result and result.get('success', False):
                    solution = result['x']
                    logger.info(f"  - {solver_name}: solution={solution}, fitness={-result.get('payload_fraction', 0)}")
                else:
                    logger.info(f"  - {solver_name}: No valid solution available")
            
            # Run solver with other solver results
            return solver.solve(initial_guess, bounds, other_solver_results)
        except Exception as e:
            logger.error(f"Error in solver {solver.__class__.__name__}: {str(e)}")
            return None
