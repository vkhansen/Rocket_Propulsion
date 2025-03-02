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
            
            # Create tasks for each solver
            tasks = []
            results = {}
            
            # Use context manager to ensure proper cleanup
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                try:
                    # Submit all tasks
                    futures = {
                        executor.submit(
                            self._run_solver,
                            solver,
                            initial_guess,
                            bounds
                        ): solver.__class__.__name__ 
                        for solver in solvers
                    }
                    
                    # Wait for completion with timeout
                    completed = concurrent.futures.wait(
                        futures.keys(),
                        timeout=self.timeout,
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    
                    # Process completed tasks
                    for future in completed.done:
                        solver_name = futures[future]
                        try:
                            result = future.result(timeout=0)  # Already completed
                            if result is not None:
                                results[solver_name] = result
                                logger.info(f"{solver_name} completed successfully")
                            else:
                                logger.warning(f"{solver_name} failed to find valid solution")
                        except Exception as e:
                            logger.error(f"Error in {solver_name}: {str(e)}")
                    
                    # Handle timeouts
                    for future in completed.not_done:
                        solver_name = futures[future]
                        logger.warning(f"{solver_name} timed out")
                        future.cancel()
                        
                except Exception as e:
                    logger.error(f"Error during parallel execution: {str(e)}")
                finally:
                    # Ensure all processes are terminated
                    executor._processes.clear()
                    
            # Log summary
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
            
            # Run solver and return results
            return solver.solve(initial_guess, bounds)
        except Exception as e:
            logger.error(f"Error in solver {solver.__class__.__name__}: {str(e)}")
            return None
