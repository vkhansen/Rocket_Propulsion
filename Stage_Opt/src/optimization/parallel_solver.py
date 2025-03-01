"""Parallel solver implementation."""
import concurrent.futures
import multiprocessing
import numpy as np
import time
import psutil
import os
from ..utils.config import setup_logging, load_config

class SolverProcess(multiprocessing.Process):
    """Process class for running solvers with proper termination."""
    
    def __init__(self, solver, initial_guess, bounds):
        super().__init__()
        self.solver = solver
        self.initial_guess = np.array(initial_guess, dtype=float)  # Ensure numpy array
        self.bounds = bounds
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.daemon = True  # Ensure process is terminated when parent exits
        self.logger = setup_logging(solver.__class__.__name__)
        
    def run(self):
        try:
            solver_name = self.solver.__class__.__name__
            start_time = time.time()
            
            self.logger.info(f"Starting solver: {solver_name}")
            
            # Run solver with periodic stop checks
            while not self.stop_event.is_set():
                results = self.solver.solve(self.initial_guess, self.bounds)
                break  # Only run once, but check stop_event first
                
            if self.stop_event.is_set():
                return
                
            execution_time = time.time() - start_time
            
            # Add execution time if not present
            if isinstance(results, dict):
                if 'execution_metrics' not in results:
                    results['execution_metrics'] = {}
                results['execution_metrics']['execution_time'] = execution_time
                results['method'] = solver_name  # Ensure method name is set
                
            self.logger.info(f"Completed solver: {solver_name} in {execution_time:.2f}s")
            self.result_queue.put((solver_name, results))
            
        except Exception as e:
            if not self.stop_event.is_set():  # Only log error if not stopped
                self.logger.error(f"Error in solver {solver_name}: {str(e)}")
                self.result_queue.put((solver_name, {
                    'success': False,
                    'message': f"Error: {str(e)}",
                    'method': solver_name,
                    'payload_fraction': 0.0,
                    'constraint_violation': float('inf'),
                    'execution_metrics': {
                        'execution_time': 0.0,
                        'iterations': 0,
                        'function_evaluations': 0
                    },
                    'stages': []
                }))

def terminate_process(process):
    """Safely terminate a process and its children."""
    try:
        # First try graceful shutdown
        process.stop_event.set()
        process.join(timeout=0.5)  # Give it half a second to stop gracefully
        
        if process.is_alive():
            # Force terminate if still running
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                parent.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
            # Ensure process is terminated
            process.terminate()
            process.join(timeout=0.5)
            
            # Force kill if still alive
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
    except:
        pass

class ParallelSolver:
    """Parallel solver that runs multiple optimization algorithms."""
    
    def __init__(self, config=None):
        """Initialize parallel solver.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logging("ParallelSolver")
        
        # Load configuration
        global_config = load_config()
        opt_config = global_config.get('optimization', {})
        
        # Set timeout from config or default
        self.timeout = float(opt_config.get('parallel_solver_timeout', 30))
        
        self.logger.info(f"Initialized parallel solver with {self.timeout}s timeout")
        
    def solve(self, solvers, initial_guess, bounds):
        """Run solvers in parallel.
        
        Args:
            solvers: List of solver instances
            initial_guess: Initial solution guess
            bounds: Solution bounds
            
        Returns:
            dict: Dictionary mapping solver names to their results
        """
        self.logger.info(f"Starting parallel optimization with {len(solvers)} solvers")
        start_time = time.time()
        
        # Convert initial guess to numpy array
        initial_guess = np.array(initial_guess, dtype=float)
        
        # Create processes
        processes = []
        for solver in solvers:
            process = SolverProcess(solver, initial_guess, bounds)
            processes.append(process)
            
        # Start processes
        for process in processes:
            process.start()
            
        # Wait for results or timeout
        results = {}  # Map solver names to results
        remaining_time = self.timeout
        
        while remaining_time > 0 and any(p.is_alive() for p in processes):
            # Check for results
            for process in processes:
                try:
                    while not process.result_queue.empty():
                        solver_name, result = process.result_queue.get_nowait()
                        # Ensure result is properly formatted
                        if isinstance(result, dict):
                            result['method'] = solver_name
                            results[solver_name] = result
                except Exception as e:
                    self.logger.error(f"Error getting results: {str(e)}")
                    
            # Update remaining time
            elapsed_time = time.time() - start_time
            remaining_time = self.timeout - elapsed_time
            
            # Small sleep to prevent busy waiting
            time.sleep(0.1)
            
        # Terminate any remaining processes
        for process in processes:
            if process.is_alive():
                self.logger.warning(f"Terminating {process.solver.__class__.__name__} due to timeout")
                terminate_process(process)
                
        # Create default results for solvers that didn't complete
        total_time = time.time() - start_time
        for process in processes:
            solver_name = process.solver.__class__.__name__
            if solver_name not in results:
                results[solver_name] = {
                    'success': False,
                    'message': "Solver timed out or failed",
                    'method': solver_name,
                    'payload_fraction': 0.0,
                    'constraint_violation': float('inf'),
                    'execution_metrics': {
                        'execution_time': total_time,
                        'iterations': 0,
                        'function_evaluations': 0
                    },
                    'stages': []
                }
                
        if not results:
            self.logger.warning("No results from any solver")
            
        return results
