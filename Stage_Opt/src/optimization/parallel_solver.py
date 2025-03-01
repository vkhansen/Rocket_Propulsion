"""Parallel solver implementation."""
import concurrent.futures
import multiprocessing
import numpy as np
import time
import psutil
import os
from ..utils.config import logger, load_config

class SolverProcess(multiprocessing.Process):
    """Process class for running solvers with proper termination."""
    
    def __init__(self, solver, initial_guess, bounds):
        super().__init__()
        self.solver = solver
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.daemon = True  # Ensure process is terminated when parent exits
        
    def run(self):
        try:
            solver_name = self.solver.__class__.__name__
            start_time = time.time()
            
            logger.info(f"Starting solver: {solver_name}")
            
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
                
            logger.info(f"Completed solver: {solver_name} in {execution_time:.2f}s")
            self.result_queue.put((solver_name, results))
            
        except Exception as e:
            if not self.stop_event.is_set():  # Only log error if not stopped
                logger.error(f"Error in solver {solver_name}: {str(e)}")
                self.result_queue.put((solver_name, {
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
    """Class for running multiple solvers in parallel."""
    
    def __init__(self, config):
        """Initialize parallel solver.
        
        Args:
            config: Configuration dictionary for parallel execution
        """
        self.config = config or {}  # Ensure config is not None
        
        # Load global config
        global_config = load_config() or {}
        opt_config = global_config.get('optimization', {})
        
        # Set number of workers based on available CPUs and config
        n_cpus = multiprocessing.cpu_count()
        max_processes = opt_config.get('max_processes', 4)
        config_workers = self.config.get('max_workers')
        
        # If max_workers is None, use max_processes from config
        if config_workers is None:
            self.max_workers = min(max_processes, n_cpus)
        else:
            self.max_workers = min(int(config_workers), max_processes, n_cpus)
        
        # Get timeout settings from both configs, ensure they have default values
        self.timeout = float(opt_config.get('parallel_solver_timeout') or 3600)  # Default 1 hour if None
        self.solver_timeout = float(self.config.get('solver_timeout') or 600)  # Default 10 minutes if None
        
        logger.info(f"Initialized parallel solver with {self.max_workers} workers and {self.timeout}s timeout")
            
    def _run_solver_with_timeout(self, solver, initial_guess, bounds):
        """Run a single solver instance with timeout handling.
        
        Args:
            solver: Solver instance
            initial_guess: Initial solution
            bounds: Solution bounds
            
        Returns:
            tuple: (solver_name, results)
        """
        solver_name = solver.__class__.__name__
        
        # Create and start solver process
        process = SolverProcess(solver, initial_guess, bounds)
        process.start()
        
        start_time = time.time()
        result = None
        
        try:
            # Wait for result or timeout
            while (time.time() - start_time) < self.solver_timeout:
                if not process.is_alive():
                    # Process finished, get result
                    if not process.result_queue.empty():
                        result = process.result_queue.get()
                        break
                time.sleep(0.1)
                
            if result is None:
                # Timeout occurred
                logger.warning(f"Solver timeout reached for {solver_name} - terminating process")
                terminate_process(process)
                result = (solver_name, {
                    'success': False,
                    'message': "Solver timeout - process terminated",
                    'payload_fraction': 0.0,
                    'constraint_violation': float('inf'),
                    'execution_metrics': {
                        'execution_time': self.solver_timeout,
                        'iterations': 0,
                        'function_evaluations': 0
                    },
                    'stages': []
                })
        finally:
            # Ensure process is terminated
            terminate_process(process)
            
        return result
            
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
        processes = []
        
        try:
            # Start solvers up to max_workers
            active_solvers = solvers[:self.max_workers]
            remaining_solvers = solvers[self.max_workers:]
            
            # Create and start initial processes
            for solver in active_solvers:
                solver_guess = np.array(initial_guess, dtype=float)
                process = SolverProcess(solver, solver_guess, bounds)
                process.start()
                processes.append((process, solver))
            
            # Monitor processes and handle results
            while processes and (time.time() - start_time) < self.timeout:
                # Check each process
                for process, solver in processes[:]:
                    if not process.is_alive():
                        solver_name = solver.__class__.__name__
                        try:
                            if not process.result_queue.empty():
                                name, results = process.result_queue.get()
                                all_results[name] = {
                                    'method': name,
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
                            
                        # Remove finished process
                        processes.remove((process, solver))
                        
                        # Start next solver if any remain
                        if remaining_solvers:
                            next_solver = remaining_solvers.pop(0)
                            solver_guess = np.array(initial_guess, dtype=float)
                            new_process = SolverProcess(next_solver, solver_guess, bounds)
                            new_process.start()
                            processes.append((new_process, next_solver))
                            
                time.sleep(0.1)
                
            # Timeout reached or all processes finished
            # Terminate any remaining processes
            for process, solver in processes:
                solver_name = solver.__class__.__name__
                if process.is_alive():
                    logger.warning(f"Terminating {solver_name} due to timeout")
                    terminate_process(process)
                    all_results[solver_name] = {
                        'method': solver_name,
                        'success': False,
                        'message': "Global timeout - process terminated",
                        'payload_fraction': 0.0,
                        'constraint_violation': float('inf'),
                        'execution_time': self.timeout,
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
        finally:
            # Ensure all processes are terminated
            for process, _ in processes:
                terminate_process(process)
                    
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
            
        return all_results
