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
            
            # Categorize solvers by type for proper execution order
            ga_solvers = []
            de_solvers = []
            pso_solvers = []
            other_solvers = []
            
            for solver in solvers:
                solver_name = solver.__class__.__name__
                if 'GA' in solver_name or 'Genetic' in solver_name:
                    ga_solvers.append(solver)
                elif 'Differential' in solver_name or 'DE' in solver_name:
                    de_solvers.append(solver)
                elif 'PSO' in solver_name or 'Particle' in solver_name:
                    pso_solvers.append(solver)
                else:
                    other_solvers.append(solver)
                    
            logger.info(f"Found {len(ga_solvers)} GA solvers, {len(de_solvers)} DE solvers, "
                      f"{len(pso_solvers)} PSO solvers, and {len(other_solvers)} other solvers")
            
            # Run non-population-based solvers first
            for solver in other_solvers:
                solver_name = solver.__class__.__name__
                try:
                    logger.info(f"Running {solver_name}...")
                    result = solver.solve(initial_guess, bounds)
                    if result and result.get('success', False):
                        # Extract solution from stages if available
                        solution = None
                        if 'stages' in result and result['stages']:
                            solution = np.array([stage['delta_v'] for stage in result['stages']])
                        
                        results[solver_name] = {
                            'solver_name': solver_name,
                            'solution': solution,
                            'fitness': result.get('payload_fraction', 0.0),
                            'success': True,
                            'payload_fraction': result.get('payload_fraction', 0.0),
                            'constraint_violation': result.get('constraint_violation', 0.0),
                            'message': result.get('message', ''),
                            'execution_metrics': result.get('execution_metrics', {}),
                            'stages': result.get('stages', []),
                            'raw_result': result
                        }
                        logger.info(f"{solver_name} completed successfully")
                    else:
                        logger.warning(f"{solver_name} failed to find valid solution")
                except Exception as e:
                    logger.error(f"Error in {solver_name}: {str(e)}")
            
            # Prepare bootstrap solutions for population-based methods
            other_solver_results = []
            for solver_name, result in results.items():
                if result.get('success', False) and result.get('solution') is not None:
                    other_solver_results.append({
                        'solver_name': solver_name,
                        'solution': result['solution'],
                        'fitness': result.get('fitness', float('inf'))
                    })
            
            # Run all population-based solvers with bootstrapped solutions
            for solver_group in [ga_solvers, de_solvers, pso_solvers]:
                for solver in solver_group:
                    solver_name = solver.__class__.__name__
                    try:
                        logger.info(f"Running {solver_name} with {len(other_solver_results)} bootstrapped solutions...")
                        result = solver.solve(initial_guess, bounds, other_solver_results=other_solver_results)
                        
                        # Check if the solution is valid - either success flag is True or there are valid stages
                        is_valid = result.get('success', False) or (
                            'stages' in result and result['stages'] and len(result['stages']) > 0
                        )
                        
                        if result and is_valid:
                            # Extract solution from stages if available
                            solution = None
                            if 'stages' in result and result['stages']:
                                solution = np.array([stage['delta_v'] for stage in result['stages']])
                            
                            results[solver_name] = {
                                'solver_name': solver_name,
                                'solution': solution,
                                'fitness': result.get('payload_fraction', 0.0),
                                'success': True,  # Mark as successful if we have valid stages
                                'payload_fraction': result.get('payload_fraction', 0.0),
                                'constraint_violation': result.get('constraint_violation', 0.0),
                                'message': result.get('message', ''),
                                'execution_metrics': result.get('execution_metrics', {}),
                                'stages': result.get('stages', []),
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
            
            # Get list of successful solvers
            successful_solvers = [name for name, result in results.items() if result.get('success', False)]
            logger.info(f"Successful solvers: {successful_solvers}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return {}
