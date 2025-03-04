"""SLSQP solver implementation."""
import numpy as np
from scipy.optimize import minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty
import time

class SLSQPSolver(BaseSolver):
    """SLSQP solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config, max_iterations=100, ftol=1e-6):
        """Initialize SLSQP solver with problem parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.max_iterations = max_iterations
        self.ftol = ftol
        
    def objective(self, x):
        """Objective function for optimization."""
        x = np.asarray(x, dtype=np.float64)  # Ensure float64 precision
        return objective_with_penalty(
            dv=x,
            G0=self.G0,
            ISP=self.ISP,
            EPSILON=self.EPSILON,
            TOTAL_DELTA_V=self.TOTAL_DELTA_V,
            return_tuple=False  # Get scalar for SLSQP
        )
        
    def constraint_dv(self, x):
        """Total delta-v constraint."""
        x = np.asarray(x, dtype=np.float64)
        total_dv = np.sum(x)
        return (self.TOTAL_DELTA_V - total_dv) / self.TOTAL_DELTA_V  # Relative error
        
    def solve(self, initial_guess, bounds, other_solver_results=None):
        """Solve using SLSQP.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            other_solver_results: Optional dictionary of solutions from other solvers
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            logger.info("Starting SLSQP optimization...")
            start_time = time.time()
            
            # Convert inputs to float64
            initial_guess = np.asarray(initial_guess, dtype=np.float64)
            bounds = [(float(lb), float(ub)) for lb, ub in bounds]
            
            # If we have results from other solvers, use the best one as initial guess
            if other_solver_results is not None and len(other_solver_results) > 0:
                logger.info(f"Found {len(other_solver_results)} other solver results for bootstrapping")
                best_fitness = float('inf')
                best_solution = None
                
                for solver_name, result in other_solver_results.items():
                    if 'x' in result and np.all(np.isfinite(result['x'])) and len(result['x']) == len(bounds):
                        solution = result['x']
                        payload_fraction = result.get('payload_fraction', 0)
                        fitness = -payload_fraction if payload_fraction else float('inf')
                        
                        if fitness < best_fitness:
                            best_fitness = fitness
                            best_solution = solution
                            logger.info(f"Using solution from {solver_name} as initial guess: {solution}")
                
                if best_solution is not None:
                    initial_guess = best_solution.copy()
                    logger.info(f"Using bootstrapped solution with fitness {best_fitness:.6f}")
                else:
                    logger.info("No valid solutions from other solvers, using original initial guess")
            
            # Define constraints
            constraints = [
                {
                    'type': 'eq',
                    'fun': self.constraint_dv,
                    'jac': lambda x: np.ones_like(x) / self.TOTAL_DELTA_V  # Analytical gradient
                }
            ]
            
            # Run optimization
            result = minimize(
                self.objective,
                x0=initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.ftol,
                    'eps': 1e-8,  # Finite difference step size
                    'disp': False
                }
            )
            
            execution_time = time.time() - start_time
            
            # Check constraint satisfaction
            final_dv_error = abs(self.constraint_dv(result.x))
            feasible = final_dv_error <= 1e-4  # Feasibility threshold
            
            if not feasible:
                logger.warning(f"Solution may not satisfy constraints. DV error: {final_dv_error:.2e}")
            
            return self.process_results(
                x=result.x,
                success=result.success and feasible,
                message=result.message,
                n_iterations=result.nit,
                n_function_evals=result.nfev,
                time=execution_time,
                constraint_violation=final_dv_error
            )
            
        except Exception as e:
            logger.error(f"Error in SLSQP solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0,
                constraint_violation=float('inf')
            )
