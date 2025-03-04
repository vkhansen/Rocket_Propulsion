import time
import numpy as np
from scipy.optimize import basinhopping
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class BasinHoppingOptimizer(BaseSolver):
    """Basin Hopping optimization solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config, niter=100, T=1.0, stepsize=0.5, minimizer_options=None):
        """Initialize Basin Hopping optimizer with direct problem parameters and BH-specific settings.
        
        Args:
            G0: Gravitational constant
            ISP: List of specific impulse values for each stage
            EPSILON: List of structural coefficients for each stage
            TOTAL_DELTA_V: Required total delta-v
            bounds: List of (min, max) bounds for each variable
            config: Configuration dictionary
            niter: Number of basin hopping iterations
            T: Temperature parameter for BH
            stepsize: Step size for local minimizer
            minimizer_options: Dictionary of options for the local minimizer
        """
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.minimizer_options = minimizer_options if minimizer_options is not None else {}
        
        logger.debug(
            f"Initialized {self.name} with parameters: niter={niter}, T={T}, stepsize={stepsize}, \
            minimizer_options={self.minimizer_options}"
        )
        
    def generate_initial_guess(self):
        """Generate initial guess that satisfies total ΔV constraint."""
        n_vars = len(self.bounds)
        
        # Generate random fractions that sum to 1
        fractions = np.random.random(n_vars)
        fractions /= np.sum(fractions)
        
        # Scale by total ΔV
        x0 = fractions * self.TOTAL_DELTA_V
        
        # Ensure bounds constraints
        for i in range(n_vars):
            lower, upper = self.bounds[i]
            x0[i] = np.clip(x0[i], lower, upper)
        
        # Re-normalize to maintain total ΔV
        total = np.sum(x0)
        if total > 0:
            x0 *= self.TOTAL_DELTA_V / total
            
        return x0
        
    def take_step(self, x):
        """Custom step-taking function that maintains total ΔV constraint."""
        n_vars = len(self.bounds)
        
        # Take random step
        step = np.random.normal(0, self.stepsize, n_vars)
        new_x = x + step
        
        # Ensure bounds constraints
        for i in range(n_vars):
            lower, upper = self.bounds[i]
            new_x[i] = np.clip(new_x[i], lower, upper)
        
        # Project back to total ΔV constraint
        total = np.sum(new_x)
        if total > 0:
            new_x *= self.TOTAL_DELTA_V / total
            
        return new_x
        
    def objective(self, x):
        """Objective function for Basin Hopping optimization."""
        # Project solution to feasible space
        x_scaled = x.copy()
        total = np.sum(x_scaled)
        if total > 0:
            x_scaled *= self.TOTAL_DELTA_V / total
            
        return objective_with_penalty(
            dv=x_scaled,
            G0=self.G0,
            ISP=self.ISP,
            EPSILON=self.EPSILON,
            TOTAL_DELTA_V=self.TOTAL_DELTA_V,
            return_tuple=False
        )
        
    def solve(self, initial_guess, bounds, other_solver_results=None):
        """Solve using Basin Hopping optimization.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            other_solver_results: Optional dictionary of solutions from other solvers
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            logger.info("Starting Basin Hopping optimization...")
            start_time = time.time()
            
            # Generate feasible initial guess
            x0 = self.generate_initial_guess()
            
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
                    x0 = best_solution.copy()
                    logger.info(f"Using bootstrapped solution with fitness {best_fitness:.6f}")
                else:
                    logger.info("No valid solutions from other solvers, using generated initial guess")
            
            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'bounds': bounds,
                'options': {'ftol': 1e-6, 'maxiter': 100}
            }
            minimizer_kwargs.update(self.minimizer_options)
            
            result = basinhopping(
                self.objective,
                x0=x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=self.niter,
                T=self.T,
                take_step=self.take_step,
                stepsize=self.stepsize,
                disp=False
            )
            
            execution_time = time.time() - start_time
            
            # Project final solution to feasible space
            x_final = result.x
            total = np.sum(x_final)
            if total > 0:
                x_final *= self.TOTAL_DELTA_V / total
            
            return self.process_results(
                x=x_final,
                success=result.lowest_optimization_result.success,
                message=result.lowest_optimization_result.message,
                n_iterations=self.niter,
                n_function_evals=0,
                time=execution_time
            )
        except Exception as e:
            logger.error(f"Error in Basin Hopping optimizer: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
