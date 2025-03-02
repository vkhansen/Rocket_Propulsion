import time
import numpy as np
from scipy.optimize import differential_evolution
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class DifferentialEvolutionSolver(BaseSolver):
    """Differential Evolution solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config, strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7):
        """Initialize DE solver with direct problem parameters and DE-specific settings.
        
        Args:
            G0: Gravitational constant
            ISP: List of specific impulse values for each stage
            EPSILON: List of structural coefficients for each stage
            TOTAL_DELTA_V: Required total delta-v
            bounds: List of (min, max) bounds for each variable
            config: Configuration dictionary
            strategy: DE strategy (default 'best1bin')
            maxiter: Maximum number of iterations
            popsize: Population size multiplier
            tol: Tolerance for convergence
            mutation: Mutation constant or tuple
            recombination: Recombination constant
        """
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        
        logger.debug(
            f"Initialized {self.name} with parameters: strategy={strategy}, maxiter={maxiter}, "
            f"popsize={popsize}, tol={tol}, mutation={mutation}, recombination={recombination}"
        )
        
    def custom_init(self):
        """Generate initial population that satisfies total ΔV constraint."""
        n_vars = len(self.bounds)
        population = np.zeros((self.popsize * n_vars, n_vars))
        
        for i in range(self.popsize * n_vars):
            # Generate random fractions that sum to 1
            fractions = np.random.random(n_vars)
            fractions /= np.sum(fractions)
            
            # Scale by total ΔV
            population[i] = fractions * self.TOTAL_DELTA_V
            
            # Ensure bounds constraints
            for j in range(n_vars):
                lower, upper = self.bounds[j]
                population[i,j] = np.clip(population[i,j], lower, upper)
            
            # Re-normalize to maintain total ΔV
            total = np.sum(population[i])
            if total > 0:
                population[i] *= self.TOTAL_DELTA_V / total
                
        return population
        
    def objective(self, x):
        """Objective function for DE optimization."""
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
        
    def solve(self, initial_guess, bounds):
        """Solve using Differential Evolution."""
        try:
            logger.info("Starting Differential Evolution optimization...")
            start_time = time.time()
            
            result = differential_evolution(
                self.objective,
                bounds=bounds,
                strategy=self.strategy,
                maxiter=self.maxiter,
                popsize=self.popsize,
                tol=self.tol,
                mutation=self.mutation,
                recombination=self.recombination,
                init='random',
                disp=False,
                workers=1  # Required for custom objective with state
            )
            
            execution_time = time.time() - start_time
            
            # Project final solution to feasible space
            x_final = result.x
            total = np.sum(x_final)
            if total > 0:
                x_final *= self.TOTAL_DELTA_V / total
            
            return self.process_results(
                x=x_final,
                success=result.success,
                message=result.message,
                n_iterations=result.nit if hasattr(result, 'nit') else self.maxiter,
                n_function_evals=result.nfev if hasattr(result, 'nfev') else 0,
                time=execution_time
            )
        except Exception as e:
            logger.error(f"Error in Differential Evolution solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
