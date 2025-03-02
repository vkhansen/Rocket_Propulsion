"""Differential Evolution solver implementation."""
import time
import numpy as np
from scipy.optimize import differential_evolution
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class DifferentialEvolutionSolver(BaseSolver):
    """Differential Evolution solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config=None, strategy='best1bin', maxiter=1000, 
                 popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7):
        """Initialize DE solver with parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.n_stages = len(bounds)
        
    def project_to_feasible(self, x):
        """Project solution to feasible space with high precision."""
        x_proj = np.array(x, dtype=np.float64)  # Higher precision
        
        # Ensure bounds constraints
        for i in range(self.n_stages):
            lower, upper = self.bounds[i]
            x_proj[i] = np.clip(x_proj[i], lower, upper)
        
        # Normalize to total ΔV
        total = np.sum(x_proj)
        if total > 0:
            x_proj *= self.TOTAL_DELTA_V / total
            
            # Verify and adjust for exact constraint
            error = np.abs(np.sum(x_proj) - self.TOTAL_DELTA_V)
            if error > 1e-10:
                # Distribute remaining error proportionally
                adjustment = (self.TOTAL_DELTA_V - np.sum(x_proj)) / self.n_stages
                x_proj += adjustment
                
                # Final bounds check after adjustment
                for i in range(self.n_stages):
                    lower, upper = self.bounds[i]
                    x_proj[i] = np.clip(x_proj[i], lower, upper)
                
                # One final normalization if needed
                total = np.sum(x_proj)
                if np.abs(total - self.TOTAL_DELTA_V) > 1e-10:
                    x_proj *= self.TOTAL_DELTA_V / total
        
        return x_proj
        
    def custom_init(self):
        """Generate initial population that satisfies constraints."""
        population = np.zeros((self.popsize * self.n_stages, self.n_stages))
        
        for i in range(self.popsize * self.n_stages):
            # Generate random fractions that sum to 1
            fractions = np.random.random(self.n_stages)
            fractions /= np.sum(fractions)
            
            # Scale by total ΔV and project to feasible space
            population[i] = self.project_to_feasible(fractions * self.TOTAL_DELTA_V)
            
        return population
        
    def objective(self, x):
        """Objective function for DE optimization."""
        # Project solution to feasible space
        x_proj = self.project_to_feasible(x)
        
        # Check constraints
        is_feasible, violation = self.check_constraints(x_proj)
        if not is_feasible:
            return float('inf')  # Return worst possible fitness for infeasible solutions
        
        return -objective_with_penalty(  # Negative because DE minimizes
            dv=x_proj,
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
            
            # Generate feasible initial population
            init_pop = self.custom_init()
            
            result = differential_evolution(
                self.objective,
                bounds=bounds,
                strategy=self.strategy,
                maxiter=self.maxiter,
                popsize=self.popsize,
                tol=self.tol,
                mutation=self.mutation,
                recombination=self.recombination,
                init='random',  # Using random since we handle constraints in objective
                disp=False,
                workers=1,  # Required for custom objective with state
                updating='deferred',  # Better for constrained optimization
                polish=True,  # Final polishing step
                seed=42  # For reproducibility
            )
            
            execution_time = time.time() - start_time
            
            # Project and check final solution
            x_final = self.project_to_feasible(result.x)
            is_feasible, violation = self.check_constraints(x_final)
            
            if is_feasible:
                return self.process_results(
                    x=x_final,
                    success=result.success,
                    message=result.message,
                    n_iterations=result.nit if hasattr(result, 'nit') else self.maxiter,
                    n_function_evals=result.nfev if hasattr(result, 'nfev') else 0,
                    time=execution_time
                )
            else:
                return self.process_results(
                    x=initial_guess,
                    success=False,
                    message=f"Solution violates constraints (violation={violation:.2e})",
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
