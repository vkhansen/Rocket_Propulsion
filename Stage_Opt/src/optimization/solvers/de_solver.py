"""Differential Evolution solver implementation with enhanced constraint handling."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class DifferentialEvolutionSolver(BaseSolver):
    """Differential Evolution solver implementation with enhanced constraint handling."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config=None, strategy='best1bin', maxiter=2000, 
                 popsize=30, tol=1e-8, mutation=(0.3, 0.7), recombination=0.9):
        """Initialize DE solver with optimized parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.n_stages = len(bounds)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_violation = float('inf')
        
    def project_to_feasible(self, x):
        """Project solution to feasible space with high precision."""
        x_proj = np.array(x, dtype=np.float64)  # Higher precision
        
        # First ensure bounds constraints
        for i in range(self.n_stages):
            lower, upper = self.bounds[i]
            x_proj[i] = np.clip(x_proj[i], lower, upper)
            
        # Scale to ensure sum equals 1.0 with high precision
        total = np.sum(x_proj)
        if abs(total - 1.0) > 1e-10:  # Tighter tolerance
            x_proj /= total
            
        return x_proj

    def polish_solution(self, x, violation):
        """Polish promising solutions using L-BFGS-B."""
        if violation > 0.01:  # Only polish low-violation solutions
            return x, self.objective(x)
            
        scales = np.linspace(0.98, 1.02, 5)  # Try multiple scaling factors
        best_x = x
        best_obj = self.objective(x)
        
        for scale in scales:
            try:
                result = minimize(
                    self.objective,
                    x * scale,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'ftol': 1e-10, 'maxiter': 100}
                )
                if result.success and result.fun < best_obj:
                    best_x = result.x
                    best_obj = result.fun
            except:
                continue
                
        return best_x, best_obj

    def _objective_wrapper(self, x):
        """Wrapper for the objective function to ensure proper mapping."""
        result = self.objective(x)
        if isinstance(result, tuple):
            return float(result[0])  # Return just the objective value
        return float(result)
        
    def solve(self, initial_guess, bounds):
        """Solve using enhanced Differential Evolution."""
        try:
            logger.info("Starting Differential Evolution optimization...")
            start_time = time.time()
            
            # Generate feasible initial population
            population = np.zeros((self.popsize, self.n_stages))
            for i in range(self.popsize):
                # Start with uniform random values
                x = np.random.uniform(low=[b[0] for b in bounds], 
                                    high=[b[1] for b in bounds], 
                                    size=self.n_stages)
                # Project to feasible space
                population[i] = self.project_to_feasible(x)
            
            # Run DE with enhanced parameters
            result = differential_evolution(
                func=self._objective_wrapper,  # Use wrapper function
                bounds=bounds,
                strategy=self.strategy,
                maxiter=self.maxiter,
                popsize=self.popsize,
                tol=self.tol,
                mutation=self.mutation,
                recombination=self.recombination,
                x0=initial_guess,
                init='latinhypercube',  # Use Latin Hypercube sampling for better coverage
                updating='immediate',
                workers=1,  # Force single worker to avoid mapping issues
                disp=False,
                polish=False
            )
            
            # Polish the solution if promising
            final_x, final_obj = self.polish_solution(result.x, self.get_violation(result.x))
            
            # Track best solution
            if final_obj < result.fun:
                result.x = final_x
                result.fun = final_obj
            
            duration = time.time() - start_time
            
            # Process results
            violation = self.get_violation(result.x)
            success = violation < 1e-4 and result.fun < float('inf')  # Check both violation and objective
            
            if not success:
                message = f"Failed to find feasible solution (violation={violation:.2e})"
            else:
                message = "Optimization completed successfully"
            
            logger.info(f"DE optimization completed in {duration:.2f} seconds. Success: {success}")
            
            return {
                'x': result.x,
                'fun': result.fun if success else float('inf'),
                'success': success,
                'message': message,
                'nfev': result.nfev,
                'time': duration
            }
            
        except Exception as e:
            logger.error(f"DE optimization failed: {str(e)}")
            return {
                'x': initial_guess,
                'fun': float('inf'),
                'success': False,
                'message': str(e),
                'nfev': 0,
                'time': time.time() - start_time
            }
            
    def get_violation(self, x):
        """Calculate constraint violation."""
        total = np.sum(x)
        violation = abs(total - 1.0)
        
        # Check bound constraints
        for i, (lower, upper) in enumerate(self.bounds):
            if x[i] < lower:
                violation += abs(x[i] - lower)
            elif x[i] > upper:
                violation += abs(x[i] - upper)
                
        return violation

    def objective(self, x):
        """Objective function with enhanced constraint handling."""
        violation = self.get_violation(x)
        
        if violation > 0.1:  # Major violation
            return 100.0 * violation
        elif violation > 0:  # Minor violation
            return 10.0 * violation
            
        # Calculate payload fraction
        try:
            result = objective_with_penalty(
                x, self.G0, self.ISP, self.EPSILON,
                self.TOTAL_DELTA_V, self.bounds,
                return_tuple=False  # Ensure we get a scalar value
            )
            return result
        except:
            return float('inf')
