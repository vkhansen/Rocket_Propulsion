"""Base solver class for optimization."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np
import time

from ...utils.config import logger
from ..cache import OptimizationCache
from ..physics import calculate_stage_ratios, calculate_payload_fraction
from ..objective import objective_with_penalty

class BaseSolver(ABC):
    """Base class for all optimization solvers."""
    
    def __init__(self, G0: float, ISP: List[float], EPSILON: List[float], 
                 TOTAL_DELTA_V: float, bounds: List[Tuple[float, float]], config: Dict):
        """Initialize solver with problem parameters.
        
        Args:
            G0: Gravitational constant
            ISP: List of specific impulse values for each stage
            EPSILON: List of structural coefficients for each stage
            TOTAL_DELTA_V: Required total delta-v
            bounds: List of (min, max) bounds for each variable
            config: Configuration dictionary
        """
        self.G0 = float(G0)
        self.ISP = np.array(ISP, dtype=np.float64)
        self.EPSILON = np.array(EPSILON, dtype=np.float64)
        self.TOTAL_DELTA_V = float(TOTAL_DELTA_V)
        self.bounds = bounds
        self.config = config
        self.n_stages = len(bounds)
        self.name = self.__class__.__name__
        
        # Common solver parameters
        self.population_size = 150
        self.max_iterations = 300
        self.precision_threshold = 1e-6
        self.feasibility_threshold = 1e-6
        self.max_projection_iterations = 20
        self.stall_limit = 30
        
        # Statistics tracking
        self.n_feasible = 0
        self.n_infeasible = 0
        self.best_feasible = None
        self.best_feasible_score = float('inf')
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_is_feasible = False
        self.best_violation = float('inf')
        
        logger.debug(f"Initialized {self.name} with {self.n_stages} stages")
        
        # Initialize cache
        self.cache = OptimizationCache()
        
    def evaluate_solution(self, x: np.ndarray) -> float:
        """Evaluate a solution vector.
        
        Args:
            x: Solution vector (delta-v values)
            
        Returns:
            float: Objective value with penalties
        """
        try:
            # Ensure x is a 1D array
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Check cache first
            cached = self.cache.get(tuple(x))
            if cached is not None:
                return cached
            
            # Calculate objective with penalties
            score = objective_with_penalty(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON,
                TOTAL_DELTA_V=self.TOTAL_DELTA_V
            )
            
            # Cache result
            self.cache.add(tuple(x), score)
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating solution: {str(e)}")
            return float('inf')
            
    def check_feasibility(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check if solution is feasible.
        
        Args:
            x: Solution vector
            
        Returns:
            Tuple of (is_feasible, violation)
        """
        try:
            # Ensure x is a 1D array
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Check for NaN or inf values
            if not np.all(np.isfinite(x)):
                return False, float('inf')
            
            # Check if total delta-v is within tolerance
            total_dv = np.sum(x)
            dv_violation = abs(total_dv - self.TOTAL_DELTA_V) / self.TOTAL_DELTA_V
            
            # Check if any stage is negative or exceeds bounds
            for i, (lower, upper) in enumerate(self.bounds):
                if x[i] < lower or x[i] > upper:
                    return False, float('inf')
            
            # Check stage fraction constraints
            if self.config is not None:
                stage_constraints = self.config.get('constraints', {}).get('stage_fractions', {})
                if stage_constraints:
                    first_stage = stage_constraints.get('first_stage', {})
                    other_stages = stage_constraints.get('other_stages', {})
                    
                    # First stage constraints
                    min_first = first_stage.get('min_fraction', 0.15) * self.TOTAL_DELTA_V
                    max_first = first_stage.get('max_fraction', 0.80) * self.TOTAL_DELTA_V
                    if x[0] < min_first or x[0] > max_first:
                        return False, float('inf')
                    
                    # Other stages constraints
                    min_other = other_stages.get('min_fraction', 0.01) * self.TOTAL_DELTA_V
                    max_other = other_stages.get('max_fraction', 1.0) * self.TOTAL_DELTA_V
                    for i in range(1, self.n_stages):
                        if x[i] < min_other or x[i] > max_other:
                            return False, float('inf')
            else:
                # Use default constraints if config is None
                # First stage constraints
                min_first = 0.15 * self.TOTAL_DELTA_V
                max_first = 0.80 * self.TOTAL_DELTA_V
                if x[0] < min_first or x[0] > max_first:
                    return False, float('inf')
                
                # Other stages constraints
                min_other = 0.01 * self.TOTAL_DELTA_V
                max_other = 1.0 * self.TOTAL_DELTA_V
                for i in range(1, self.n_stages):
                    if x[i] < min_other or x[i] > max_other:
                        return False, float('inf')
            
            # Use a more robust approach for physics constraints
            try:
                # Get objective components with a try-except to catch any numerical issues
                _, dv_const, phys_const = objective_with_penalty(
                    dv=x,
                    G0=self.G0,
                    ISP=self.ISP,
                    EPSILON=self.EPSILON,
                    TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                    return_tuple=True
                )
                
                # Cap extremely large values to prevent inf
                if not np.isfinite(dv_const):
                    dv_const = 1e6
                if not np.isfinite(phys_const):
                    phys_const = 1e6
                    
                total_violation = dv_const + phys_const
                
                # Cap the total violation to prevent numerical issues
                if total_violation > 1e6:
                    total_violation = 1e6
                
            except Exception as e:
                logger.warning(f"Error in physics constraint calculation: {str(e)}")
                return False, 1e6
            
            is_feasible = total_violation <= self.feasibility_threshold
            
            return is_feasible, total_violation
            
        except Exception as e:
            logger.error(f"Error checking feasibility: {str(e)}")
            return False, 1e6
            
    def update_best_solution(self, x: np.ndarray, score: float, 
                           is_feasible: bool, violation: float) -> bool:
        """Update best solution if improvement found.
        
        Args:
            x: Solution vector
            score: Objective value
            is_feasible: Whether solution is feasible
            violation: Constraint violation measure
            
        Returns:
            True if improvement found
        """
        try:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Track statistics
            if is_feasible:
                self.n_feasible += 1
            else:
                self.n_infeasible += 1
            
            # Update best feasible solution if this is feasible and better
            if is_feasible:
                if score < self.best_feasible_score:
                    self.best_feasible = x.copy()
                    self.best_feasible_score = score
                    
                    # Also update best overall solution if this is the best feasible one
                    if not hasattr(self, 'best_solution') or not hasattr(self, 'best_fitness') or \
                       self.best_solution is None or not self.best_is_feasible or score < self.best_fitness:
                        self.best_solution = x.copy()
                        self.best_fitness = score
                        self.best_is_feasible = True
                        self.best_violation = violation
                        return True
            
            # If we don't have a feasible solution yet, or this infeasible solution is better than our current best
            if not hasattr(self, 'best_solution') or self.best_solution is None or \
               (not is_feasible and not self.best_is_feasible and score < self.best_fitness) or \
               (not self.best_is_feasible and is_feasible):
                self.best_solution = x.copy()
                self.best_fitness = score
                self.best_is_feasible = is_feasible
                self.best_violation = violation
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error updating best solution: {str(e)}")
            return False
            
    def project_to_feasible(self, x: np.ndarray) -> np.ndarray:
        """Project solution to feasible space.
        
        Args:
            x: Solution vector
            
        Returns:
            Projected solution
        """
        try:
            # Ensure x is a 1D array of float64 for numerical stability
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Check for NaN or inf values and replace with reasonable values
            if not np.all(np.isfinite(x)):
                logger.warning(f"Non-finite values in solution: {x}")
                # Replace non-finite values with bounds midpoints
                for i in range(len(x)):
                    if not np.isfinite(x[i]):
                        lower, upper = self.bounds[i]
                        x[i] = (lower + upper) / 2.0
            
            # Apply bounds constraints first
            for i, (lower, upper) in enumerate(self.bounds):
                x[i] = np.clip(x[i], lower, upper)
            
            # Ensure each stage has a minimum delta-v value for physics calculations
            # Increased from 1.0 to 10.0 to avoid numerical issues
            min_dv_value = 10.0  # 10 m/s minimum
            for i in range(len(x)):
                if x[i] < min_dv_value:
                    x[i] = min_dv_value
            
            # Scale to match total delta-v
            total = np.sum(x)
            if total > 0:
                x = x * (self.TOTAL_DELTA_V / total)
            else:
                # If total is zero or negative, distribute equally
                x = np.full_like(x, self.TOTAL_DELTA_V / len(x))
            
            # Apply stage-specific constraints if defined
            if self.config is not None:
                stage_constraints = self.config.get('constraints', {}).get('stage_fractions', {})
                if stage_constraints:
                    first_stage = stage_constraints.get('first_stage', {})
                    other_stages = stage_constraints.get('other_stages', {})
                    
                    # First stage constraints
                    min_first = max(first_stage.get('min_fraction', 0.15) * self.TOTAL_DELTA_V, min_dv_value)
                    max_first = first_stage.get('max_fraction', 0.80) * self.TOTAL_DELTA_V
                    x[0] = np.clip(x[0], min_first, max_first)
                    
                    # Other stages constraints
                    min_other = max(other_stages.get('min_fraction', 0.05) * self.TOTAL_DELTA_V, min_dv_value)
                    max_other = other_stages.get('max_fraction', 0.95) * self.TOTAL_DELTA_V
                    for i in range(1, self.n_stages):
                        x[i] = np.clip(x[i], min_other, max_other)
            else:
                # Use default constraints if config is None
                # First stage constraints
                min_first = max(0.15 * self.TOTAL_DELTA_V, min_dv_value)
                max_first = 0.80 * self.TOTAL_DELTA_V
                x[0] = np.clip(x[0], min_first, max_first)
                
                # Other stages constraints
                min_other = max(0.05 * self.TOTAL_DELTA_V, min_dv_value)
                max_other = 0.95 * self.TOTAL_DELTA_V
                for i in range(1, self.n_stages):
                    x[i] = np.clip(x[i], min_other, max_other)
            
            # Ensure the total is exactly TOTAL_DELTA_V
            total = np.sum(x)
            if abs(total - self.TOTAL_DELTA_V) > 1e-10:
                # Distribute the difference proportionally
                scale_factor = self.TOTAL_DELTA_V / total if total > 0 else 1.0
                x = x * scale_factor
                
                # After scaling, ensure minimum values are still maintained
                for i in range(len(x)):
                    if x[i] < min_dv_value:
                        x[i] = min_dv_value
                
                # Final check for exact constraint
                error = self.TOTAL_DELTA_V - np.sum(x)
                if abs(error) > 1e-10:
                    # Distribute error to stages above minimum threshold
                    above_min = [i for i in range(len(x)) if x[i] > min_dv_value * 1.1]
                    if above_min:
                        # Distribute error to stages that are well above minimum
                        correction = error / len(above_min)
                        for i in above_min:
                            x[i] += correction
                    else:
                        # If all stages are near minimum, adjust the largest one
                        largest_idx = np.argmax(x)
                        x[largest_idx] += error
            
            # Final validation - log warning if any stage is still too small
            if np.any(x < min_dv_value * 0.99):
                logger.warning(f"After projection, some stages still have very small delta-v: {x}")
            
            return x
            
        except Exception as e:
            logger.error(f"Error in projection: {str(e)}")
            # Fallback to equal distribution
            return np.full(self.n_stages, self.TOTAL_DELTA_V / self.n_stages)
        
    def initialize_population_lhs(self) -> np.ndarray:
        """Initialize population using Latin Hypercube Sampling."""
        try:
            from scipy.stats import qmc
            
            # Use Latin Hypercube Sampling for better coverage
            sampler = qmc.LatinHypercube(d=self.n_stages)
            samples = sampler.random(n=self.population_size)
            
            # Convert to float64 for numerical stability
            population = np.zeros((self.population_size, self.n_stages), dtype=np.float64)
            
            # Scale samples to stage-specific ranges
            for i in range(self.population_size):
                for j in range(self.n_stages):
                    lower, upper = self.bounds[j]
                    population[i,j] = lower + samples[i,j] * (upper - lower)
                    
                # Project to feasible space
                population[i] = self.project_to_feasible(population[i])
                    
            return population
            
        except Exception as e:
            logger.warning(f"LHS initialization failed: {str(e)}, using uniform random")
            return self.initialize_population_uniform()
            
    def initialize_population_uniform(self) -> np.ndarray:
        """Initialize population using uniform random sampling."""
        population = np.zeros((self.population_size, self.n_stages), dtype=np.float64)
        
        for i in range(self.population_size):
            # Generate random position within bounds
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                population[i,j] = np.random.uniform(lower, upper)
                
            # Project to feasible space
            population[i] = self.project_to_feasible(population[i])
            
        return population
        
    def process_results(self, x: np.ndarray, success: bool = True, message: str = "", 
                       n_iterations: int = 0, n_function_evals: int = 0, 
                       time: float = 0.0, constraint_violation: float = None) -> Dict:
        """Process optimization results into a standardized format.
        
        Args:
            x: Solution vector (delta-v values)
            success: Whether optimization succeeded
            message: Status message from optimizer
            n_iterations: Number of iterations performed
            n_function_evals: Number of function evaluations
            time: Execution time in seconds
            constraint_violation: Optional pre-computed constraint violation
            
        Returns:
            Dictionary containing standardized optimization results
        """
        try:
            # Convert x to numpy array and validate
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if x.size == 0 or not np.all(np.isfinite(x)):
                raise ValueError("Invalid solution vector")
            
            # Calculate ratios and payload fraction
            stage_ratios, mass_ratios = calculate_stage_ratios(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
            payload_fraction = calculate_payload_fraction(mass_ratios)
            
            # Check if solution is feasible
            if constraint_violation is None:
                is_feasible, violation = self.check_feasibility(x)
            else:
                violation = constraint_violation
                is_feasible = violation <= self.feasibility_threshold
            
            # Build stages info if solution is feasible
            stages = []
            if is_feasible:
                for i, (dv, mr, sr) in enumerate(zip(x, mass_ratios, stage_ratios)):
                    stages.append({
                        'stage': i + 1,
                        'delta_v': float(dv),
                        'Lambda': float(sr)
                    })
            
            # Only update success flag if the solution is not feasible
            # This preserves the success flag from the solver if it's already True
            if not is_feasible:
                success = False
            
            # Update message if constraints are violated
            if not is_feasible:
                message = f"Solution violates constraints (violation={violation:.2e})"
            
            return {
                'success': success,
                'message': message,
                'payload_fraction': float(payload_fraction) if is_feasible else 0.0,
                'constraint_violation': float(violation),
                'execution_metrics': {
                    'iterations': n_iterations,
                    'function_evaluations': n_function_evals,
                    'execution_time': time
                },
                'stages': stages
            }
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return {
                'success': False,
                'message': f"Failed to process results: {str(e)}",
                'payload_fraction': 0.0,
                'constraint_violation': float('inf'),
                'execution_metrics': {
                    'iterations': n_iterations,
                    'function_evaluations': n_function_evals,
                    'execution_time': time
                },
                'stages': []
            }
        
    @abstractmethod
    def solve(self, initial_guess, bounds, other_solver_results=None):
        """Solve optimization problem.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            other_solver_results: Optional dictionary of solutions from other solvers
            
        Returns:
            Dictionary containing optimization results
        """
        pass
