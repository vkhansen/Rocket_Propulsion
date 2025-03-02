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
        """Check if solution satisfies all constraints.
        
        Args:
            x: Solution vector
            
        Returns:
            Tuple of (is_feasible, violation_measure)
        """
        try:
            # Ensure x is a 1D array
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Get objective components
            _, dv_const, phys_const = objective_with_penalty(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON,
                TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                return_tuple=True
            )
            
            total_violation = dv_const + phys_const
            is_feasible = total_violation <= self.feasibility_threshold
            
            return is_feasible, total_violation
            
        except Exception as e:
            logger.error(f"Error checking feasibility: {str(e)}")
            return False, float('inf')
            
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
            
            if is_feasible:
                self.n_feasible += 1
                if score < self.best_feasible_score:
                    self.best_feasible = x.copy()
                    self.best_feasible_score = score
                    return True
            else:
                self.n_infeasible += 1
                
            return False
            
        except Exception as e:
            logger.error(f"Error updating best solution: {str(e)}")
            return False
            
    def iterative_projection(self, x: np.ndarray) -> np.ndarray:
        """Project solution to feasible space using iterative water-filling approach.
        
        This method uses a sophisticated water-filling algorithm to redistribute delta-v:
        1. First clips values to their bounds
        2. Identifies stages at their limits (fixed)
        3. Redistributes remaining delta-v among unfixed stages
        4. Repeats until convergence or max iterations reached
        """
        try:
            x_proj = np.asarray(x, dtype=np.float64).reshape(-1)
            
            for iteration in range(self.max_projection_iterations):
                # Track which stages are fixed at their bounds
                fixed_stages = np.zeros(self.n_stages, dtype=bool)
                fixed_values = np.zeros(self.n_stages, dtype=np.float64)
                
                # First pass: Clip to bounds and identify fixed stages
                for i in range(self.n_stages):
                    lower, upper = self.bounds[i]
                    if x_proj[i] <= lower:
                        x_proj[i] = lower
                        fixed_stages[i] = True
                        fixed_values[i] = lower
                    elif x_proj[i] >= upper:
                        x_proj[i] = upper
                        fixed_stages[i] = True
                        fixed_values[i] = upper
                
                # Calculate remaining delta-v to distribute
                fixed_dv = np.sum(fixed_values)
                remaining_dv = self.TOTAL_DELTA_V - fixed_dv
                unfixed_stages = ~fixed_stages
                n_unfixed = np.sum(unfixed_stages)
                
                if n_unfixed == 0:
                    # All stages are fixed - cannot satisfy constraints
                    logger.warning("All stages fixed at bounds - cannot satisfy constraints")
                    break
                
                # Calculate relative proportions for unfixed stages
                if n_unfixed > 1:
                    current_sum = np.sum(x_proj[unfixed_stages])
                    if current_sum > 0:
                        # Preserve relative proportions
                        proportions = x_proj[unfixed_stages] / current_sum
                    else:
                        # Equal distribution if current sum is zero
                        proportions = np.ones(n_unfixed) / n_unfixed
                    
                    # Redistribute remaining delta-v according to proportions
                    x_proj[unfixed_stages] = remaining_dv * proportions
                else:
                    # Only one unfixed stage - assign all remaining delta-v
                    x_proj[unfixed_stages] = remaining_dv
                
                # Check if solution is feasible
                total = np.sum(x_proj)
                rel_error = abs(total - self.TOTAL_DELTA_V) / self.TOTAL_DELTA_V
                
                if rel_error <= self.precision_threshold:
                    break
                
                # Apply stage-specific constraints if defined
                stage_constraints = self.config.get('constraints', {}).get('stage_fractions', {})
                if stage_constraints:
                    first_stage = stage_constraints.get('first_stage', {})
                    other_stages = stage_constraints.get('other_stages', {})
                    
                    # First stage constraints
                    min_first = first_stage.get('min_fraction', 0.15) * self.TOTAL_DELTA_V
                    max_first = first_stage.get('max_fraction', 0.80) * self.TOTAL_DELTA_V
                    x_proj[0] = np.clip(x_proj[0], min_first, max_first)
                    
                    # Other stages constraints
                    min_other = other_stages.get('min_fraction', 0.01) * self.TOTAL_DELTA_V
                    max_other = other_stages.get('max_fraction', 1.0) * self.TOTAL_DELTA_V
                    for i in range(1, self.n_stages):
                        x_proj[i] = np.clip(x_proj[i], min_other, max_other)
            
            return x_proj
            
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
                population[i] = self.iterative_projection(population[i])
                    
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
            population[i] = self.iterative_projection(population[i])
            
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
            
            # Update success flag based on both optimizer success and constraint feasibility
            success = success and is_feasible
            
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
    def solve(self, initial_guess, bounds):
        """Solve optimization problem.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            Tuple of (best solution, best objective value)
        """
        pass
