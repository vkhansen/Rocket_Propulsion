"""Base solver class for optimization."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np

from ...utils.config import logger
from ..cache import OptimizationCache
from ..physics import calculate_stage_ratios, calculate_payload_fraction
from ..objective import enforce_stage_constraints, objective_with_penalty

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
        
    def evaluate_solution(self, x: np.ndarray) -> Tuple[float, bool]:
        """Evaluate a solution vector.
        
        Args:
            x: Solution vector (delta-v values)
            
        Returns:
            Tuple of (objective value, is feasible)
        """
        try:
            # Ensure x is a 1D array
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Check cache first
            cached = self.cache.get(tuple(x))
            if cached is not None:
                return cached
            
            # Calculate stage ratios
            stage_ratios, mass_ratios = calculate_stage_ratios(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
            
            # Calculate payload fraction
            payload_fraction = calculate_payload_fraction(stage_ratios)
            
            # Check constraints and apply penalties
            is_feasible, violation = self.check_feasibility(x)
            objective = objective_with_penalty(payload_fraction, violation)
            
            # Cache result
            self.cache.add(tuple(x), (objective, is_feasible))
            
            return objective, is_feasible
            
        except Exception as e:
            logger.error(f"Error evaluating solution: {str(e)}")
            return float('inf'), False
            
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
            
            # Check total delta-v constraint
            total_dv = np.sum(x)
            rel_error = abs(total_dv - self.TOTAL_DELTA_V) / self.TOTAL_DELTA_V
            
            if rel_error > self.feasibility_threshold:
                return False, rel_error
                
            # Check stage bounds
            for i, (lower, upper) in enumerate(self.bounds):
                if x[i] < lower or x[i] > upper:
                    return False, abs(min(x[i] - lower, x[i] - upper))
                    
            # Check stage ratios are valid
            stage_ratios, _ = calculate_stage_ratios(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
            
            if np.any(stage_ratios <= 1.0):
                return False, np.sum(1.0 - stage_ratios[stage_ratios <= 1.0])
                
            return True, 0.0
            
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
        """Project solution to feasible space using iterative refinement."""
        try:
            x_proj = np.asarray(x, dtype=np.float64).reshape(-1)
            
            for _ in range(self.max_projection_iterations):
                # First ensure bounds constraints
                for i in range(self.n_stages):
                    lower, upper = self.bounds[i]
                    x_proj[i] = np.clip(x_proj[i], lower, upper)
                
                # Check total ΔV constraint using relative error
                total = np.sum(x_proj)
                rel_error = abs(total - self.TOTAL_DELTA_V) / self.TOTAL_DELTA_V
                
                if rel_error <= self.precision_threshold:
                    break
                    
                # Scale to match total ΔV
                x_proj *= self.TOTAL_DELTA_V / total
                
            return x_proj
            
        except Exception as e:
            logger.error(f"Error in projection: {str(e)}")
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
