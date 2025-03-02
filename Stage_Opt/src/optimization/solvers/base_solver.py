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
        self.ISP = np.array(ISP, dtype=float)
        self.EPSILON = np.array(EPSILON, dtype=float)
        self.TOTAL_DELTA_V = float(TOTAL_DELTA_V)
        self.bounds = bounds
        self.config = config
        self.n_stages = len(bounds)
        self.name = self.__class__.__name__
        
        # Constraint parameters
        self.feasibility_threshold = 1e-4
        self.precision_threshold = 1e-10
        self.max_projection_iterations = 10
        
        # Statistics tracking
        self.n_feasible = 0
        self.n_infeasible = 0
        self.best_feasible = None
        self.best_feasible_score = float('inf')
        
        logger.debug(f"Initialized {self.name} with {self.n_stages} stages")
        
        # Initialize cache
        self.cache = OptimizationCache()
        
    def calculate_stage_ratios(self, x: np.ndarray):
        """Calculate stage ratios for a solution.
        
        Args:
            x: Array of delta-v values
            
        Returns:
            Tuple of (stage_ratios, mass_ratios)
        """
        try:
            # Ensure x is a 1D array matching the number of stages
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != len(self.ISP):
                raise ValueError(f"Expected {len(self.ISP)} stages, got {x.size}")
                
            return calculate_stage_ratios(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
        except Exception as e:
            logger.error(f"Error calculating stage ratios: {str(e)}")
            return np.zeros_like(self.ISP), np.zeros_like(self.ISP)

    def project_to_feasible(self, x: np.ndarray) -> np.ndarray:
        """Project solution to feasible space with high precision.
        
        Args:
            x: Input solution vector
            
        Returns:
            Projected solution satisfying all constraints
        """
        try:
            # Convert to high precision
            x = np.array(x, dtype=np.float64)
            
            # First normalize to total delta-v
            total = np.sum(x)
            if total > 0:
                x *= self.TOTAL_DELTA_V / total
            else:
                x = np.full_like(x, self.TOTAL_DELTA_V / len(x))
            
            # Iterative projection to handle bounds and total delta-v
            for _ in range(self.max_projection_iterations):
                # Project to bounds
                for i in range(len(x)):
                    lower, upper = self.bounds[i]
                    x[i] = np.clip(x[i], lower, upper)
                
                # Re-normalize to maintain total delta-v
                total = np.sum(x)
                if total > 0:
                    x *= self.TOTAL_DELTA_V / total
            
            # Final high-precision correction
            error = abs(np.sum(x) - self.TOTAL_DELTA_V)
            if error > self.precision_threshold:
                # Distribute error proportionally
                correction = (self.TOTAL_DELTA_V - np.sum(x)) / len(x)
                x += correction
                
                # Final bounds check with redistribution
                for i in range(len(x)):
                    lower, upper = self.bounds[i]
                    if x[i] < lower:
                        excess = (lower - x[i]) / (len(x) - 1)
                        x[i] = lower
                        for j in range(len(x)):
                            if j != i:
                                x[j] += excess
                    elif x[i] > upper:
                        deficit = (x[i] - upper) / (len(x) - 1)
                        x[i] = upper
                        for j in range(len(x)):
                            if j != i:
                                x[j] -= deficit
            
            return x
            
        except Exception as e:
            logger.error(f"Error in projection: {str(e)}")
            return np.full(len(x), self.TOTAL_DELTA_V / len(x))

    def evaluate_solution(self, x: np.ndarray, return_components: bool = False) -> Union[float, Tuple[float, float, float]]:
        """Evaluate solution with constraint handling.
        
        Args:
            x: Solution vector
            return_components: If True, returns (objective, dv_violation, physical_violation)
            
        Returns:
            Either total score or components tuple
        """
        try:
            # Project to feasible space first
            x = self.project_to_feasible(x)
            
            # Get objective and constraints
            obj, dv_const, phys_const = objective_with_penalty(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON,
                TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                return_tuple=True
            )
            
            # Track feasibility
            total_violation = dv_const + phys_const
            if total_violation <= self.feasibility_threshold:
                self.n_feasible += 1
                if obj < self.best_feasible_score:
                    self.best_feasible = x.copy()
                    self.best_feasible_score = obj
            else:
                self.n_infeasible += 1
            
            if return_components:
                return obj, dv_const, phys_const
            else:
                return obj + total_violation
                
        except Exception as e:
            logger.error(f"Error evaluating solution: {str(e)}")
            if return_components:
                return float('inf'), float('inf'), float('inf')
            else:
                return float('inf')

    def check_feasibility(self, x: np.ndarray) -> Tuple[bool, float]:
        """Check if solution is feasible and get violation amount.
        
        Args:
            x: Solution vector
            
        Returns:
            Tuple of (is_feasible, violation_amount)
        """
        _, dv_const, phys_const = self.evaluate_solution(x, return_components=True)
        total_violation = dv_const + phys_const
        return total_violation <= self.feasibility_threshold, total_violation

    def check_constraints(self, x):
        """Check if solution violates any constraints.
        
        Returns:
            tuple: (is_feasible, violation_amount)
            - is_feasible: True if solution satisfies all constraints
            - violation_amount: Amount of constraint violation (0 if feasible)
        """
        # Check total ΔV constraint
        total_dv = np.sum(x)
        dv_violation = abs(total_dv - self.TOTAL_DELTA_V) / self.TOTAL_DELTA_V
        
        # Check bounds constraints
        bounds_violation = 0
        for i, (lower, upper) in enumerate(self.bounds):
            if x[i] < lower:
                bounds_violation += abs(x[i] - lower) / self.TOTAL_DELTA_V
            elif x[i] > upper:
                bounds_violation += abs(x[i] - upper) / self.TOTAL_DELTA_V
                
        total_violation = dv_violation + bounds_violation
        is_feasible = total_violation < 1e-6  # Strict tolerance
        
        return is_feasible, total_violation

    def process_results(self, x: np.ndarray, success: bool = True, message: str = "", 
                       n_iterations: int = 0, n_function_evals: int = 0, 
                       time: float = 0.0) -> Dict:
        """Process optimization results into a standardized format.
        
        Args:
            x: Solution vector (delta-v values)
            success: Whether optimization succeeded
            message: Status message from optimizer
            n_iterations: Number of iterations performed
            n_function_evals: Number of function evaluations
            time: Execution time in seconds
            
        Returns:
            Dictionary containing standardized optimization results
        """
        try:
            # Convert x to numpy array and validate
            x = np.asarray(x, dtype=float)
            if x.size == 0 or not np.all(np.isfinite(x)):
                raise ValueError("Invalid solution vector")
            
            # Calculate ratios and payload fraction
            stage_ratios, mass_ratios = self.calculate_stage_ratios(x)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            
            # Calculate total constraint violation
            constraint_violation = enforce_stage_constraints(
                dv_array=x,
                total_dv_required=self.TOTAL_DELTA_V,
                config=self.config
            )
            
            # Check if solution is feasible (constraints satisfied within tolerance)
            feasibility_threshold = 1e-4  # Threshold for considering constraints satisfied
            is_feasible = constraint_violation <= feasibility_threshold
            
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
                message = f"Solution violates constraints (violation={constraint_violation:.2e})"
            
            return {
                'success': success,  # Now properly reflects both optimizer success and feasibility
                'message': message,
                'payload_fraction': float(payload_fraction) if is_feasible else 0.0,
                'constraint_violation': float(constraint_violation),
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
        """Solve the optimization problem.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            Dictionary containing optimization results
        """
        pass
