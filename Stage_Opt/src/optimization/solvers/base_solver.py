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
        
        # Bootstrap solution tracking
        self.best_bootstrap_solution = None
        self.best_bootstrap_fitness = float('inf')
        
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
            
    def evaluate_solution_with_details(self, x: np.ndarray) -> tuple:
        """Evaluate a solution and return detailed information.
        
        Args:
            x: Solution vector
            
        Returns:
            Tuple of (fitness, is_feasible, violation)
        """
        try:
            # Check feasibility
            is_feasible, violation = self.check_feasibility(x)
            
            # Calculate objective value
            fitness = self.evaluate_solution(x)
            
            return fitness, is_feasible, violation
            
        except Exception as e:
            logger.error(f"Error evaluating solution with details: {str(e)}")
            return float('-inf'), False, float('inf')
            
    def update_best_solution(self, x: np.ndarray, score: float, 
                           is_feasible: bool, violation: float) -> bool:
        """Update best solution if improvement found.
        
        Args:
            x: Solution vector
            score: Objective value (negative payload fraction, lower is better)
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
            
            # Calculate payload fraction for comparison (score is negative payload fraction)
            current_payload_fraction = -score if is_feasible else -float('inf')
            best_feasible_payload_fraction = -self.best_feasible_score if self.best_feasible is not None else -float('inf')
            best_overall_payload_fraction = -self.best_fitness if self.best_solution is not None else -float('inf')
            
            # Enhanced logging to track solution comparison
            logger.debug(f"Evaluating solution: payload_fraction={current_payload_fraction:.6f}, feasible={is_feasible}, violation={violation:.6f}")
            logger.debug(f"Solution vector: {x}")
            
            if self.best_solution is not None:
                logger.debug(f"Current best: payload_fraction={best_overall_payload_fraction:.6f}, feasible={self.best_is_feasible}, violation={self.best_violation:.6f}")
                logger.debug(f"Current best vector: {self.best_solution}")
                
                # Log comparison details
                if is_feasible and self.best_is_feasible:
                    logger.debug(f"Comparing feasible solutions: current={current_payload_fraction:.6f} vs best={best_overall_payload_fraction:.6f}")
                    if current_payload_fraction > best_overall_payload_fraction:
                        logger.debug(f"IMPROVEMENT: Better feasible solution found (+{current_payload_fraction - best_overall_payload_fraction:.6f})")
                    else:
                        logger.debug(f"REJECTED: Current solution is worse than best (-{best_overall_payload_fraction - current_payload_fraction:.6f})")
                elif is_feasible and not self.best_is_feasible:
                    logger.debug(f"IMPROVEMENT: First feasible solution found or replacing infeasible best")
                elif not is_feasible and self.best_is_feasible:
                    logger.debug(f"REJECTED: Current solution is infeasible while best is feasible")
                else:  # both infeasible
                    logger.debug(f"Comparing infeasible solutions: current violation={violation:.6f} vs best={self.best_violation:.6f}")
                    if violation < self.best_violation:
                        logger.debug(f"IMPROVEMENT: Better infeasible solution found (violation reduced by {self.best_violation - violation:.6f})")
                    else:
                        logger.debug(f"REJECTED: Current infeasible solution has higher violation (+{violation - self.best_violation:.6f})")
            else:
                logger.debug(f"No existing best solution, will accept current solution")
            
            # Check if solution is worse than bootstrap solution
            if self.is_worse_than_bootstrap(score, is_feasible):
                logger.info(f"Rejected solution with payload fraction {current_payload_fraction:.6f} as it's worse than bootstrap solution")
                return False
            
            # Update best feasible solution if this is feasible and has better payload fraction
            if is_feasible:
                if self.best_feasible is None or current_payload_fraction > best_feasible_payload_fraction:
                    self.best_feasible = x.copy()
                    self.best_feasible_score = score
                    logger.debug(f"New best feasible solution with payload fraction {current_payload_fraction:.6f}")
                    
                    # Also update best overall solution if this is the best feasible one
                    if self.best_solution is None or not self.best_is_feasible or current_payload_fraction > best_overall_payload_fraction:
                        self.best_solution = x.copy()
                        self.best_fitness = score
                        self.best_is_feasible = True
                        self.best_violation = violation
                        logger.info(f"New best overall solution (feasible) with payload fraction {current_payload_fraction:.6f}")
                        return True
            
            # If we don't have a feasible solution yet, or this infeasible solution is better than our current best
            elif self.best_solution is None or \
               (not is_feasible and not self.best_is_feasible and score < self.best_fitness) or \
               (not self.best_is_feasible and is_feasible):
                self.best_solution = x.copy()
                self.best_fitness = score
                self.best_is_feasible = is_feasible
                self.best_violation = violation
                
                if is_feasible:
                    logger.info(f"New best overall solution (feasible) with payload fraction {current_payload_fraction:.6f}")
                else:
                    logger.info(f"New best overall solution (infeasible) with score {score:.6f} and violation {violation:.6f}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error updating best solution: {str(e)}")
            return False
            
    def is_worse_than_bootstrap(self, score: float, is_feasible: bool) -> bool:
        """Check if a solution is worse than the best bootstrap solution.
        
        Args:
            score: Objective value (negative payload fraction, lower is better)
            is_feasible: Whether solution is feasible
            
        Returns:
            True if solution is worse than bootstrap, False otherwise
        """
        # If we don't have a bootstrap solution, nothing can be worse than it
        if self.best_bootstrap_solution is None:
            return False
            
        # Calculate payload fraction for comparison (score is negative payload fraction)
        current_payload_fraction = -score if is_feasible else -float('inf')
        bootstrap_payload_fraction = -self.best_bootstrap_fitness
        
        # Only feasible solutions can be compared properly
        if is_feasible:
            # If the current payload fraction is worse than the bootstrap
            if current_payload_fraction < bootstrap_payload_fraction:
                logger.debug(f"Solution rejected: payload fraction {current_payload_fraction:.6f} is worse than bootstrap {bootstrap_payload_fraction:.6f}")
                return True
        
        return False
            
    def project_to_feasible(self, x: np.ndarray) -> np.ndarray:
        """Project solution to feasible space.
        
        Args:
            x: Solution vector to project
            
        Returns:
            Projected solution
        """
        try:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Ensure all values are positive
            x = np.maximum(x, 1e-10)
            
            # Ensure sum equals TOTAL_DELTA_V
            x = x * (self.TOTAL_DELTA_V / np.sum(x))
            
            # Apply stage fraction constraints from bounds
            for i in range(len(x)):
                lower, upper = self.bounds[i]
                if x[i] < lower:
                    x[i] = lower
                elif x[i] > upper:
                    x[i] = upper
            
            # Re-normalize to ensure sum equals TOTAL_DELTA_V after bounds are applied
            x = x * (self.TOTAL_DELTA_V / np.sum(x))
            
            return x
            
        except Exception as e:
            logger.error(f"Error projecting to feasible space: {str(e)}")
            # Return original solution if error occurs
            return x
            
    def perturb_solution(self, solution, scale=0.1):
        """Create a perturbed version of a solution for exploration.
        
        Args:
            solution: Original solution to perturb
            scale: Scale of perturbation (relative to solution range)
            
        Returns:
            Perturbed solution
        """
        try:
            # Generate random perturbation
            perturbation = np.random.uniform(-scale, scale, len(solution))
            perturbed = solution * (1 + perturbation)
            
            # Ensure the solution still sums to TOTAL_DELTA_V
            perturbed = perturbed * (self.TOTAL_DELTA_V / np.sum(perturbed))
            
            # Project to feasible space to ensure all constraints are met
            perturbed = self.project_to_feasible(perturbed)
            
            return perturbed
            
        except Exception as e:
            logger.error(f"Error perturbing solution: {str(e)}")
            return solution.copy()  # Return original solution if error occurs
            
    def process_bootstrap_solutions(self, bootstrap_solutions, n_perturbed=5, perturbation_scale=0.1):
        """Process bootstrap solutions from other solvers.
        
        Args:
            bootstrap_solutions: List of solutions from other solvers
            n_perturbed: Number of perturbed versions to generate for each bootstrap solution
            perturbation_scale: Scale of perturbation (relative to solution range)
            
        Returns:
            List of processed bootstrap solutions
        """
        if not bootstrap_solutions:
            logger.warning("No bootstrap solutions provided")
            return []
            
        processed_solutions = []
        best_bootstrap_fitness = float('-inf')
        best_bootstrap_solution = None
        
        # First, evaluate all bootstrap solutions to find the best one
        for solution in bootstrap_solutions:
            try:
                # Evaluate the solution
                fitness, is_feasible, violation = self.evaluate_solution_with_details(solution)
                
                # Track the best bootstrap solution
                if is_feasible and (best_bootstrap_solution is None or fitness > best_bootstrap_fitness):
                    best_bootstrap_solution = solution.copy()
                    best_bootstrap_fitness = fitness
                    logger.info(f"New best bootstrap solution found with fitness {fitness:.6f}")
                
                # Add original solution
                processed_solutions.append(solution.copy())
                
                # Generate perturbed versions
                for _ in range(n_perturbed):
                    perturbed = self.perturb_solution(solution, scale=perturbation_scale)
                    processed_solutions.append(perturbed)
                    
                    # Also evaluate the perturbed solution
                    perturbed_fitness, is_perturbed_feasible, _ = self.evaluate_solution_with_details(perturbed)
                    
                    # Check if this perturbed solution is better than our best bootstrap
                    if is_perturbed_feasible and (best_bootstrap_solution is None or perturbed_fitness > best_bootstrap_fitness):
                        best_bootstrap_solution = perturbed.copy()
                        best_bootstrap_fitness = perturbed_fitness
                        logger.info(f"New best bootstrap solution (perturbed) found with fitness {perturbed_fitness:.6f}")
                
            except Exception as e:
                logger.error(f"Error processing bootstrap solution: {str(e)}")
        
        # Store the best bootstrap solution
        if best_bootstrap_solution is not None:
            self.best_bootstrap_solution = best_bootstrap_solution.copy()
            self.best_bootstrap_fitness = best_bootstrap_fitness
            
            # Also update best solution if it's better than current best
            if self.best_solution is None or best_bootstrap_fitness > self.best_fitness:
                self.best_solution = best_bootstrap_solution.copy()
                self.best_fitness = best_bootstrap_fitness
                self.best_is_feasible = True
                _, self.best_violation = self.check_feasibility(best_bootstrap_solution)
                logger.info(f"Best bootstrap solution is better than current best, updating best solution")
                
            # Log detailed information about the best bootstrap solution
            payload_fraction = -best_bootstrap_fitness  # Convert from negative to positive
            logger.info(f"Best bootstrap solution has payload fraction: {payload_fraction:.6f}")
            
            # Calculate and log stage ratios for the best bootstrap solution
            try:
                stage_ratios = self.calculate_stage_ratios(best_bootstrap_solution)
                logger.info(f"Best bootstrap solution stage ratios: {stage_ratios}")
            except Exception as e:
                logger.error(f"Error calculating stage ratios for best bootstrap: {str(e)}")
        else:
            logger.warning("No feasible bootstrap solutions found")
        
        return processed_solutions
        
    def iterative_projection(self, x: np.ndarray, max_iterations: int = 5) -> np.ndarray:
        """Apply projection iteratively to ensure constraints are met.
        
        This method applies the projection multiple times to ensure that
        all constraints are satisfied, especially when there are competing
        constraints that might be violated after a single projection.
        
        Args:
            x: Solution vector
            max_iterations: Maximum number of projection iterations
            
        Returns:
            Projected solution that satisfies all constraints
        """
        try:
            # Start with initial projection
            projected = self.project_to_feasible(x)
            
            # Iteratively apply projection until convergence or max iterations
            for i in range(max_iterations - 1):
                prev_projected = projected.copy()
                projected = self.project_to_feasible(projected)
                
                # Check if solution has converged
                error = np.max(np.abs(projected - prev_projected))
                if error < 1e-8:
                    logger.debug(f"Projection converged after {i+1} iterations")
                    break
                    
            return projected
            
        except Exception as e:
            logger.error(f"Error in iterative projection: {str(e)}")
            # Fallback to single projection
            return self.project_to_feasible(x)
            
    def calculate_stage_ratios(self, x: np.ndarray) -> np.ndarray:
        """Calculate stage ratios for a solution.
        
        Args:
            x: Solution vector (delta-v per stage)
            
        Returns:
            Array of stage ratios
        """
        try:
            # Use the imported calculate_stage_ratios function
            stage_ratios, _ = calculate_stage_ratios(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
            
            return stage_ratios
            
        except Exception as e:
            logger.error(f"Error calculating stage ratios: {str(e)}")
            return np.zeros_like(x)
            
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
