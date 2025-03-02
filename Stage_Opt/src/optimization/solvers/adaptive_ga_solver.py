"""Adaptive Genetic Algorithm Solver implementation."""
import numpy as np
import time
from typing import Dict, List, Tuple
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from scipy.spatial.distance import pdist

from ...utils.config import logger
from .base_ga_solver import BaseGASolver
from ..pymoo_problem import RocketStageProblem

class AdaptiveGeneticAlgorithmSolver(BaseGASolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config: Dict, problem_params: Dict):
        """Initialize the adaptive GA solver."""
        super().__init__(config, problem_params)
        
        # Initialize adaptive parameters
        self.min_pop_size = 50
        self.max_pop_size = 300
        self.min_mutation_rate = 0.01
        self.max_mutation_rate = 0.4
        self.min_crossover_rate = 0.3
        self.max_crossover_rate = 1.0
        
        # Initialize tracking variables
        self.best_fitness = float('inf')
        self.generations_without_improvement = 0
        self.diversity_history = []
        self.constraint_violation_history = []
        
    def create_problem(self, initial_guess: np.ndarray, bounds: List[Tuple[float, float]]) -> Problem:
        """Create optimization problem instance."""
        n_var = len(initial_guess)
        bounds = np.array(bounds)  # Convert to numpy array
        
        return RocketStageProblem(
            solver=self,
            n_var=n_var,
            bounds=bounds
        )
        
    def setup_algorithm(self) -> GA:
        """Setup the GA algorithm with current parameters."""
        return GA(
            pop_size=self.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.crossover_rate),
            mutation=PM(prob=self.mutation_rate),
            eliminate_duplicates=True
        )
        
    def calculate_diversity(self, pop) -> float:
        """Calculate population diversity using pairwise distances."""
        try:
            if len(pop) < 2:
                return 0.0
            distances = pdist(pop.get("X"))
            return float(np.mean(distances))
        except Exception as e:
            logger.error(f"Error calculating diversity: {str(e)}")
            return 0.0
            
    def update_parameters(self, algorithm):
        """Update algorithm parameters based on progress."""
        try:
            # Get current best fitness
            current_best = algorithm.opt.get("F")[0]
            
            # Calculate diversity
            diversity = self.calculate_diversity(algorithm.pop)
            self.diversity_history.append(diversity)
            
            # Calculate mean constraint violation
            violations = algorithm.pop.get("CV")
            mean_violation = float(np.mean(violations)) if violations is not None else 0.0
            self.constraint_violation_history.append(mean_violation)
            
            # Check for improvement
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
            
            # Adjust parameters based on progress
            if self.generations_without_improvement > 10:
                # Increase exploration
                self.mutation_rate = min(self.max_mutation_rate, 
                                      self.mutation_rate * 1.5)
                self.pop_size = min(self.max_pop_size, 
                                  int(self.pop_size * 1.2))
            else:
                # Reduce exploration
                self.mutation_rate = max(self.min_mutation_rate, 
                                      self.mutation_rate * 0.9)
                self.pop_size = max(self.min_pop_size, 
                                  int(self.pop_size * 0.9))
            
            # Update algorithm parameters
            algorithm.pop_size = self.pop_size
            algorithm.mutation.prob = self.mutation_rate
            
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            
    def solve(self, initial_guess: np.ndarray, bounds: List[Tuple[float, float]]) -> Dict:
        """Solve the optimization problem using adaptive GA."""
        try:
            logger.info("Starting Adaptive GA optimization...")
            
            # Create problem instance
            problem = self.create_problem(initial_guess, bounds)
            
            # Setup algorithm
            algorithm = self.setup_algorithm()
            
            # Setup termination
            termination = get_termination("n_gen", self.n_generations)
            
            # Run optimization
            start_time = time.time()
            
            result = minimize(
                problem,
                algorithm,
                termination,
                seed=42,
                save_history=True,
                verbose=True
            )
            
            execution_time = time.time() - start_time
            
            # Process results
            return self.process_results(
                x=result.X,
                success=result.success,
                message=result.message,
                n_iterations=result.algorithm.n_gen,
                n_function_evals=result.algorithm.evaluator.n_eval,
                time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in Adaptive GA optimization: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
