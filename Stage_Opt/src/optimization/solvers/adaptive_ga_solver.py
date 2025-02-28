"""Adaptive Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from .ga_solver import RocketStageProblem
from .base_solver import BaseSolver
from ...utils.config import logger

class AdaptiveCallback(Callback):
    """Callback for adaptive parameter tuning."""
    
    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self.last_best = float('-inf')
        self.stall_count = 0
        self.adaptation_interval = 5  # Only adapt every N generations
        self.generation_counter = 0
        
    def notify(self, algorithm):
        """Called after each generation."""
        self.generation_counter += 1
        
        # Only adapt parameters every N generations
        if self.generation_counter % self.adaptation_interval != 0:
            return
            
        if algorithm.opt is None:
            return
            
        current_best = -float(algorithm.opt.get("F")[0])
        
        # If improvement is small or negative
        if current_best <= self.last_best * 1.001:  # Allow for 0.1% improvement
            self.stall_count += 1
            
            # Increase exploration (mutation rate) and decrease exploitation (crossover rate)
            if self.stall_count >= 3:  # If stalled for several generations
                self.solver.curr_mutation_rate = min(
                    self.solver.curr_mutation_rate * (1 + self.solver.adaptation_rate),
                    self.solver.max_mutation_rate
                )
                self.solver.curr_crossover_rate = max(
                    self.solver.curr_crossover_rate * (1 - self.solver.adaptation_rate),
                    self.solver.min_crossover_rate
                )
                self.stall_count = 0  # Reset stall count
        else:
            # If significant improvement, favor exploitation
            self.stall_count = 0
            self.solver.curr_mutation_rate = max(
                self.solver.curr_mutation_rate * (1 - self.solver.adaptation_rate),
                self.solver.min_mutation_rate
            )
            self.solver.curr_crossover_rate = min(
                self.solver.curr_crossover_rate * (1 + self.solver.adaptation_rate),
                self.solver.max_crossover_rate
            )
        
        self.last_best = current_best
        
        # Update algorithm parameters
        algorithm.mutation.prob = self.solver.curr_mutation_rate
        algorithm.crossover.prob = self.solver.curr_crossover_rate

class AdaptiveGeneticAlgorithmSolver(BaseSolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Initialize adaptive parameters with more conservative values
        self.initial_mutation_rate = self.solver_specific.get('initial_mutation_rate', 0.1)
        self.initial_crossover_rate = self.solver_specific.get('initial_crossover_rate', 0.8)
        self.min_mutation_rate = self.solver_specific.get('min_mutation_rate', 0.05)  # Increased minimum
        self.max_mutation_rate = self.solver_specific.get('max_mutation_rate', 0.3)   # Reduced maximum
        self.min_crossover_rate = self.solver_specific.get('min_crossover_rate', 0.6)  # Increased minimum
        self.max_crossover_rate = self.solver_specific.get('max_crossover_rate', 0.9)  # Reduced maximum
        self.adaptation_rate = self.solver_specific.get('adaptation_rate', 0.05)  # More gradual adaptation
        
        # Current rates
        self.curr_mutation_rate = self.initial_mutation_rate
        self.curr_crossover_rate = self.initial_crossover_rate
        
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm."""
        try:
            # Get solver parameters with more efficient defaults
            pop_size = self.solver_specific.get('population_size', 50)  # Reduced population
            n_gen = self.solver_specific.get('n_generations', 100)      # Reduced generations
            
            # Create problem with caching
            problem = RocketStageProblem(self, len(initial_guess), bounds)
            
            # Initialize algorithm with improved settings
            algorithm = GA(
                pop_size=pop_size,
                mutation=('real_pm', {
                    'eta': 15,  # Slightly reduced for better exploration
                    'prob': self.initial_mutation_rate
                }),
                crossover=('real_sbx', {
                    'eta': 10,  # Reduced for more exploration
                    'prob': self.initial_crossover_rate
                }),
                eliminate_duplicates=True  # Maintain diversity
            )
            
            # Create callback with optimized settings
            callback = AdaptiveCallback(self)
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
                callback=callback,
                seed=1,
                verbose=False
            )
            
            # Get best solution
            x = result.X
            payload_fraction = self.calculate_fitness(x)
            stage_ratios, stage_info = calculate_stage_ratios(
                x, self.G0, self.ISP, self.EPSILON
            )
            
            return {
                'success': True,
                'message': "Optimization completed",
                'payload_fraction': payload_fraction,
                'stages': stage_info,
                'n_iterations': n_gen,
                'n_function_evals': result.algorithm.evaluator.n_eval,
                'final_mutation_rate': self.curr_mutation_rate,
                'final_crossover_rate': self.curr_crossover_rate
            }
            
        except Exception as e:
            logger.error(f"Error in Adaptive GA solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'stages': [],
                'n_iterations': 0,
                'n_function_evals': 0
            }
