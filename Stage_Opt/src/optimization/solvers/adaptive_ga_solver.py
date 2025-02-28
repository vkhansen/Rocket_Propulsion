"""Adaptive Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from .ga_solver import RocketStageProblem
from .base_solver import BaseSolver
from utils.config import logger

class AdaptiveCallback(Callback):
    """Callback for adaptive parameter tuning."""
    
    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self.last_best = float('-inf')
        self.stall_count = 0
        
    def notify(self, algorithm):
        """Update mutation and crossover rates based on progress."""
        curr_best = -algorithm.opt.get("F")[0]
        
        if curr_best > self.last_best:
            # Improvement found, reduce mutation rate
            self.solver.curr_mutation_rate = max(
                self.solver.min_mutation_rate,
                self.solver.curr_mutation_rate * (1 - self.solver.adaptation_rate)
            )
            self.stall_count = 0
        else:
            # No improvement, increase mutation rate
            self.solver.curr_mutation_rate = min(
                self.solver.max_mutation_rate,
                self.solver.curr_mutation_rate * (1 + self.solver.adaptation_rate)
            )
            self.stall_count += 1
            
        # Update crossover rate inversely to mutation rate
        self.solver.curr_crossover_rate = min(
            self.solver.max_crossover_rate,
            1 - self.solver.curr_mutation_rate
        )
        
        # Update algorithm parameters
        algorithm.mutation.prob = self.solver.curr_mutation_rate
        algorithm.crossover.prob = self.solver.curr_crossover_rate
        
        self.last_best = curr_best

class AdaptiveGASolver(BaseSolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Initialize adaptive parameters
        self.initial_mutation_rate = self.solver_specific.get('initial_mutation_rate', 0.1)
        self.initial_crossover_rate = self.solver_specific.get('initial_crossover_rate', 0.8)
        self.min_mutation_rate = self.solver_specific.get('min_mutation_rate', 0.01)
        self.max_mutation_rate = self.solver_specific.get('max_mutation_rate', 0.5)
        self.min_crossover_rate = self.solver_specific.get('min_crossover_rate', 0.5)
        self.max_crossover_rate = self.solver_specific.get('max_crossover_rate', 0.95)
        self.adaptation_rate = self.solver_specific.get('adaptation_rate', 0.1)
        
        # Current rates
        self.curr_mutation_rate = self.initial_mutation_rate
        self.curr_crossover_rate = self.initial_crossover_rate
        
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm."""
        try:
            # Get solver parameters
            pop_size = self.solver_specific.get('population_size', 100)
            n_gen = self.solver_specific.get('n_generations', 200)
            
            # Create problem
            problem = RocketStageProblem(self, len(initial_guess), bounds)
            
            # Initialize algorithm
            algorithm = GA(
                pop_size=pop_size,
                mutation=('real_pm', {
                    'eta': 20,
                    'prob': self.initial_mutation_rate
                }),
                crossover=('real_sbx', {
                    'eta': 15,
                    'prob': self.initial_crossover_rate
                })
            )
            
            # Create callback
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
