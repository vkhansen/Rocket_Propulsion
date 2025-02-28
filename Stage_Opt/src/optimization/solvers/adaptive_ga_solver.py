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
        try:
            # Only adapt parameters periodically
            self.generation_counter += 1
            if self.generation_counter % self.adaptation_interval != 0:
                return
                
            # Get current best fitness
            if algorithm.opt is None:
                return
                
            curr_best = -float(algorithm.opt.get("F")[0])  # Negative because we minimize
            
            # Check for improvement
            if curr_best > self.last_best:
                # Progress is good, reduce exploration
                self.solver.curr_mutation_rate = max(
                    self.solver.min_mutation_rate,
                    self.solver.curr_mutation_rate - self.solver.adaptation_rate
                )
                self.solver.curr_crossover_rate = min(
                    self.solver.max_crossover_rate,
                    self.solver.curr_crossover_rate + self.solver.adaptation_rate
                )
                self.stall_count = 0
            else:
                # No improvement, increase exploration
                self.solver.curr_mutation_rate = min(
                    self.solver.max_mutation_rate,
                    self.solver.curr_mutation_rate + self.solver.adaptation_rate
                )
                self.solver.curr_crossover_rate = max(
                    self.solver.min_crossover_rate,
                    self.solver.curr_crossover_rate - self.solver.adaptation_rate
                )
                self.stall_count += 1
            
            # Update algorithm parameters
            algorithm.prob_mut = self.solver.curr_mutation_rate
            algorithm.prob_cx = self.solver.curr_crossover_rate
            
            # Update last best
            self.last_best = curr_best
            
        except Exception as e:
            logger.error(f"Error in adaptive callback: {str(e)}")

class AdaptiveGeneticAlgorithmSolver(BaseSolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Initialize adaptive parameters
        self.initial_mutation_rate = self.solver_specific.get('initial_mutation_rate', 0.1)
        self.initial_crossover_rate = self.solver_specific.get('initial_crossover_rate', 0.8)
        self.min_mutation_rate = self.solver_specific.get('min_mutation_rate', 0.05)
        self.max_mutation_rate = self.solver_specific.get('max_mutation_rate', 0.3)
        self.min_crossover_rate = self.solver_specific.get('min_crossover_rate', 0.6)
        self.max_crossover_rate = self.solver_specific.get('max_crossover_rate', 0.9)
        self.adaptation_rate = self.solver_specific.get('adaptation_rate', 0.05)
        
        # Current rates
        self.curr_mutation_rate = self.initial_mutation_rate
        self.curr_crossover_rate = self.initial_crossover_rate
        
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Starting Adaptive GA optimization...")
            
            # Get solver parameters
            pop_size = self.solver_specific.get('population_size', 100)
            n_gen = self.solver_specific.get('n_generations', 100)
            
            # Initialize problem
            problem = RocketStageProblem(self, len(initial_guess), bounds)
            
            # Initialize algorithm with adaptive parameters
            algorithm = GA(
                pop_size=pop_size,
                prob_mut=self.curr_mutation_rate,
                prob_cx=self.curr_crossover_rate,
                eliminate_duplicates=True
            )
            
            # Add adaptive callback
            callback = AdaptiveCallback(self)
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
                callback=callback,
                seed=42,
                verbose=False
            )
            
            # Process results
            if result.success:
                x = result.X
                stage_ratios, mass_ratios = self.calculate_stage_ratios(x)
                payload_fraction = self.calculate_fitness(x)
                
                return {
                    'success': True,
                    'x': x.tolist(),
                    'fun': float(result.F[0]),
                    'payload_fraction': float(payload_fraction),
                    'stage_ratios': stage_ratios.tolist(),
                    'mass_ratios': mass_ratios.tolist(),
                    'stages': self.create_stage_results(x, stage_ratios),
                    'n_iterations': result.algorithm.n_iter,
                    'n_function_evals': result.algorithm.evaluator.n_eval,
                    'final_mutation_rate': self.curr_mutation_rate,
                    'final_crossover_rate': self.curr_crossover_rate
                }
            else:
                return {
                    'success': False,
                    'message': "Adaptive GA optimization failed to converge"
                }
            
        except Exception as e:
            logger.error(f"Error in Adaptive GA solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error in Adaptive GA solver: {str(e)}"
            }
