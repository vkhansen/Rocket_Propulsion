"""Adaptive Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from ...utils.config import logger
from ..objective import objective_with_penalty
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem, tournament_comp

class AdaptiveGeneticAlgorithmSolver(BaseSolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Initialize adaptive parameters
        self.min_pop_size = int(self.solver_specific.get('min_pop_size', 50))
        self.max_pop_size = int(self.solver_specific.get('max_pop_size', 200))
        self.min_gen = int(self.solver_specific.get('min_gen', 50))
        self.max_gen = int(self.solver_specific.get('max_gen', 500))
        self.improvement_threshold = float(self.solver_specific.get('improvement_threshold', 0.01))
        self.adaptation_interval = int(self.solver_specific.get('adaptation_interval', 10))
        
        logger.debug(f"Initialized {self.name} with parameters: "
                    f"min_pop_size={self.min_pop_size}, max_pop_size={self.max_pop_size}, "
                    f"min_gen={self.min_gen}, max_gen={self.max_gen}")
    
    def adapt_parameters(self, algorithm, history):
        """Adapt algorithm parameters based on performance."""
        if len(history) < 2:
            return
            
        # Calculate improvement
        current = history[-1]
        previous = history[-2]
        improvement = (previous - current) / abs(previous)
        
        # Adapt population size
        if improvement < self.improvement_threshold:
            algorithm.pop_size = min(
                self.max_pop_size,
                int(algorithm.pop_size * 1.5)
            )
        else:
            algorithm.pop_size = max(
                self.min_pop_size,
                int(algorithm.pop_size * 0.8)
            )
            
        # Adapt genetic operators
        if improvement < self.improvement_threshold:
            algorithm.crossover.prob = min(0.95, algorithm.crossover.prob * 1.1)
            algorithm.mutation.prob = min(0.2, algorithm.mutation.prob * 1.2)
        else:
            algorithm.crossover.prob = max(0.5, algorithm.crossover.prob * 0.9)
            algorithm.mutation.prob = max(0.05, algorithm.mutation.prob * 0.8)
    
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm."""
        try:
            logger.info(f"Starting {self.name} optimization")
            
            # Setup problem
            n_var = len(initial_guess)
            problem = RocketStageProblem(
                solver=self,
                n_var=n_var,
                bounds=bounds,
                objective_func=objective_with_penalty
            )
            
            # Initialize algorithm
            algorithm = GA(
                pop_size=self.min_pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=30),
                mutation=PM(prob=0.1, eta=30),
                eliminate_duplicates=True,
                selection=tournament_comp
            )
            
            # Initialize history
            history = []
            best_f = float('inf')
            
            # Run optimization with adaptation
            for gen in range(self.max_gen):
                res = minimize(
                    problem,
                    algorithm,
                    seed=1,
                    verbose=False
                )
                
                current_f = float(res.F[0])
                history.append(current_f)
                
                # Update best solution
                if current_f < best_f:
                    best_f = current_f
                    best_x = res.X
                
                # Adapt parameters periodically
                if gen % self.adaptation_interval == 0:
                    self.adapt_parameters(algorithm, history)
                    logger.debug(f"Generation {gen}: Adapted parameters - "
                               f"pop_size={algorithm.pop_size}, "
                               f"crossover_prob={algorithm.crossover.prob:.2f}, "
                               f"mutation_prob={algorithm.mutation.prob:.2f}")
                
                # Check convergence
                if gen >= self.min_gen and len(history) >= 10:
                    recent_improvement = (history[-10] - history[-1]) / abs(history[-10])
                    if recent_improvement < self.improvement_threshold:
                        logger.info(f"Converged after {gen} generations")
                        break
            
            return self.process_results(
                x=best_x,
                success=True,
                message=f"Optimization completed after {gen} generations",
                n_iterations=gen,
                n_function_evals=res.algorithm.evaluator.n_eval,
                time=0.0  # Time not tracked by pymoo
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
