"""Adaptive Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from ...utils.config import logger
from .base_ga_solver import BaseGASolver

class AdaptiveGeneticAlgorithmSolver(BaseGASolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)
        
        # Initialize adaptive parameters with consistent naming
        self.min_pop_size = int(self.solver_specific.get('min_pop_size', 50))
        self.max_pop_size = int(self.solver_specific.get('max_pop_size', 200))
        self.min_generations = int(self.solver_specific.get('min_generations', 50))
        self.max_generations = int(self.solver_specific.get('max_generations', 500))
        self.improvement_threshold = float(self.solver_specific.get('improvement_threshold', 0.01))
        self.adaptation_interval = int(self.solver_specific.get('adaptation_interval', 10))
        
        # Mutation and crossover rate adaptation parameters
        self.min_mutation_rate = float(self.solver_specific.get('min_mutation_rate', 0.01))
        self.max_mutation_rate = float(self.solver_specific.get('max_mutation_rate', 0.2))
        self.min_crossover_rate = float(self.solver_specific.get('min_crossover_rate', 0.5))
        self.max_crossover_rate = float(self.solver_specific.get('max_crossover_rate', 1.0))
        
        # Override base parameters for initial values
        self.pop_size = int(self.solver_specific.get('initial_pop_size', 100))
        self.mutation_rate = float(self.solver_specific.get('initial_mutation_rate', 0.1))
        self.crossover_rate = float(self.solver_specific.get('initial_crossover_rate', 0.9))
        
        # Initialize tracking variables
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.adaptation_history = []
        
        logger.debug(f"Initialized {self.name} with adaptive parameters: "
                    f"pop_size=[{self.min_pop_size}, {self.max_pop_size}], "
                    f"generations=[{self.min_generations}, {self.max_generations}], "
                    f"mutation_rate=[{self.min_mutation_rate}, {self.max_mutation_rate}], "
                    f"crossover_rate=[{self.min_crossover_rate}, {self.max_crossover_rate}]")
    
    def _log_generation_stats(self, algorithm):
        """Log statistics for current generation."""
        pop = algorithm.pop
        fitness_values = np.array([ind.F[0] for ind in pop])
        best_fitness = np.min(fitness_values)
        avg_fitness = np.mean(fitness_values)
        diversity = np.std(fitness_values)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        
        # Calculate improvement percentage
        if len(self.best_fitness_history) > 1:
            improvement = ((self.best_fitness_history[-2] - best_fitness) / 
                         self.best_fitness_history[-2] * 100)
        else:
            improvement = 0.0
            
        logger.info(f"Generation {algorithm.n_gen}/{self.max_generations}:")
        logger.info(f"  Best Fitness: {best_fitness:.6f}")
        logger.info(f"  Avg Fitness: {avg_fitness:.6f}")
        logger.info(f"  Population Diversity: {diversity:.6f}")
        logger.info(f"  Improvement: {improvement:+.2f}%")
        logger.info(f"  Population Size: {algorithm.pop_size}")
        logger.info(f"  Mutation Rate: {self.mutation_rate:.3f}")
        logger.info(f"  Crossover Rate: {self.crossover_rate:.3f}")
        
        # Print convergence warning if diversity is too low
        if diversity < 1e-6:
            logger.warning("  Low population diversity detected - possible premature convergence")
            
        # Print stagnation warning if no improvement for many generations
        if len(self.best_fitness_history) > 10:
            recent_improvement = (self.best_fitness_history[-10] - best_fitness) / self.best_fitness_history[-10]
            if abs(recent_improvement) < 1e-6:
                logger.warning("  Optimization appears to be stagnating - will adapt parameters more aggressively")
    
    def adapt_parameters(self, algorithm):
        """Adapt algorithm parameters based on performance."""
        if not hasattr(algorithm, 'pop') or algorithm.pop is None:
            return
            
        # Get current population fitness values
        fitness_values = algorithm.pop.get("F")
        if fitness_values is None or len(fitness_values) == 0:
            return
            
        current_best = np.min(fitness_values)
        if not hasattr(self, '_prev_best'):
            self._prev_best = current_best
            return
            
        # Calculate improvement
        improvement = (self._prev_best - current_best) / self._prev_best if self._prev_best != 0 else 0
        self._prev_best = current_best
        
        # Store previous parameters for logging
        prev_pop_size = algorithm.pop_size
        prev_mutation_rate = self.mutation_rate
        prev_crossover_rate = self.crossover_rate
        
        # Adjust population size based on improvement
        new_pop_size = algorithm.pop_size
        if improvement < self.improvement_threshold:
            new_pop_size = min(self.max_pop_size, int(algorithm.pop_size * 1.5))
        else:
            new_pop_size = max(self.min_pop_size, int(algorithm.pop_size * 0.8))
            
        # Adjust mutation and crossover rates
        if improvement < self.improvement_threshold:
            # Increase exploration
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.2)
            self.crossover_rate = max(self.min_crossover_rate, self.crossover_rate * 0.9)
        else:
            # Increase exploitation
            self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.9)
            self.crossover_rate = min(self.max_crossover_rate, self.crossover_rate * 1.1)
        
        # Create new algorithm with adapted parameters if population size changed
        if new_pop_size != algorithm.pop_size:
            new_algorithm = self.create_algorithm(pop_size=new_pop_size)
            if hasattr(new_algorithm, 'pop') and new_algorithm.pop is not None:
                algorithm.pop = new_algorithm.pop
                algorithm.pop_size = new_pop_size
            
        # Update operators if they exist
        if hasattr(algorithm, 'mating'):
            if hasattr(algorithm.mating, 'mutation'):
                algorithm.mating.mutation.prob = self.mutation_rate
            if hasattr(algorithm.mating, 'crossover'):
                algorithm.mating.crossover.prob = self.crossover_rate
        
        # Log adaptation details
        logger.info("\nParameter Adaptation:")
        logger.info(f"  Improvement: {improvement:.6f}")
        logger.info(f"  Population Size: {prev_pop_size} -> {new_pop_size}")
        logger.info(f"  Mutation Rate: {prev_mutation_rate:.3f} -> {self.mutation_rate:.3f}")
        logger.info(f"  Crossover Rate: {prev_crossover_rate:.3f} -> {self.crossover_rate:.3f}")
        
        # Store adaptation history
        self.adaptation_history.append({
            'generation': algorithm.n_gen,
            'improvement': improvement,
            'pop_size_change': new_pop_size - prev_pop_size,
            'mutation_rate_change': self.mutation_rate - prev_mutation_rate,
            'crossover_rate_change': self.crossover_rate - prev_crossover_rate
        })
    
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm."""
        try:
            logger.info(f"\nStarting {self.name} optimization...")
            logger.info(f"Initial population size: {self.pop_size}")
            logger.info(f"Maximum generations: {self.max_generations}")
            logger.info(f"Initial mutation rate: {self.mutation_rate}")
            logger.info(f"Initial crossover rate: {self.crossover_rate}")
            logger.info(f"Adaptation interval: {self.adaptation_interval}")
            logger.info("=" * 50)
            
            # Setup problem and algorithm
            problem = self.create_problem(initial_guess, bounds)
            algorithm = self.create_algorithm(pop_size=self.pop_size)
            
            def callback(algorithm):
                """Callback to track progress and adapt parameters."""
                self._log_generation_stats(algorithm)
                if algorithm.n_gen > 0 and algorithm.n_gen % self.adaptation_interval == 0:
                    self.adapt_parameters(algorithm)
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.max_generations),
                seed=1,
                callback=callback,
                verbose=False
            )
            
            # Process results
            success = res.success if hasattr(res, 'success') else True
            message = res.message if hasattr(res, 'message') else ""
            x = res.X if hasattr(res, 'X') else initial_guess
            n_gen = res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else 0
            n_eval = res.algorithm.evaluator.n_eval if hasattr(res.algorithm, 'evaluator') else 0
            
            # Log final statistics and adaptation summary
            logger.info("\nOptimization completed:")
            logger.info(f"  Number of generations: {n_gen}")
            logger.info(f"  Number of evaluations: {n_eval}")
            logger.info(f"  Final best fitness: {res.F[0]:.6f}")
            logger.info(f"  Success: {success}")
            if message:
                logger.info(f"  Message: {message}")
            
            logger.info("\nParameter Adaptation Summary:")
            logger.info(f"  Number of adaptations: {len(self.adaptation_history)}")
            if self.adaptation_history:
                avg_pop_change = np.mean([h['pop_size_change'] for h in self.adaptation_history])
                avg_mut_change = np.mean([h['mutation_rate_change'] for h in self.adaptation_history])
                avg_cross_change = np.mean([h['crossover_rate_change'] for h in self.adaptation_history])
                logger.info(f"  Average population size change: {avg_pop_change:.1f}")
                logger.info(f"  Average mutation rate change: {avg_mut_change:.3f}")
                logger.info(f"  Average crossover rate change: {avg_cross_change:.3f}")
            logger.info("=" * 50)
            
            return self.process_results(
                x=x,
                success=success,
                message=message,
                n_iterations=n_gen,
                n_function_evals=n_eval,
                time=0.0
            )
            
        except Exception as e:
            logger.error(f"Error in Adaptive GA solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
