"""Adaptive Genetic Algorithm solver implementation."""
import numpy as np
import time
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem, tournament_comp
from ...utils.config import logger

class AdaptiveCallback(Callback):
    """Callback for adaptive population size and operator parameters."""
    
    def __init__(self, solver, stagnation_limit=10, min_mutation_rate=0.01, max_mutation_rate=0.5, 
                 min_crossover_rate=0.5, max_crossover_rate=1.0, elite_size=2, convergence_threshold=1e-6):
        """Initialize adaptive callback."""
        super().__init__()
        self.solver = solver
        self.stagnation_limit = int(stagnation_limit)
        self.min_mutation_rate = float(min_mutation_rate)
        self.max_mutation_rate = float(max_mutation_rate)
        self.min_crossover_rate = float(min_crossover_rate)
        self.max_crossover_rate = float(max_crossover_rate)
        self.elite_size = int(elite_size)
        self.convergence_threshold = float(convergence_threshold)
        self.stagnation_counter = 0
        self.best_fitness = -float('inf')
        self.diversity_history = []
        self.generation_history = []
        logger.debug(f"Initialized AdaptiveCallback with: stagnation_limit={stagnation_limit}, "
                    f"mutation_rate=[{min_mutation_rate}, {max_mutation_rate}], "
                    f"crossover_rate=[{min_crossover_rate}, {max_crossover_rate}]")
    
    def _get_current_best(self, algorithm):
        """Get current best fitness value."""
        if algorithm.opt is None or len(algorithm.pop) == 0:
            return -float('inf')
        pop_F = np.array([ind.F[0] for ind in algorithm.pop])
        return -float(np.min(pop_F))  # Negative since we're minimizing
        
    def calculate_diversity(self, pop):
        """Calculate population diversity using normalized standard deviation."""
        if len(pop) == 0:
            return 0.0
        X = np.array([ind.X for ind in pop])
        # Normalize by variable ranges to get consistent diversity measure
        X_std = np.std(X, axis=0)
        X_range = np.max(X, axis=0) - np.min(X, axis=0)
        X_range[X_range == 0] = 1  # Avoid division by zero
        normalized_std = X_std / X_range
        return float(np.mean(normalized_std))
    
    def _check_improvement(self, current_best):
        """Check if there's an improvement in best fitness."""
        if current_best > self.best_fitness + self.convergence_threshold:
            self.best_fitness = float(current_best)
            self.stagnation_counter = 0
            return True
        self.stagnation_counter += 1
        return False
    
    def _adapt_parameters(self, algorithm, diversity):
        """Adapt algorithm parameters based on search progress."""
        old_mutation = float(algorithm.mating.mutation.prob)
        old_crossover = float(algorithm.mating.crossover.prob)
        old_pop_size = int(algorithm.pop_size)
        
        # Adapt population size based on stagnation
        if self.stagnation_counter > self.stagnation_limit:
            new_pop_size = min(int(old_pop_size * 1.5), 500)
            if new_pop_size != old_pop_size:
                algorithm.pop_size = new_pop_size
                logger.info(f"Adapting population size: {old_pop_size} -> {new_pop_size}")
                self.stagnation_counter = 0
        
        # Adapt operator rates based on diversity
        avg_diversity = float(np.mean(self.diversity_history[-3:]) if len(self.diversity_history) > 2 else diversity)
        
        if diversity < avg_diversity:
            # Low diversity - increase exploration
            new_mutation = min(float(old_mutation * 1.2), float(self.max_mutation_rate))
            new_crossover = max(float(old_crossover * 0.9), float(self.min_crossover_rate))
        else:
            # High diversity - increase exploitation
            new_mutation = max(float(old_mutation * 0.8), float(self.min_mutation_rate))
            new_crossover = min(float(old_crossover * 1.1), float(self.max_crossover_rate))
        
        if new_mutation != old_mutation or new_crossover != old_crossover:
            algorithm.mating.mutation.prob = float(new_mutation)
            algorithm.mating.crossover.prob = float(new_crossover)
            logger.debug(f"Adapting operators: mutation {old_mutation:.3f}->{new_mutation:.3f}, "
                        f"crossover {old_crossover:.3f}->{new_crossover:.3f}")
    
    def notify(self, algorithm):
        """Adapt parameters based on optimization progress."""
        try:
            current_best = self._get_current_best(algorithm)
            diversity = self.calculate_diversity(algorithm.pop)
            
            # Store history
            self.diversity_history.append(float(diversity))
            self.generation_history.append({
                'generation': int(algorithm.n_gen),
                'best_fitness': float(current_best),
                'diversity': float(diversity),
                'pop_size': int(algorithm.pop_size),
                'mutation_rate': float(algorithm.mating.mutation.prob),
                'crossover_rate': float(algorithm.mating.crossover.prob)
            })
            
            # Log progress
            logger.debug(f"Generation {algorithm.n_gen}: Best fitness={current_best:.6f}, "
                        f"Diversity={diversity:.6f}, Stagnation={self.stagnation_counter}")
            
            if self._check_improvement(current_best):
                logger.info(f"New best solution found: {current_best:.6f}")
            
            self._adapt_parameters(algorithm, diversity)
            
        except Exception as e:
            logger.error(f"Error in adaptive callback: {str(e)}")

class AdaptiveGeneticAlgorithmSolver(BaseSolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Initialize solver parameters with defaults
        self.pop_size = int(self.solver_specific.get('population_size', 100))
        self.n_gen = int(self.solver_specific.get('n_generations', 200))
        self.tournament_size = int(self.solver_specific.get('tournament_size', 2))
        self.mutation_rate = float(self.solver_specific.get('initial_mutation_rate', 0.1))
        self.crossover_rate = float(self.solver_specific.get('initial_crossover_rate', 0.8))
        
        # Initialize adaptive parameters
        self.min_mutation_rate = float(self.solver_specific.get('min_mutation_rate', 0.01))
        self.max_mutation_rate = float(self.solver_specific.get('max_mutation_rate', 0.5))
        self.min_crossover_rate = float(self.solver_specific.get('min_crossover_rate', 0.5))
        self.max_crossover_rate = float(self.solver_specific.get('max_crossover_rate', 0.95))
        self.stagnation_limit = int(self.solver_specific.get('stagnation_limit', 10))
        self.convergence_threshold = float(self.solver_specific.get('convergence_threshold', 1e-6))
        
        logger.debug(f"Initialized {self.name} with parameters: "
                    f"pop_size={self.pop_size}, n_gen={self.n_gen}, "
                    f"mutation_rate={self.mutation_rate}, crossover_rate={self.crossover_rate}")
    
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm."""
        try:
            start_time = time.time()
            logger.info(f"Starting {self.name} optimization")
            
            # Create problem instance
            problem = RocketStageProblem(
                solver=self,
                n_var=len(initial_guess),
                bounds=bounds
            )
            
            # Initialize algorithm
            algorithm = GA(
                pop_size=self.pop_size,
                sampling=np.array(initial_guess).reshape(1, -1),
                selection=TournamentSelection(func_comp=tournament_comp),
                crossover=SBX(prob=self.crossover_rate, eta=20),
                mutation=PM(prob=self.mutation_rate, eta=20),
                eliminate_duplicates=True
            )
            
            # Create callback
            callback = AdaptiveCallback(
                solver=self,
                stagnation_limit=self.stagnation_limit,
                min_mutation_rate=self.min_mutation_rate,
                max_mutation_rate=self.max_mutation_rate,
                min_crossover_rate=self.min_crossover_rate,
                max_crossover_rate=self.max_crossover_rate,
                convergence_threshold=self.convergence_threshold
            )
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.n_gen),
                callback=callback,
                verbose=False
            )
            
            # Process results
            best_x = result.X
            best_f = float(-result.F[0])  # Convert back to maximization
            
            # Calculate final stage ratios and create results
            stage_ratios, mass_ratios = self.calculate_stage_ratios(best_x)
            stages = self.create_stage_results(best_x, stage_ratios)
            
            execution_time = time.time() - start_time
            logger.info(f"Optimization completed in {execution_time:.2f} seconds")
            logger.info(f"Best fitness achieved: {best_f:.6f}")
            
            return {
                'x': best_x.tolist(),
                'f': best_f,
                'success': True,
                'stages': stages,
                'execution_time': execution_time,
                'generations': len(callback.generation_history),
                'history': callback.generation_history
            }
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
