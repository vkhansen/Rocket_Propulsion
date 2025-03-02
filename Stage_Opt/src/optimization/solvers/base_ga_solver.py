"""Base Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.optimize import minimize
from ...utils.config import logger, setup_logging
from ..cache import OptimizationCache
from .base_solver import BaseSolver
from ..pymoo_problem import RocketStageProblem

class BaseGASolver(BaseSolver):
    """Base class for GA-based solvers."""
    
    def __init__(self, config, problem_params):
        """Initialize base GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Common GA parameters with consistent naming
        self.pop_size = int(self.solver_specific.get('pop_size', 100))
        self.n_generations = int(self.solver_specific.get('n_generations', 100))
        self.mutation_rate = float(self.solver_specific.get('mutation_rate', 0.1))
        self.crossover_rate = float(self.solver_specific.get('crossover_rate', 0.9))
        self.tournament_size = int(self.solver_specific.get('tournament_size', 3))
        self.eta_crossover = float(self.solver_specific.get('eta_crossover', 30))
        self.eta_mutation = float(self.solver_specific.get('eta_mutation', 30))
        
        # Add timeout parameter
        self.max_time = float(self.solver_specific.get('max_time', 3600))  # Default 1 hour
        
        self.logger = setup_logging(self.__class__.__name__)
        self.logger.info(f"Initialized {self.name} with parameters:")
        self.logger.info(f"  Population Size: {self.pop_size}")
        self.logger.info(f"  Generations: {self.n_generations}")
        self.logger.info(f"  Mutation Rate: {self.mutation_rate}")
        self.logger.info(f"  Crossover Rate: {self.crossover_rate}")
        self.logger.info(f"  Tournament Size: {self.tournament_size}")
        self.logger.info(f"  Max Time: {self.max_time} seconds")

    def create_tournament_selection(self):
        """Create tournament selection operator with comparison function."""
        def tournament_comp(pop, P, **kwargs):
            """Tournament selection comparator."""
            n_tournaments = P.shape[0]
            S = np.zeros(n_tournaments, dtype=int)
            
            for i in range(n_tournaments):
                tournament = P[i]
                candidates = []
                for idx in tournament:
                    # Handle None cases and missing attributes
                    if pop[idx] is None or not hasattr(pop[idx], "get"):
                        continue
                        
                    F = pop[idx].get("F")
                    if F is None or len(F) == 0:
                        continue
                        
                    f = F[0]
                    
                    # Handle constraint violations
                    G = pop[idx].get("G")
                    cv = np.sum(np.maximum(G, 0)) if G is not None else float('inf')
                    candidates.append((idx, f, cv))
                
                if not candidates:  # If no valid candidates
                    S[i] = tournament[0]  # Take first individual
                    continue
                    
                # Sort by constraint violation first, then by fitness
                candidates.sort(key=lambda x: (x[2], x[1]))
                S[i] = candidates[0][0]  # Select the best candidate
            
            return S

        return TournamentSelection(
            pressure=self.tournament_size,
            func_comp=tournament_comp
        )

    def create_algorithm(self, pop_size=None):
        """Create GA algorithm with specified parameters."""
        if pop_size is None:
            pop_size = self.pop_size
            
        algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.crossover_rate, eta=self.eta_crossover),
            mutation=PM(prob=self.mutation_rate, eta=self.eta_mutation),
            selection=self.create_tournament_selection(),
            eliminate_duplicates=True
        )
        
        # Add callback for monitoring
        def callback(algorithm):
            try:
                gen = algorithm.n_gen
                pop = algorithm.pop
                
                if gen % 10 == 0:  # Log every 10 generations
                    if pop is None:
                        self.logger.warning("Population is None")
                        return
                        
                    # Count valid solutions with proper error handling
                    valid_solutions = 0
                    for ind in pop:
                        if ind is None or not hasattr(ind, "get"):
                            continue
                        G = ind.get("G")
                        if G is not None and np.all(G <= 0):
                            valid_solutions += 1
                            
                    best_cv = float('inf')
                    best_f = float('inf')
                    
                    for ind in pop:
                        if ind is None or not hasattr(ind, "get"):
                            continue
                            
                        G = ind.get("G")
                        F = ind.get("F")
                        
                        if G is not None:
                            cv = np.sum(np.maximum(G, 0))
                        else:
                            cv = float('inf')
                            
                        if F is not None and len(F) > 0:
                            f = F[0]
                        else:
                            continue
                            
                        if cv < best_cv or (cv == best_cv and f < best_f):
                            best_cv = cv
                            best_f = f
                    
                    self.logger.info(f"Generation {gen}:")
                    self.logger.info(f"  Valid Solutions: {valid_solutions}/{len(pop)}")
                    self.logger.info(f"  Best CV: {best_cv:.6f}")
                    self.logger.info(f"  Best F: {best_f:.6f}")
                    
            except Exception as e:
                self.logger.error(f"Error in callback: {str(e)}")
        
        algorithm.callback = callback
        return algorithm

    def create_problem(self, initial_guess, bounds):
        """Create optimization problem."""
        n_var = len(initial_guess)
        bounds = np.array(bounds)
        return RocketStageProblem(
            solver=self,
            n_var=n_var,
            bounds=bounds
        )

    def solve(self, initial_guess, bounds):
        """Base solve method with improved error handling and logging."""
        try:
            self.logger.info("Starting optimization")
            self.logger.info(f"Initial guess: {initial_guess}")
            self.logger.info(f"Bounds: {bounds}")
            
            problem = self.create_problem(initial_guess, bounds)
            algorithm = self.create_algorithm()
            
            result = minimize(
                problem,
                algorithm,
                termination=('n_gen', self.n_generations),
                seed=42,  # For reproducibility
                verbose=False,  # We handle our own logging
                save_history=True
            )
            
            if result.success:
                self.logger.info("Optimization completed successfully")
                self.logger.info(f"Best solution X: {result.X}")
                self.logger.info(f"Best fitness F: {result.F[0]}")
                if hasattr(result, 'G'):
                    self.logger.info(f"Constraint violations G: {result.G}")
            else:
                self.logger.error("Optimization failed")
                self.logger.error(f"Termination message: {result.message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Critical error in solve method: {str(e)}", exc_info=True)
            raise
