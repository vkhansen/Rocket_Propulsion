"""Base Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem

class BaseGASolver(BaseSolver):
    """Base class for GA-based solvers."""
    
    def __init__(self, config, problem_params):
        """Initialize base GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Common GA parameters
        self.pop_size = int(self.solver_specific.get('pop_size', 100))
        self.max_generations = int(self.solver_specific.get('max_generations', 100))
        self.mutation_rate = float(self.solver_specific.get('mutation_rate', 0.1))
        self.crossover_rate = float(self.solver_specific.get('crossover_rate', 0.9))
        self.tournament_size = int(self.solver_specific.get('tournament_size', 3))
        
        logger.debug(f"Initialized {self.name} with parameters: "
                    f"pop_size={self.pop_size}, max_generations={self.max_generations}")

    def create_tournament_selection(self):
        """Create tournament selection operator with comparison function."""
        def tournament_comp(pop, P, **kwargs):
            """Tournament selection comparator."""
            # P is a 2D array with shape (n_tournaments, tournament_size)
            n_tournaments = P.shape[0]
            S = np.zeros(n_tournaments, dtype=int)
            
            for i in range(n_tournaments):
                tournament = P[i]  # Get indices for this tournament
                # Get fitness values for individuals in tournament
                fitness = np.array([pop[j].get("F")[0] for j in tournament])
                # Get constraint violations
                cv = np.array([pop[j].get("CV")[0] if pop[j].get("CV") is not None else 0.0 for j in tournament])
                
                # Select winner based on constraint violation and fitness
                if np.any(cv > 0):
                    winner_idx = tournament[np.argmin(cv)]
                else:
                    winner_idx = tournament[np.argmin(fitness)]
                    
                S[i] = winner_idx
            
            return S

        return TournamentSelection(
            pressure=self.tournament_size,
            func_comp=tournament_comp
        )

    def create_algorithm(self, pop_size=None):
        """Create GA algorithm with specified parameters."""
        if pop_size is None:
            pop_size = self.pop_size
            
        return GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.crossover_rate, eta=30),
            mutation=PM(prob=self.mutation_rate, eta=30),
            selection=self.create_tournament_selection(),
            eliminate_duplicates=True
        )

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
        """Base solve method - override in subclasses."""
        raise NotImplementedError("Solve method must be implemented by subclass")
