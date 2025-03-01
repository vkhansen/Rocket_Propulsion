"""Genetic Algorithm solver implementation."""
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_ga_solver import BaseGASolver

class GeneticAlgorithmSolver(BaseGASolver):
    """Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Get solver parameters from config
        self.n_generations = int(self.solver_specific.get('max_generations', 100))
        
        logger.debug(f"Initialized {self.name} with parameters: "
                    f"max_generations={self.n_generations}")

    def solve(self, initial_guess, bounds):
        """Solve using Genetic Algorithm."""
        try:
            logger.info("Starting GA optimization...")
            
            # Setup problem and algorithm
            problem = self.create_problem(initial_guess, bounds)
            algorithm = self.create_algorithm()
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.n_generations),  
                seed=1,
                verbose=False
            )
            
            # Process results
            success = res.success if hasattr(res, 'success') else True
            message = res.message if hasattr(res, 'message') else ""
            x = res.X if hasattr(res, 'X') else initial_guess
            n_gen = res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else 0
            n_eval = res.algorithm.evaluator.n_eval if hasattr(res.algorithm, 'evaluator') else 0
            
            return self.process_results(
                x=x,
                success=success,
                message=message,
                n_iterations=n_gen,
                n_function_evals=n_eval,
                time=0.0  # Time not tracked by pymoo
            )
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
