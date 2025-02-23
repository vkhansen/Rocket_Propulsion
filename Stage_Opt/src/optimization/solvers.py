"""Optimization solvers."""
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.repair import Repair
from pymoo.termination.default import DefaultSingleObjectiveTermination

from .objective import payload_fraction_objective, objective_with_penalty
from ..utils.config import logger
from ..utils.data import calculate_mass_ratios, calculate_payload_fraction

class DeltaVRepair(Repair):
    """Repair operator to ensure delta-v sum constraint."""
    def __init__(self, total_delta_v):
        super().__init__()
        self.total_delta_v = total_delta_v

    def _do(self, problem, X, **kwargs):
        """Repair the solution to meet the total delta-v constraint."""
        X = np.maximum(X, 0)  # Ensure non-negative values
        sums = np.sum(X, axis=1)
        scale = self.total_delta_v / sums
        X = X * scale[:, None]
        return X

class RocketOptimizationProblem(Problem):
    """Problem definition for rocket stage optimization."""
    def __init__(self, n_var, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=1,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds])
        )
        self.G0 = G0
        self.ISP = ISP
        self.EPSILON = EPSILON
        self.TOTAL_DELTA_V = TOTAL_DELTA_V

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the objective and constraints."""
        f = np.array([payload_fraction_objective(dv, self.G0, self.ISP, self.EPSILON) for dv in x])
        g = np.array([np.sum(dv) - self.TOTAL_DELTA_V for dv in x])
        out["F"] = f
        out["G"] = g

def solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Sequential Least Squares Programming (SLSQP)."""
    try:
        logger.info(f"Starting SLSQP optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        def objective(dv):
            return payload_fraction_objective(dv, G0, ISP, EPSILON)
            
        def constraint(dv):
            return float(np.sum(dv) - TOTAL_DELTA_V)
            
        constraints = {'type': 'eq', 'fun': constraint}
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'ftol': config["optimization"]["tolerance"],
                'maxiter': config["optimization"]["max_iterations"]
            }
        )
        
        if not result.success:
            logger.warning(f"SLSQP optimization warning: {result.message}")
        else:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(result.x, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"SLSQP optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result.x]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"SLSQP optimization failed: {e}")
        raise

def solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Basin-Hopping."""
    try:
        logger.info(f"Starting Basin-Hopping optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        def objective(dv):
            return objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
            
        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "options": {
                'ftol': config["optimization"]["tolerance"],
                'maxiter': config["optimization"]["max_iterations"]
            }
        }
        
        result = basinhopping(
            objective,
            initial_guess,
            minimizer_kwargs=minimizer_kwargs,
            niter=config["optimization"]["basin_hopping"]["n_iterations"],
            T=config["optimization"]["basin_hopping"]["temperature"],
            stepsize=config["optimization"]["basin_hopping"]["step_size"]
        )
        
        if not result.lowest_optimization_result.success:
            logger.warning(f"Basin-Hopping optimization warning: {result.message}")
        else:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(result.x, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"Basin-Hopping optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result.x]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"Basin-Hopping optimization failed: {e}")
        raise

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Differential Evolution."""
    try:
        logger.info(f"Starting Differential Evolution optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        def objective(dv):
            return objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
            
        result = differential_evolution(
            objective,
            bounds,
            strategy=config["optimization"]["differential_evolution"]["strategy"],
            maxiter=config["optimization"]["differential_evolution"]["max_iterations"],
            popsize=config["optimization"]["differential_evolution"]["population_size"],
            tol=config["optimization"]["differential_evolution"]["tol"],
            mutation=config["optimization"]["differential_evolution"]["mutation"],
            recombination=config["optimization"]["differential_evolution"]["recombination"]
        )
        
        if not result.success:
            logger.warning(f"Differential Evolution optimization warning: {result.message}")
        else:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(result.x, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"Differential Evolution optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result.x]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"Differential Evolution optimization failed: {e}")
        raise

def solve_with_genetic_algorithm(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Genetic Algorithm from pymoo."""
    try:
        logger.info(f"Starting Genetic Algorithm optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")

        problem = RocketOptimizationProblem(
            n_var=len(initial_guess),
            bounds=bounds,
            G0=G0,
            ISP=ISP,
            EPSILON=EPSILON,
            TOTAL_DELTA_V=TOTAL_DELTA_V
        )

        ga_config = config["optimization"]["genetic_algorithm"]
        
        algorithm = GA(
            pop_size=ga_config["population_size"],
            sampling=np.array([initial_guess]),  # Convert to 2D array for pymoo
            crossover=SBX(prob=0.9, eta=15),
            mutation=PolynomialMutation(eta=20),
            repair=DeltaVRepair(TOTAL_DELTA_V),
            eliminate_duplicates=True
        )

        termination = DefaultSingleObjectiveTermination(
            xtol=ga_config["xtol"],
            ftol=ga_config["ftol"],
            period=ga_config["period"],
            n_max_gen=ga_config["max_generations"]
        )

        result = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            verbose=True
        )

        if not result.success:
            logger.warning(f"Genetic Algorithm optimization warning: {result.message}")
        else:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(result.X, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"Genetic Algorithm optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result.X]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")

        return result.X

    except Exception as e:
        logger.error(f"Genetic Algorithm optimization failed: {e}")
        raise
