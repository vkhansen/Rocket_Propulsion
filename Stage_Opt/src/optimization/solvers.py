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

def solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Sequential Least Squares Programming (SLSQP)."""
    try:
        logger.info(f"Starting SLSQP optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"Bounds: {bounds}")
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
            logger.info(f"SLSQP optimization succeeded with x: {result.x}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"SLSQP optimization failed: {e}")
        raise

def solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Basin-Hopping."""
    try:
        logger.info(f"Starting Basin-Hopping optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"Bounds: {bounds}")
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
            logger.info(f"Basin-Hopping optimization succeeded with x: {result.x}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"Basin-Hopping optimization failed: {e}")
        raise

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Differential Evolution."""
    try:
        logger.info(f"Starting Differential Evolution optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"Bounds: {bounds}")
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
            logger.info(f"Differential Evolution optimization succeeded with x: {result.x}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"Differential Evolution optimization failed: {e}")
        raise
