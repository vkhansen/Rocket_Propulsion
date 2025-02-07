# solvers.py
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from constants import X0, BOUNDS, GRAVITY_LOSS, DRAG_LOSS
from model import RocketModel

# For the GA solver (requires pymoo)
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem

class Solver:
    def __init__(self, rocket_model: RocketModel):
        self.rocket_model = rocket_model

    def solve(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

class SLSQPSolver(Solver):
    def solve(self):
        start_time = time.time()
        result = minimize(self.rocket_model.objective, X0, bounds=BOUNDS, method='SLSQP')
        elapsed_time = time.time() - start_time

        solution = result.x
        error = result.fun
        produced = np.sum(self.rocket_model.delta_v_function(solution))
        required_engine_dv = self.rocket_model.total_delta_v + GRAVITY_LOSS + DRAG_LOSS
        mismatch = produced - required_engine_dv

        return {"name": "SLSQP", "time": elapsed_time, "solution": solution, "error": error, "mismatch": mismatch}

class TrustConstrSolver(Solver):
    def solve(self):
        start_time = time.time()
        result = minimize(self.rocket_model.objective, X0, bounds=BOUNDS, method='trust-constr')
        elapsed_time = time.time() - start_time

        solution = result.x
        error = result.fun
        produced = np.sum(self.rocket_model.delta_v_function(solution))
        required_engine_dv = self.rocket_model.total_delta_v + GRAVITY_LOSS + DRAG_LOSS
        mismatch = produced - required_engine_dv

        return {"name": "Trust-Constr", "time": elapsed_time, "solution": solution, "error": error, "mismatch": mismatch}

class DifferentialEvolutionSolver(Solver):
    def solve(self):
        start_time = time.time()
        result = differential_evolution(self.rocket_model.objective, BOUNDS)
        elapsed_time = time.time() - start_time

        solution = result.x
        error = result.fun
        produced = np.sum(self.rocket_model.delta_v_function(solution))
        required_engine_dv = self.rocket_model.total_delta_v + GRAVITY_LOSS + DRAG_LOSS
        mismatch = produced - required_engine_dv

        return {"name": "Differential Evolution", "time": elapsed_time, "solution": solution, "error": error, "mismatch": mismatch}

# --- pymoo Problem class for the GA Solver ---
class StageOptimizationProblem(Problem):
    def __init__(self, rocket_model: RocketModel):
        num_stages = len(rocket_model.isp)
        xl = np.array([b[0] for b in BOUNDS])
        xu = np.array([b[1] for b in BOUNDS])
        super().__init__(n_var=num_stages, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.rocket_model = rocket_model

    def _evaluate(self, x, out, *args, **kwargs):
        required_engine_delta_v = self.rocket_model.total_delta_v + GRAVITY_LOSS + DRAG_LOSS
        # x is an array of shape (n_samples, n_var)
        produced = np.sum(self.rocket_model.delta_v_function(x), axis=1)
        f = (produced - required_engine_delta_v) ** 2
        out["F"] = f[:, np.newaxis]

class GeneticAlgorithmSolver(Solver):
    def solve(self):
        problem = StageOptimizationProblem(self.rocket_model)
        algorithm = GA(pop_size=200, eliminate_duplicates=True)
        history = []

        def callback(alg):
            # Save a copy of the current objective values from the population
            history.append(np.copy(alg.pop.get("F")))

        start_time = time.time()
        res = pymoo_minimize(problem, algorithm, termination=('n_gen', 200),
                               seed=1, verbose=True, save_history=True, callback=callback)
        elapsed_time = time.time() - start_time

        # Extract a solution (ensure we have a 1D solution vector)
        solution = res.X if res.X.ndim == 1 else res.X[0]
        error = self.rocket_model.objective(solution)
        produced = np.sum(self.rocket_model.delta_v_function(solution))
        required_engine_dv = self.rocket_model.total_delta_v + GRAVITY_LOSS + DRAG_LOSS
        mismatch = produced - required_engine_dv

        # Store history from the callback (if available)
        res.history = history if len(history) > 0 else res.history

        return {"name": "Genetic Algorithm", "time": elapsed_time, "solution": solution,
                "error": error, "mismatch": mismatch, "history": res.history}
