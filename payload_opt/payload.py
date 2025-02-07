import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem

# Constants
G0 = 9.81  # gravitational acceleration (m/s²)
ISP = np.array([282, 348])  # specific impulse for each stage (s)
EPSILON = np.array([0.03, 0.07])  # structural mass fractions for each stage
TOTAL_DELTA_V = 10500  # required ΔV (m/s)

# Payload Optimization Function
def payload_fraction_objective(DV1):
    DV2 = TOTAL_DELTA_V - DV1
    f1 = np.exp(-DV1 / (G0 * ISP[0])) - EPSILON[0]
    f2 = np.exp(-DV2 / (G0 * ISP[1])) - EPSILON[1]
    if f1 <= 0 or f2 <= 0:
        return 1e6  # Penalty for infeasible split
    return -f1 * f2  # Maximize payload fraction

# Solver Implementations
class StageOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=0, xu=TOTAL_DELTA_V)
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([payload_fraction_objective(xi) for xi in x])[:, np.newaxis]

solvers = {}

# Scalar Minimization
res_scalar = minimize_scalar(payload_fraction_objective, bounds=(0, TOTAL_DELTA_V), method='bounded')
solvers['Scalar Minimization'] = res_scalar

# Differential Evolution
res_de = differential_evolution(payload_fraction_objective, bounds=[(0, TOTAL_DELTA_V)])
solvers['Differential Evolution'] = res_de

# Genetic Algorithm (pymoo)
ga_problem = StageOptimizationProblem()
ga_algorithm = GA(pop_size=200, eliminate_duplicates=True)
res_ga = pymoo_minimize(ga_problem, ga_algorithm, termination=('n_gen', 200), seed=1, verbose=False)
solvers['Genetic Algorithm'] = res_ga

# Extract Optimal Results
optimal_DV1 = {name: sol.x if isinstance(sol, dict) else sol.x[0] for name, sol in solvers.items()}
optimal_DV2 = {name: TOTAL_DELTA_V - dv1 for name, dv1 in optimal_DV1.items()}
optimal_payload_fraction = {name: -payload_fraction_objective(dv1) for name, dv1 in optimal_DV1.items()}

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(optimal_payload_fraction.keys(), optimal_payload_fraction.values(), color='blue', alpha=0.7)
plt.xlabel("Solver")
plt.ylabel("Payload Fraction")
plt.title("Optimized Payload Fraction for Different Solvers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ΔV Allocation Plot
plt.figure(figsize=(10, 5))
for name in optimal_DV1.keys():
    plt.bar(name, optimal_DV1[name], color='blue', alpha=0.6, label='Stage 1 ΔV' if name == list(optimal_DV1.keys())[0] else "")
    plt.bar(name, optimal_DV2[name], color='orange', alpha=0.6, bottom=optimal_DV1[name], label='Stage 2 ΔV' if name == list(optimal_DV1.keys())[0] else "")
plt.xlabel("Solver")
plt.ylabel("ΔV Allocation (m/s)")
plt.title("ΔV Allocation Across Solvers")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
