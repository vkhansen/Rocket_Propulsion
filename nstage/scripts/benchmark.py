import time
import numpy as np
from nstage import Nstage

vf = 3000
beta = np.array([0.5, 0.3, 0.2])
epsilon = np.array([0.08, 0.10, 0.15])
alpha = np.array([1.1, 1.2, 1.3])
solvers = ['newton', 'bisection', 'secant', 'scipy', 'genetic']
results = {}

for solver in solvers:
    start_time = time.time()
    try:
        p_opt = Nstage(vf, beta, epsilon, alpha, solver=solver)
        elapsed_time = time.time() - start_time
        results[solver] = (p_opt, elapsed_time)
        print(f"{solver.capitalize()} Method: p_opt = {p_opt:.6f}, Time: {elapsed_time:.6f} sec")
    except ValueError as e:
        print(f"{solver.capitalize()} Method: Failed to converge.")
