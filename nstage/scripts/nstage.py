import numpy as np
from scipy.optimize import root_scalar
from deap import base, creator, tools, algorithms

# Ensure DEAP creator components are only created once
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

def Nstage(vf, beta, epsilon, alpha, solver='newton', tol=1e-9, max_iter=100):
    def f(p):
        """ Function to find the optimal staging parameter. """
        p = np.clip(p, 1e-6, 1)  # Restrict p to valid range (avoid log(0) or negative values)
        return vf + np.sum(beta * np.log(np.maximum(epsilon + alpha * (1 - epsilon) * p, 1e-9)))

    def df(p):
        """ Derivative of the function. """
        p = np.clip(p, 1e-6, 1)  # Ensure p is valid before calculating derivative
        return np.sum(alpha * beta / np.maximum(epsilon + alpha * (1 - epsilon) * p, 1e-9))

    if solver == 'newton':
        p = 0.1
        for _ in range(max_iter):
            f_p, df_p = f(p), df(p)
            if abs(f_p) < tol:
                return p
            if abs(df_p) < 1e-12:  # Avoid division by zero
                raise ValueError("Newton's method encountered zero derivative.")
            p -= f_p / df_p
        raise ValueError("Newton's method did not converge.")

    elif solver == 'bisection':
        a, b = 0, 1
        if f(a) * f(b) > 0:
            # Dynamically expand the bracket if needed
            for _ in range(10):
                a -= 0.1
                b += 0.1
                if f(a) * f(b) < 0:
                    break
            else:
                raise ValueError("Bisection method: could not find valid bracket.")

        for _ in range(max_iter):
            p = (a + b) / 2
            f_p = f(p)
            if abs(f_p) < tol:
                return p
            if f(a) * f_p < 0:
                b = p
            else:
                a = p
        raise ValueError("Bisection method did not converge.")

    elif solver == 'secant':
        p0, p1 = 0.1, 0.9
        for _ in range(max_iter):
            f_p0, f_p1 = f(p0), f(p1)
            if abs(f_p1) < tol:
                return p1
            if abs(f_p1 - f_p0) < 1e-12:  # Avoid division by zero
                raise ValueError("Secant method encountered nearly identical function values.")
            p_new = p1 - f_p1 * (p1 - p0) / (f_p1 - f_p0)
            p0, p1 = p1, p_new
        raise ValueError("Secant method did not converge.")

    elif solver == 'scipy':
        sol = root_scalar(f, bracket=[0, 1], method='brentq', xtol=tol)
        if sol.converged:
            return sol.root
        raise ValueError("SciPy root-finding method did not converge.")

    elif solver == 'genetic':
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: (abs(f(ind[0])),))
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=100)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=False)
        return tools.selBest(pop, k=1)[0][0]

    else:
        raise ValueError("Invalid solver selection.")
