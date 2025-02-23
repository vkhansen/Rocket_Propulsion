# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from constants import TOTAL_DELTA_V, GRAVITY_LOSS, DRAG_LOSS, BOUNDS

class Visualizer:
    def __init__(self, solvers_results, rocket_model):
        self.results = solvers_results
        self.rocket_model = rocket_model

    def plot_execution_time(self, filename="execution_time.png"):
        solver_names = [res["method"] for res in self.results]
        times = [res["time"] for res in self.results]
        plt.figure(figsize=(10, 5))
        plt.bar(solver_names, times, color='skyblue', alpha=0.8)
        plt.xlabel("Optimization Method")
        plt.ylabel("Execution Time (s)")
        plt.title("Solver Execution Time")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_objective_error(self, filename="objective_error.png"):
        solver_names = [res["method"] for res in self.results]
        errors = [res["error"] for res in self.results]
        plt.figure(figsize=(10, 5))
        plt.bar(solver_names, errors, color='salmon', alpha=0.8)
        plt.xlabel("Optimization Method")
        plt.ylabel("Final Objective Error")
        plt.title("Solver Accuracy (Lower is Better)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_mismatch(self, filename="physical_mismatch.png"):
        solver_names = [res["method"] for res in self.results]
        mismatches = [res["mismatch"] for res in self.results]
        plt.figure(figsize=(10, 5))
        plt.bar(solver_names, mismatches, color='lightgreen', alpha=0.8)
        plt.xlabel("Optimization Method")
        plt.ylabel("ΔV Mismatch (m/s)")
        plt.title("Physical Mismatch (Produced Engine ΔV - Required Engine ΔV)")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_dv_breakdown(self, filename="dv_breakdown.png"):
        solver_names = [res["method"] for res in self.results]
        indices = np.arange(len(solver_names))
        bar_width = 0.15

        required_engine_dv = self.rocket_model.total_delta_v + GRAVITY_LOSS + DRAG_LOSS

        plt.figure(figsize=(12, 6))
        for i, res in enumerate(self.results):
            sol = res.get("solution", res["dv"])  # Fallback to dv if solution not present
            dv_per_stage = self.rocket_model.delta_v_function(sol)
            total_engine_dv = np.sum(dv_per_stage)
            plt.bar(i, dv_per_stage[0], bar_width, label='Stage 1' if i == 0 else "", color='dodgerblue')
            plt.bar(i, dv_per_stage[1], bar_width, bottom=dv_per_stage[0], label='Stage 2' if i == 0 else "", color='orange')
            plt.text(i, total_engine_dv + 50, f"{total_engine_dv:.0f}", ha='center', va='bottom', fontsize=9)
        plt.axhline(required_engine_dv, color='red', linestyle='--', linewidth=2, label='Required Engine ΔV')
        plt.xticks(indices, solver_names)
        plt.ylabel("Engine-Provided ΔV (m/s)")
        plt.title("ΔV Breakdown per Solver")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_objective_contour(self, filename="objective_contour.png"):
        f1 = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 100)
        f2 = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 100)
        F1, F2 = np.meshgrid(f1, f2)
        Z = np.zeros_like(F1)
        for i in range(F1.shape[0]):
            for j in range(F1.shape[1]):
                x_val = np.array([F1[i, j], F2[i, j]])
                Z[i, j] = self.rocket_model.objective(x_val)
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(F1, F2, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Squared Error')
        plt.xlabel("Stage 1 Mass Fraction")
        plt.ylabel("Stage 2 Mass Fraction")
        plt.title("Objective Function Landscape")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_ga_convergence(self, ga_history, filename="ga_convergence.png"):
        if not ga_history:
            print("No GA history available for convergence plot.")
            return
        generations = np.arange(len(ga_history))
        best_F = [np.min(gen) for gen in ga_history]
        plt.figure(figsize=(8, 5))
        plt.plot(generations, best_F, marker='o', linestyle='-')
        plt.xlabel("Generation")
        plt.ylabel("Best Objective Value")
        plt.title("GA Convergence Curve")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_ga_population(self, ga_history, filename="ga_population_hist.png"):
        if not ga_history:
            print("No GA history available for population distribution plot.")
            return
        final_F = ga_history[-1].flatten()
        plt.figure(figsize=(8, 5))
        plt.hist(final_F, bins=20, color='orchid', edgecolor='black', alpha=0.75)
        plt.xlabel("Objective Value")
        plt.ylabel("Frequency")
        plt.title("Final Population Objective Value Distribution (GA)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
