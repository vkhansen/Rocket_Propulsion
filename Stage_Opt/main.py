# main.py
from constants import ISP, TOTAL_DELTA_V, GRAVITY_LOSS, DRAG_LOSS, G0, EPSILON, PAYLOAD_FRACTION, BOUNDS, X0
from model import RocketModel
from solvers import SLSQPSolver, TrustConstrSolver, DifferentialEvolutionSolver, GeneticAlgorithmSolver
from visualization import Visualizer
from report import ReportGenerator

def main():
    # Initialize the rocket model with given ISP and total ΔV
    rocket_model = RocketModel(ISP, TOTAL_DELTA_V)

    # Create solver instances
    solvers = [
        SLSQPSolver(rocket_model),
        TrustConstrSolver(rocket_model),
        DifferentialEvolutionSolver(rocket_model),
        GeneticAlgorithmSolver(rocket_model)
    ]

    # Run solvers and collect results
    results = []
    for solver in solvers:
        print(f"\nRunning solver: {solver.__class__.__name__}")
        res = solver.solve()
        print(f"{res['name']} -> Time: {res['time']:.4f}s | Final Error: {res['error']:.2e} | "
              f"Mismatch: {res['mismatch']:.2f} | Solution: {res['solution']}")
        results.append(res)

    # Generate visualizations
    visualizer = Visualizer(results, rocket_model)
    visualizer.plot_execution_time()
    visualizer.plot_objective_error()
    visualizer.plot_mismatch()
    visualizer.plot_dv_breakdown()
    visualizer.plot_objective_contour()

    # If available, generate GA-specific plots
    ga_history = None
    for res in results:
        if res["name"] == "Genetic Algorithm" and "history" in res:
            ga_history = res["history"]
            break

    if ga_history:
        visualizer.plot_ga_convergence(ga_history)
        visualizer.plot_ga_population(ga_history)

    # Prepare data for LaTeX report
    input_data = {
        "g0 (m/s²)": G0,
        "ISP (Stage 1, Stage 2)": ISP.tolist(),
        "EPSILON (Stage 1, Stage 2)": EPSILON.tolist(),
        "Payload Fraction": PAYLOAD_FRACTION,
        "Total ΔV (m/s)": TOTAL_DELTA_V,
        "Gravity Loss (m/s)": GRAVITY_LOSS,
        "Drag Loss (m/s)": DRAG_LOSS,
        "Bounds": str(BOUNDS),
        "Initial Guess": X0.tolist()
    }

    assumptions = r"""
\begin{enumerate}
    \item The rocket equation is applied in its simplified form.
    \item Structural mass fractions and payload fraction are assumed constant.
    \item The design space is bounded as specified.
    \item The objective function minimizes the squared error between produced and required ΔV.
\end{enumerate}
"""

    plot_filenames = {
        "Execution Time": "execution_time.png",
        "Final Objective Error": "objective_error.png",
        "Physical ΔV Mismatch": "physical_mismatch.png",
        "ΔV Breakdown": "dv_breakdown.png",
        "Objective Function Contour": "objective_contour.png",
        "GA Convergence Curve": "ga_convergence.png",
        "GA Population Histogram": "ga_population_hist.png"
    }

    report_gen = ReportGenerator(results, input_data, assumptions, plot_filenames)
    report_gen.create_latex_summary()

if __name__ == "__main__":
    main()
