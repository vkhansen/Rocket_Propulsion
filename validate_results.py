import json
import numpy as np
import pandas as pd

# 1. Load the JSON data from file
json_path = "./Stage_Opt/output/optimization_report.json"
with open(json_path, "r") as f:
    data = json.load(f)

# 2. Define reference constants
g0 = 9.81  # m/s^2
total_delta_v = 9300
isp_stage1 = 312.0
isp_stage2 = 348.0
epsilon1 = 0.07
epsilon2 = 0.08

# 3. Helpers to compute rocket‐equation lambda and 2‐stage payload fraction
def calculate_lambda(delta_v, isp):
    return np.exp(-delta_v / (g0 * isp))

def calculate_payload_fraction(lambda1, lambda2, epsilon1, epsilon2):
    term1 = (lambda1 - epsilon1) / (1 - epsilon1)
    term2 = (lambda2 - epsilon2) / (1 - epsilon2)
    return term1 * term2

# 4. Create a structure to hold comparisons
results = {
    "Solver": [],
    "Stage 1 Δv": [],
    "Stage 1 λ (Comp)": [],
    "Stage 1 λ (Doc)": [],
    "Stage 2 Δv": [],
    "Stage 2 λ (Comp)": [],
    "Stage 2 λ (Doc)": [],
    "Total Δv": [],
    "Δv Valid?": [],
    "PF (Comp)": [],
    "PF (Doc)": [],
    "Δ PF": []
}

# 5. Optional mapping for better display
solver_display_names = {
    "SLSQPSolver": "SLSQP",
    "BasinHoppingOptimizer": "Basin Hopping",
    "DifferentialEvolutionSolver": "Differential Evolution",
    "ParticleSwarmOptimizer": "Particle Swarm",
    "GeneticAlgorithmSolver": "Genetic Algorithm",
    "AdaptiveGeneticAlgorithmSolver": "Adaptive Genetic Algorithm"
}

# 6. Loop over each solver's data, do computations & store
for solver_key, solver_data in data["results"].items():
    solver_name = solver_display_names.get(solver_key, solver_key)
    
    # Check if the solver had valid stages
    stages_list = solver_data.get("stages", [])
    if len(stages_list) < 2:
        # Not enough stage data or constraints were violated
        results["Solver"].append(solver_name)
        results["Stage 1 Δv"].append(None)
        results["Stage 1 λ (Comp)"].append(None)
        results["Stage 1 λ (Doc)"].append(None)
        results["Stage 2 Δv"].append(None)
        results["Stage 2 λ (Comp)"].append(None)
        results["Stage 2 λ (Doc)"].append(None)
        results["Total Δv"].append(None)
        results["Δv Valid?"].append(False)
        # We still show the doc's payload fraction (could be zero or negative)
        payload_doc = solver_data.get("payload_fraction", 0.0)
        results["PF (Comp)"].append(None)
        results["PF (Doc)"].append(payload_doc)
        results["Δ PF"].append(None)
        continue
    
    # Extract data from the JSON
    delta_v1 = stages_list[0]["delta_v"]
    delta_v2 = stages_list[1]["delta_v"]
    lambda1_doc = stages_list[0]["Lambda"]
    lambda2_doc = stages_list[1]["Lambda"]
    payload_fraction_doc = solver_data["payload_fraction"]
    
    # Compute rocket-eq lambdas
    lambda1_comp = calculate_lambda(delta_v1, isp_stage1)
    lambda2_comp = calculate_lambda(delta_v2, isp_stage2)
    
    # 2-stage payload fraction
    payload_fraction_comp = calculate_payload_fraction(
        lambda1_comp, lambda2_comp, epsilon1, epsilon2
    )
    
    total_dv_comp = delta_v1 + delta_v2
    dv_is_valid = np.isclose(total_dv_comp, total_delta_v, rtol=1e-5)
    
    # Store
    results["Solver"].append(solver_name)
    results["Stage 1 Δv"].append(delta_v1)
    results["Stage 1 λ (Comp)"].append(lambda1_comp)
    results["Stage 1 λ (Doc)"].append(lambda1_doc)
    results["Stage 2 Δv"].append(delta_v2)
    results["Stage 2 λ (Comp)"].append(lambda2_comp)
    results["Stage 2 λ (Doc)"].append(lambda2_doc)
    results["Total Δv"].append(total_dv_comp)
    results["Δv Valid?"].append(bool(dv_is_valid))
    results["PF (Comp)"].append(payload_fraction_comp)
    results["PF (Doc)"].append(payload_fraction_doc)
    diff_val = abs(payload_fraction_comp - payload_fraction_doc)
    results["Δ PF"].append(diff_val)

# 7. Convert to DataFrame for display
df = pd.DataFrame(results)

# Display the results with Pandas
print("Validation of Solvers\n")
pd.set_option("display.max_columns", None)
print(df)

# 8. Print a summary
print("\nSummary of Validation:\n")

headers = [
    "Solver",
    "Stg1 Δv",
    "Stg1 λ (Comp)",
    "Stg1 λ (Doc)",
    "Stg2 Δv",
    "Stg2 λ (Comp)",
    "Stg2 λ (Doc)",
    "Total Δv",
    "Δv Valid?",
    "PF (Comp)",
    "PF (Doc)",
    "Δ PF",
]

# Create a format string that has the correct alignment/width
row_format = (
    "{:>24}  "   # Solver name
    "{:>10}  "   # Stg1 Δv
    "{:>13}  "   # Stg1 λ (Comp)
    "{:>13}  "   # Stg1 λ (Doc)
    "{:>10}  "   # Stg2 Δv
    "{:>13}  "   # Stg2 λ (Comp)
    "{:>13}  "   # Stg2 λ (Doc)
    "{:>10}  "   # Total Δv
    "{:>10}  "   # Δv Valid?
    "{:>10}  "   # PF (Comp)
    "{:>10}  "   # PF (Doc)
    "{:>10}"     # Δ PF
)

# Print the header row
print(row_format.format(*headers))
print("-" * 140)

# Safely format row entries. If `None`, show "None".
for i, row in df.iterrows():
    def fmt_or_none(val, fmt=".5f"):
        if val is None or pd.isna(val):
            return "None"
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return format(val, fmt)
        return str(val)
    
    print(
        row_format.format(
            str(row["Solver"]),
            fmt_or_none(row["Stage 1 Δv"], ".2f"),
            fmt_or_none(row["Stage 1 λ (Comp)"], ".6f"),
            fmt_or_none(row["Stage 1 λ (Doc)"], ".6f"),
            fmt_or_none(row["Stage 2 Δv"], ".2f"),
            fmt_or_none(row["Stage 2 λ (Comp)"], ".6f"),
            fmt_or_none(row["Stage 2 λ (Doc)"], ".6f"),
            fmt_or_none(row["Total Δv"], ".2f"),
            fmt_or_none(row["Δv Valid?"], ""),
            fmt_or_none(row["PF (Comp)"], ".8f"),
            fmt_or_none(row["PF (Doc)"], ".8f"),
            fmt_or_none(row["Δ PF"], ".8f"),
        )
    )
