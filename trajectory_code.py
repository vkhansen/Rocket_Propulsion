#!/usr/bin/env python
"""
Example: Combining Delta‑V Split Optimization with a Detailed Ascent Trajectory Simulation
using Poliastro’s atmosphere model and variable engine performance.

This example does the following:
  1. Reads global parameters and stage data (ISP, epsilon) from a CSV file.
  2. Optimizes the ΔV allocation among stages (your existing delta‑V split code).
  3. For demonstration, uses the first stage’s ΔV allocation as the target for
     a detailed vertical ascent simulation.
  4. In the simulation, we optimize the burn time (using SciPy) to achieve the
     allocated ΔV while accounting for variable thrust, variable ISP, drag, and gravity.
  5. The Poliastro US Standard Atmosphere is used to compute air density.
  
Usage:
    python detailed_ascent_simulation.py input_data.csv
"""

import csv
import sys
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Poliastro and Astropy imports for the atmospheric model ---
from poliastro.atmosphere import USStandardAtmosphere1976
import astropy.units as u

# -------------------------------
# Delta‑V Split Optimization Code (your original code)
# -------------------------------
def read_csv_input(filename):
    """Reads CSV file; returns global parameters and stage list."""
    parameters = {}
    stages = []
    mode = "parameters"
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                if mode == "parameters":
                    mode = "stages"
                continue
            if mode == "parameters":
                if row[0].strip().lower() == "parameter":
                    continue  # Skip header
                if len(row) < 2:
                    continue
                parameters[row[0].strip()] = row[1].strip()
            else:
                if row[0].strip().lower() == "stage":
                    continue  # Skip header
                if len(row) < 3:
                    continue
                try:
                    stages.append({
                        'stage': int(row[0].strip()),
                        'ISP': float(row[1].strip()),
                        'EPSILON': float(row[2].strip())
                    })
                except Exception as e:
                    print(f"Error parsing stage row: {row}, {e}")
    return parameters, sorted(stages, key=lambda x: x['stage'])

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """
    Compute the product of stage ratios (each defined as
    exp(-ΔV/(G0·ISP)) – EPSILON) so that maximizing payload is equivalent
    to minimizing the negative product.
    """
    dv = np.asarray(dv).flatten()
    product = 1.0
    for i, dvi in enumerate(dv):
        f_i = np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i]
        if f_i <= 0:
            return 1e6
        product *= f_i
    return -product

def objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, penalty_coeff=1e6, tol=1e-6):
    dv = np.asarray(dv).flatten()
    constraint_error = abs(np.sum(dv) - total_delta_v)
    penalty = penalty_coeff * constraint_error if constraint_error > tol else 0.0
    return payload_fraction_objective(dv, G0, ISP, EPSILON) + penalty

def optimize_payload_allocation(TOTAL_DELTA_V, ISP, EPSILON, G0=9.81, method='SLSQP'):
    n = len(ISP)
    max_dv = np.array([-G0 * isp * np.log(eps) for isp, eps in zip(ISP, EPSILON)])
    if TOTAL_DELTA_V > np.sum(max_dv):
        raise ValueError("TOTAL_DELTA_V exceeds maximum achievable with given stage constraints.")
    if n == 1:
        dv = TOTAL_DELTA_V
        f = np.exp(-dv / (G0 * ISP[0])) - EPSILON[0]
        return [dv], [f], f
    initial_guess = np.full(n, TOTAL_DELTA_V / n)
    bounds = [(0, max_dv_i) for max_dv_i in max_dv]
    if method == 'SLSQP':
        constraints = {'type': 'eq', 'fun': lambda dv: np.sum(dv) - TOTAL_DELTA_V}
        res = minimize(payload_fraction_objective, initial_guess, args=(G0, ISP, EPSILON),
                       method=method, bounds=bounds, constraints=constraints)
    elif method == 'differential_evolution':
        res = differential_evolution(objective_with_penalty, bounds, args=(G0, ISP, EPSILON, TOTAL_DELTA_V),
                                     strategy='best1bin', popsize=200, tol=1e-6,
                                     mutation=(0.5, 1), recombination=0.7, polish=True)
    else:
        raise ValueError("Unsupported optimization method.")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    optimal_dv = res.x
    optimal_stage_ratio = [np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i] for i, dvi in enumerate(optimal_dv)]
    overall_payload = np.prod(optimal_stage_ratio)
    return optimal_dv, optimal_stage_ratio, overall_payload

# -------------------------------
# Detailed Ascent Trajectory Simulation
# -------------------------------

# Define a simple variable thrust model:
def T_func(t, alt, vel, T0):
    """
    Example: Thrust decreases linearly with altitude.
    Ensure a minimum factor so thrust never drops below 70% of T0.
    """
    factor = max(0.7, 1 - alt/100000)  # alt in meters
    return T0 * factor

# Define a simple variable ISP model:
def Isp_func(t, alt, vel, Isp0):
    """
    Example: ISP drops slightly with altitude.
    """
    factor = max(0.9, 1 - 0.1*(alt/100000))
    return Isp0 * factor

# Rocket equations for vertical ascent:
def rocket_equations(t, y, T0, Isp0, Cd, A, atmo, g0=9.81):
    """
    y = [altitude (m), velocity (m/s), mass (kg)]
    """
    alt, vel, mass = y
    # Compute current thrust and ISP (variable with altitude)
    T = T_func(t, alt, vel, T0)
    Isp = Isp_func(t, alt, vel, Isp0)
    # Get atmospheric density (kg/m^3) from Poliastro model.
    # (Ensure altitude is non-negative)
    alt_clipped = max(alt, 0)
    rho = atmo.density(alt_clipped * u.m).to(u.kg / u.m**3).value
    # Simple drag: 0.5 * Cd * A * rho * v^2
    drag = 0.5 * Cd * A * rho * vel**2
    dalt_dt = vel
    dvel_dt = (T / mass) - g0 - (drag / mass)
    dmass_dt = -T / (Isp * g0)  # fuel burn rate
    return [dalt_dt, dvel_dt, dmass_dt]

def simulate_stage(burn_time, initial_state, stage_params, atmo):
    """
    Simulate a stage burn over [0, burn_time] seconds.
    stage_params should be a dict with keys:
      T0    : Nominal engine thrust (N)
      Isp0  : Nominal engine specific impulse (s)
      Cd    : Drag coefficient
      A     : Reference area (m^2)
    initial_state: [altitude (m), velocity (m/s), mass (kg)]
    Returns the time array and state history.
    """
    sol = solve_ivp(
        fun=lambda t, y: rocket_equations(t, y, stage_params['T0'], stage_params['Isp0'],
                                          stage_params['Cd'], stage_params['A'], atmo),
        t_span=(0, burn_time),
        y0=initial_state,
        max_step=0.1,  # small step size for accuracy
        dense_output=True
    )
    t_vals = np.linspace(0, burn_time, 500)
    y_vals = sol.sol(t_vals)
    return t_vals, y_vals

def burn_time_objective(burn_time, initial_state, stage_params, target_delta_v, atmo):
    """
    Objective for optimizing burn time.
    We run the simulation and measure the error between achieved ΔV (change in velocity)
    and the target delta-V allocated by the delta-V split optimization.
    """
    # Ensure burn_time is positive
    burn_time = burn_time[0] if isinstance(burn_time, (list, np.ndarray)) else burn_time
    if burn_time <= 0:
        return 1e6
    try:
        sol = solve_ivp(
            fun=lambda t, y: rocket_equations(t, y, stage_params['T0'], stage_params['Isp0'],
                                              stage_params['Cd'], stage_params['A'], atmo),
            t_span=(0, burn_time),
            y0=initial_state,
            max_step=0.1,
            dense_output=True
        )
        final_state = sol.y[:, -1]
        achieved_delta_v = final_state[1] - initial_state[1]
        error = (achieved_delta_v - target_delta_v)**2
    except Exception:
        error = 1e6
    return error

# -------------------------------
# Main function: Combine both optimization and simulation
# -------------------------------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input_data.csv")
        sys.exit(1)
    
    # --- Part 1: Read CSV and perform delta-V split optimization ---
    input_csv = sys.argv[1]
    parameters, stages = read_csv_input(input_csv)
    
    try:
        G0 = float(parameters.get("G0", 9.81))
        TOTAL_DELTA_V = float(parameters["TOTAL_DELTA_V"])
    except KeyError as e:
        print(f"Missing required parameter in CSV file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid parameter value: {e}")
        sys.exit(1)
    
    ISP_list = [stage['ISP'] for stage in stages]
    EPSILON_list = [stage['EPSILON'] for stage in stages]
    
    # Use SLSQP for simplicity (you can also try differential_evolution, etc.)
    optimal_dv, optimal_stage_ratio, overall_payload = optimize_payload_allocation(
        TOTAL_DELTA_V, ISP_list, EPSILON_list, G0, method='SLSQP'
    )
    
    print("Delta-V split optimization results:")
    for i, dvi in enumerate(optimal_dv):
        print(f"  Stage {i+1}: ΔV = {dvi:.1f} m/s, Stage ratio = {optimal_stage_ratio[i]:.3f}")
    print(f"Overall Payload Fraction: {overall_payload:.4f}")
    
    # --- Part 2: Detailed Ascent Simulation for Stage 1 ---
    # For demonstration we simulate the first stage burn.
    # (In practice, you’d simulate each stage in sequence.)
    
    # Target delta-V for stage 1 from the optimization
    target_delta_v = optimal_dv[0]
    
    # Define initial state for stage 1: starting at ground level, zero velocity.
    initial_mass = 500000.0  # kg (example total mass for stage 1)
    initial_state = [0.0, 0.0, initial_mass]  # [altitude (m), velocity (m/s), mass (kg)]
    
    # Define stage 1 engine and aerodynamic parameters.
    # These values are illustrative.
    stage1_params = {
        'T0': 7.0e6,     # Nominal thrust in Newtons
        'Isp0': ISP_list[0],  # Use the ISP from CSV (in seconds)
        'Cd': 0.5,       # Drag coefficient (assumed)
        'A': 10.0        # Reference area (m^2)
    }
    
    # Create the atmosphere model instance (Poliastro US Standard Atmosphere)
    atmo = USStandardAtmosphere1976()
    
    # Optimize the burn time so that the simulation’s ΔV (at burnout) is as close as possible
    # to the target_delta_v allocated by the delta-V optimization.
    burn_time_guess = 100.0  # seconds initial guess
    res_burn = minimize(burn_time_objective, x0=[burn_time_guess],
                        args=(initial_state, stage1_params, target_delta_v, atmo),
                        bounds=[(10, 300)])  # Assume burn time between 10 and 300 seconds
    
    if not res_burn.success:
        print("Burn time optimization failed.")
        sys.exit(1)
    
    optimal_burn_time = res_burn.x[0]
    print(f"\nOptimized burn time for Stage 1: {optimal_burn_time:.2f} seconds")
    
    # Simulate the stage burn using the optimized burn time.
    t_vals, y_vals = simulate_stage(optimal_burn_time, initial_state, stage1_params, atmo)
    
    # Extract altitude, velocity, and mass histories.
    alt_history = y_vals[0, :]
    vel_history = y_vals[1, :]
    mass_history = y_vals[2, :]
    
    achieved_delta_v = vel_history[-1] - initial_state[1]
    fuel_consumed = initial_mass - mass_history[-1]
    
    print(f"Achieved ΔV: {achieved_delta_v:.1f} m/s (target was {target_delta_v:.1f} m/s)")
    print(f"Fuel consumed in Stage 1: {fuel_consumed:.1f} kg")
    print(f"Final altitude: {alt_history[-1]:.1f} m")
    
    # --- Plot the trajectory for Stage 1 ---
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.plot(t_vals, alt_history, label="Altitude (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(t_vals, vel_history, color="orange", label="Velocity (m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(t_vals, mass_history, color="green", label="Mass (kg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (kg)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("stage1_trajectory.png")
    plt.show()

if __name__ == '__main__':
    main()
