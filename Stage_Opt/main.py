#!/usr/bin/env python3
"""Main script for rocket stage optimization."""
import sys
import time
import numpy as np

from src.utils.config import CONFIG, logger, OUTPUT_DIR
from src.utils.data import load_input_data, calculate_mass_ratios, calculate_payload_fraction
from src.optimization.solvers import (
    solve_with_slsqp,
    solve_with_basin_hopping,
    solve_with_differential_evolution,
    solve_with_ga,
    solve_with_adaptive_ga,
    solve_with_pso
)
from src.visualization.plots import plot_results
from src.reporting.latex import generate_report

def plot_results(results):
    """Generate all plots."""
    try:
        # Standardize result format
        formatted_results = []
        for result in results:
            if result is not None:
                formatted_result = {
                    'method': result.get('method', 'Unknown'),
                    'time': result.get('time', result.get('execution_time', 0.0)),
                    'dv': result.get('dv', result.get('optimal_dv', [])),
                    'stage_ratios': result.get('stage_ratios', []),
                    'payload_fraction': result.get('payload_fraction', 0.0),
                    'error': result.get('error', 0.0)
                }
                formatted_results.append(formatted_result)
        
        if formatted_results:
            plot_dv_breakdown(formatted_results)
            plot_execution_time(formatted_results)
            plot_payload_fraction(formatted_results)
        else:
            logger.warning("No valid results to plot")
            
    except Exception as e:
        logger.error(f"Error in plotting results: {e}")

def optimize_stages(parameters, stages, method='SLSQP'):
    """Run optimization with specified method."""
    try:
        logger.info(f"Starting optimization with method: {method}")
        start_time = time.time()
        
        # Extract stage data
        ISP = [float(stage['ISP']) for stage in stages]
        EPSILON = [float(stage['EPSILON']) for stage in stages]
        G0 = float(parameters.get('G0', 9.81))
        TOTAL_DELTA_V = float(parameters['TOTAL_DELTA_V'])
        
        # Initial setup
        n = len(ISP)
        initial_guess = np.full(n, TOTAL_DELTA_V / n)
        max_dv = TOTAL_DELTA_V * 0.9
        bounds = [(0, max_dv) for _ in range(n)]
        
        # Select solver
        solver_result = None
        if method.upper() == 'SLSQP':
            optimal_dv = solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
            if optimal_dv is not None:
                stage_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON, G0)
                payload_fraction = calculate_payload_fraction(stage_ratios)
                solver_result = {
                    'method': 'SLSQP',
                    'optimal_dv': optimal_dv.tolist() if isinstance(optimal_dv, np.ndarray) else optimal_dv,
                    'stage_ratios': stage_ratios.tolist() if isinstance(stage_ratios, np.ndarray) else stage_ratios,
                    'payload_fraction': float(payload_fraction),
                    'execution_time': time.time() - start_time
                }
        elif method.upper() == 'BASIN-HOPPING':
            optimal_dv = solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
            if optimal_dv is not None:
                stage_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON, G0)
                payload_fraction = calculate_payload_fraction(stage_ratios)
                solver_result = {
                    'method': 'Basin-Hopping',
                    'optimal_dv': optimal_dv.tolist() if isinstance(optimal_dv, np.ndarray) else optimal_dv,
                    'stage_ratios': stage_ratios.tolist() if isinstance(stage_ratios, np.ndarray) else stage_ratios,
                    'payload_fraction': float(payload_fraction),
                    'execution_time': time.time() - start_time
                }
        elif method.upper() == 'DIFFERENTIAL_EVOLUTION':
            optimal_dv = solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
            if optimal_dv is not None:
                stage_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON, G0)
                payload_fraction = calculate_payload_fraction(stage_ratios)
                solver_result = {
                    'method': 'Differential Evolution',
                    'optimal_dv': optimal_dv.tolist() if isinstance(optimal_dv, np.ndarray) else optimal_dv,
                    'stage_ratios': stage_ratios.tolist() if isinstance(stage_ratios, np.ndarray) else stage_ratios,
                    'payload_fraction': float(payload_fraction),
                    'execution_time': time.time() - start_time
                }
        elif method.upper() == 'GA':
            solver_result = solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
            if solver_result is not None:
                logger.info(f"GA results:")
                logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in solver_result['dv']]} m/s")
                logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in solver_result['stage_ratios']]}")
                logger.info(f"  Payload fraction: {solver_result['payload_fraction']:.3f}")
                logger.info(f"  Time: {solver_result['time']:.3f} seconds")
                return solver_result
            else:
                logger.error(f"Method {method} failed to produce valid results")
                return None
        elif method.upper() == 'GA-ADAPTIVE':
            solver_result = solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
        elif method.upper() == 'PSO':
            solver_result = solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        if solver_result is None:
            raise Exception(f"{method} optimization failed to produce valid results")
            
        execution_time = time.time() - start_time
        logger.info(f"Optimization completed in {execution_time:.3f} seconds")
        
        return solver_result
        
    except Exception as e:
        logger.error(f"Error in {method} optimization: {e}")
        return None

def main():
    """Main function to run optimization and generate results."""
    try:
        # Parse command line arguments
        if len(sys.argv) != 2:
            print("Usage: python main.py <input_file>")
            sys.exit(1)
            
        # Load input data
        input_file = sys.argv[1]
        parameters, stages = load_input_data(input_file)
        
        # Initialize results list
        results = []
        
        # Initial setup for optimization
        n = len(stages)
        g0 = parameters['g0']
        isp = [stage['isp'] for stage in stages]
        epsilon = [stage['epsilon'] for stage in stages]
        dv_total = parameters['total_dv']
        x0 = np.array([dv_total/n] * n)  # Equal split initial guess
        
        # Calculate bounds
        bounds = []
        for _ in range(n):
            bounds.append([0, dv_total])
        bounds = np.array(bounds)
        
        # Run SLSQP optimization
        logger.info("Starting SLSQP optimization")
        slsqp_result = solve_with_slsqp(x0, bounds, g0, isp, epsilon, dv_total, parameters)
        if slsqp_result:
            results.append(slsqp_result)
        else:
            logger.error("Method SLSQP failed to produce valid results")
            
        # Run Basin-Hopping optimization
        logger.info("Starting Basin-Hopping optimization")
        basin_result = solve_with_basin_hopping(x0, bounds, g0, isp, epsilon, dv_total, parameters)
        if basin_result:
            results.append(basin_result)
        else:
            logger.error("Method Basin-Hopping failed to produce valid results")
        
        # Run GA optimization
        logger.info("Starting GA optimization")
        ga_result = solve_with_ga(n, g0, isp, epsilon, dv_total, x0, bounds)
        if ga_result:
            results.append(ga_result)
        else:
            logger.error("Method GA failed to produce valid results")
        
        # Run Adaptive GA optimization
        logger.info("Starting Adaptive GA optimization with parameters:")
        logger.info(f"Initial guess: {x0}")
        logger.info(f"G0: {g0}, ISP: {isp}, EPSILON: {epsilon}")
        logger.info(f"TOTAL_DELTA_V: {dv_total}")
        
        adaptive_ga_result = solve_with_adaptive_ga(x0, bounds, g0, isp, epsilon, dv_total, parameters)
        if adaptive_ga_result:
            results.append(adaptive_ga_result)
            logger.info("Adaptive GA optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{x:.2f}' for x in adaptive_ga_result['dv']]} m/s")
            logger.info(f"  Mass ratios: {[f'{x:.3f}' for x in adaptive_ga_result['stage_ratios']]}")
            logger.info(f"  Payload fraction: {adaptive_ga_result['payload_fraction']:.3f}")
            logger.info(f"  Time: {adaptive_ga_result['time']:.3f} seconds")
        
        # Run PSO optimization
        logger.info("Starting PSO optimization")
        pso_result = solve_with_pso(x0, bounds, g0, isp, epsilon, dv_total, parameters)
        if pso_result:
            results.append(pso_result)
        else:
            logger.error("Method PSO failed to produce valid results")
        
        # Generate plots if we have results
        if results:
            try:
                plot_results(results)
            except Exception as e:
                logger.error(f"Error in plotting results: {e}")
            
            try:
                generate_report(results, stages, OUTPUT_DIR)
            except Exception as e:
                logger.error(f"Error generating LaTeX report: {e}")
        
        logger.info("Optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
