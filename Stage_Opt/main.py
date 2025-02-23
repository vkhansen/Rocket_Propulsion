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
    """Main function."""
    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} input_data.json")
            sys.exit(1)

        input_file = sys.argv[1]
        parameters, stages = load_input_data(input_file)
        
        # Run optimization with different methods
        methods = ['SLSQP', 'DIFFERENTIAL_EVOLUTION', 'BASIN-HOPPING', 'GA', 'GA-ADAPTIVE', 'PSO']
        results = {}
        
        for method in methods:
            try:
                result = optimize_stages(parameters, stages, method)
                if result is not None:
                    results[method] = result
                    if method.upper() != 'GA':
                        logger.info(f"{method} results:")
                        logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result['optimal_dv']]} m/s")
                        logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in result['stage_ratios']]}")
                        logger.info(f"  Payload fraction: {result['payload_fraction']:.3f}")
                        logger.info(f"  Time: {result['execution_time']:.3f} seconds")
                
            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
                continue
        
        # Generate plots
        if results:  # Only plot if we have results
            plot_results(results)
            generate_report(results, stages, OUTPUT_DIR)
            logger.info("Optimization completed successfully")
        else:
            logger.error("No optimization methods succeeded")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise

if __name__ == "__main__":
    main()
