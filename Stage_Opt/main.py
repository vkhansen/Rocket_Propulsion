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
from src.reporting.csv_reports import write_results_to_csv


def main():
    """Main optimization routine."""
    try:
        input_file = sys.argv[1] if len(sys.argv) > 1 else "input_data.json"
        parameters, stages = load_input_data(input_file)
        
        TOTAL_DELTA_V = float(parameters['TOTAL_DELTA_V'])
        G0 = float(parameters.get('G0', 9.81))
        ISP = [float(stage['ISP']) for stage in stages]
        EPSILON = [float(stage['EPSILON']) for stage in stages]
        
        n_stages = len(stages)
        initial_guess = np.array([TOTAL_DELTA_V / n_stages] * n_stages)
        bounds = [(0, TOTAL_DELTA_V) for _ in range(n_stages)]
        
        results = {}
        
        methods = [
            ("SLSQP", solve_with_slsqp),
            ("BASIN-HOPPING", solve_with_basin_hopping),
            ("GA", solve_with_ga),
            ("ADAPTIVE-GA", solve_with_adaptive_ga),
            ("DE", solve_with_differential_evolution),
            ("PSO", solve_with_pso)
        ]
        
        for method_name, solver in methods:
            try:
                logger.info(f"Starting {method_name} optimization with parameters:")
                logger.info(f"Initial guess: {initial_guess}")
                logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
                logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
                
                start_time = time.time()
                solution = solver(
                    initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG
                )
                
                if solution is not None:
                    execution_time = time.time() - start_time
                    
                    # Extract values from the new solution format
                    payload_fraction = solution['payload_fraction']
                    stages = solution['stages']
                    stage_dvs = [stage['delta_v'] for stage in stages]
                    stage_lambdas = [stage['Lambda'] for stage in stages]
                    
                    results[method_name] = {
                        'method': method_name,
                        'execution_time': execution_time,
                        'dv': stage_dvs,
                        'stage_ratios': stage_lambdas,
                        'payload_fraction': float(payload_fraction),
                        'error': float(abs(sum(stage_dvs) - TOTAL_DELTA_V))
                    }
                    
                    logger.info(f"{method_name} optimization succeeded:")
                    logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in stage_dvs]} m/s")
                    logger.info(f"  Stage ratios (Λ): {[f'{r:.3f}' for r in stage_lambdas]}")
                    logger.info(f"  Payload fraction: {payload_fraction:.3f}")
                    logger.info(f"Successfully completed {method_name} optimization")
            except Exception as e:
                logger.error(f"Optimization with {method_name} failed: {e}")
        
        if results:
            plot_results(results)
            try:
                # Generate CSV reports first
                summary_path, detailed_path = write_results_to_csv(results, stages, output_dir=OUTPUT_DIR)
                if summary_path and detailed_path:
                    logger.info(f"CSV reports generated: {summary_path}, {detailed_path}")
                else:
                    logger.warning("Failed to generate CSV reports")
                
                # Generate LaTeX report
                generate_report(results, stages, output_dir=OUTPUT_DIR)
            except Exception as e:
                logger.error(f"Error generating reports: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in main routine: {str(e)}")
        raise


if __name__ == '__main__':
    main()
