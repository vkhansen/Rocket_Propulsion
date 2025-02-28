#!/usr/bin/env python3
"""Main script for rocket stage optimization."""
import os
import json
import time
from datetime import datetime

from src.utils.config import CONFIG, logger, OUTPUT_DIR
from src.optimization.objective import RocketStageOptimizer
from src.reporting.report_generator import generate_report
from src.visualization.plots import plot_results


def main():
    """Main optimization routine."""
    try:
        input_file = "input_data.json"
        
        parameters, stages = json.load(open(input_file))
        
        TOTAL_DELTA_V = float(parameters['TOTAL_DELTA_V'])
        G0 = float(parameters.get('G0', 9.81))
        ISP = [float(stage['ISP']) for stage in stages]
        EPSILON = [float(stage['EPSILON']) for stage in stages]
        
        n_stages = len(stages)
        initial_guess = [TOTAL_DELTA_V / n_stages] * n_stages
        bounds = [(0, TOTAL_DELTA_V) for _ in range(n_stages)]
        
        results = {}
        
        solver = RocketStageOptimizer(CONFIG, parameters, stages)
        
        try:
            logger.info(f"Starting {solver.name} optimization with parameters:")
            logger.info(f"Initial guess: {initial_guess}")
            logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
            logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
            
            start_time = time.time()
            solution = solver.solve(initial_guess, bounds)
            
            if solution is not None:
                execution_time = time.time() - start_time
                
                # Extract values from the new solution format
                payload_fraction = solution['payload_fraction']
                stages = solution['stages']
                stage_dvs = [stage['delta_v'] for stage in stages]
                stage_lambdas = [stage['Lambda'] for stage in stages]
                
                results[solver.name] = {
                    'method': solver.name,
                    'execution_time': execution_time,
                    'dv': stage_dvs,
                    'stage_ratios': stage_lambdas,
                    'payload_fraction': float(payload_fraction),
                    'error': float(abs(sum(stage_dvs) - TOTAL_DELTA_V))
                }
                
                logger.info(f"{solver.name} optimization succeeded:")
                logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in stage_dvs]} m/s")
                logger.info(f"  Stage ratios (lambda): {[f'{r:.3f}' for r in stage_lambdas]}")
                logger.info(f"  Payload fraction: {payload_fraction:.3f}")
                logger.info(f"Successfully completed {solver.name} optimization")
        except Exception as e:
            logger.error(f"Optimization with {solver.name} failed: {e}")
        
        if results:
            plot_results(results)
            try:
                # Generate LaTeX report
                generate_report(results, stages, output_dir=OUTPUT_DIR)
            except Exception as e:
                logger.error(f"Error generating reports: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in main routine: {str(e)}")
        raise


if __name__ == '__main__':
    main()
