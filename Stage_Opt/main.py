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
        
        with open(input_file) as f:
            data = json.load(f)
            parameters = data["parameters"]
            stages = data["stages"]
        
        TOTAL_DELTA_V = float(parameters['TOTAL_DELTA_V'])
        G0 = float(parameters.get('G0', 9.81))
        ISP = [float(stage['ISP']) for stage in stages]
        EPSILON = [float(stage['EPSILON']) for stage in stages]
        
        n_stages = len(stages)
        initial_guess = [TOTAL_DELTA_V / n_stages] * n_stages
        bounds = [(0, TOTAL_DELTA_V) for _ in range(n_stages)]
        
        optimizer = RocketStageOptimizer(CONFIG, parameters, stages)
        results = optimizer.solve(initial_guess, bounds)
        
        if results:
            # Generate plots
            plot_results(results)
            
            try:
                # Generate report
                report = generate_report(results, CONFIG)
                logger.info("Reports generated successfully")
            except Exception as e:
                logger.error(f"Error generating reports: {e}")
        else:
            logger.error("No optimization results to process")
            
    except Exception as e:
        logger.error(f"Error in main routine: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
