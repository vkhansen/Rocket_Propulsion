"""Report generation functions for optimization results."""
import os
import json
from datetime import datetime
from ..utils.config import logger, OUTPUT_DIR

def generate_report(results, config, filename="optimization_report.json"):
    """Generate a JSON report of optimization results."""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results': {}
        }
        
        # Process results for each method
        for method, result in results.items():
            report['results'][method] = {
                'success': result.get('success', False),
                'payload_fraction': result.get('payload_fraction', 0.0),
                'stage_ratios': result.get('stage_ratios', []),
                'mass_ratios': result.get('mass_ratios', []),
                'iterations': result.get('n_iterations', 0),
                'function_evaluations': result.get('n_function_evals', 0),
                'execution_time': result.get('execution_time', 0.0)
            }
            
            if not result.get('success', False):
                report['results'][method]['error'] = result.get('message', 'Unknown error')
        
        # Save report
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Report saved to {output_path}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None
