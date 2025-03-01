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
            method_report = {
                'success': bool(result.get('success', False)),
                'message': str(result.get('message', '')),
                'payload_fraction': float(result.get('payload_fraction', 0.0)),
                'constraint_violation': float(result.get('constraint_violation', float('inf'))),
                'execution_metrics': {
                    'iterations': int(result.get('n_iterations', 0)),
                    'function_evaluations': int(result.get('n_function_evals', 0)),
                    'execution_time': float(result.get('execution_time', 0.0))
                },
                'stages': []
            }
            
            # Process stage results
            for stage in result.get('stages', []):
                stage_data = {
                    'stage': int(stage.get('stage', 0)),
                    'delta_v': float(stage.get('delta_v', 0.0)),
                    'Lambda': float(stage.get('Lambda', 0.0)),
                    'ISP': float(stage.get('ISP', 0.0)),
                    'EPSILON': float(stage.get('EPSILON', 0.0))
                }
                method_report['stages'].append(stage_data)
            
            report['results'][method] = method_report
            
        # Save report
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Report saved to {output_path}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None
