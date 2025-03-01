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
        if not isinstance(results, dict):
            logger.warning("Results must be a dictionary")
            return None
            
        for method, result in results.items():
            # Skip invalid results
            if not isinstance(result, dict):
                logger.warning(f"Skipping invalid result for method {method}")
                continue
                
            try:
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
                stages = result.get('stages', [])
                if isinstance(stages, list):
                    for stage in stages:
                        if isinstance(stage, dict):
                            stage_data = {
                                'stage': int(stage.get('stage', 0)),
                                'delta_v': float(stage.get('delta_v', 0.0)),
                                'Lambda': float(stage.get('Lambda', 0.0)),
                                'ISP': float(stage.get('ISP', 0.0)),
                                'EPSILON': float(stage.get('EPSILON', 0.0))
                            }
                            method_report['stages'].append(stage_data)
                
                report['results'][method] = method_report
                
            except Exception as e:
                logger.error(f"Error processing result for method {method}: {str(e)}")
                continue
            
        # Save report only if we have valid results
        if report['results']:
            output_path = os.path.join(OUTPUT_DIR, filename)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Report saved to {output_path}")
            return report
        else:
            logger.warning("No valid results to include in report")
            return None
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None
