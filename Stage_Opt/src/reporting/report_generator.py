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
            
        # Track successful solvers
        successful_solvers = []
            
        for method, result in results.items():
            # Skip invalid results
            if not isinstance(result, dict):
                logger.warning(f"Skipping invalid result for method {method}")
                continue
                
            try:
                # Get execution metrics from the correct location
                execution_metrics = result.get('execution_metrics', {})
                if not isinstance(execution_metrics, dict):
                    execution_metrics = {
                        'iterations': 0,
                        'function_evaluations': 0,
                        'execution_time': 0.0
                    }
                
                method_report = {
                    'success': bool(result.get('success', False)),
                    'message': str(result.get('message', '')),
                    'payload_fraction': float(result.get('payload_fraction', 0.0)),
                    'constraint_violation': float(result.get('constraint_violation', float('inf'))),
                    'execution_metrics': {
                        'iterations': int(execution_metrics.get('iterations', 0)),
                        'function_evaluations': int(execution_metrics.get('function_evaluations', 0)),
                        'execution_time': float(execution_metrics.get('execution_time', 0.0))
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
                                'Lambda': float(stage.get('Lambda', 0.0))
                            }
                            method_report['stages'].append(stage_data)
                
                report['results'][method] = method_report
                
                # Track successful solvers - only check success flag, not constraint violation
                if method_report['success']:
                    successful_solvers.append(method)
                
            except Exception as e:
                logger.error(f"Error processing result for method {method}: {str(e)}")
                continue
            
        # Save report only if we have valid results
        if report['results']:
            output_path = os.path.join(OUTPUT_DIR, filename)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Report saved to {output_path}")
            logger.info(f"Successful solvers: {successful_solvers}")
            return report
        else:
            logger.warning("No valid results to include in report")
            return None
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None
