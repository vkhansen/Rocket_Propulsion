"""CSV report generation for optimization results."""
import os
import csv
from ..utils.config import logger

def write_results_to_csv(results, stages, output_dir):
    """Write optimization results to CSV files.
    
    Args:
        results (dict): Dictionary containing optimization results for each method
        stages (list): List of stage configurations
        output_dir (str): Directory to write CSV files to
    
    Returns:
        tuple: Paths to the generated CSV files (summary_path, detailed_path)
    """
    summary_path = None
    detailed_path = None
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write summary results
        try:
            summary_path = os.path.join(output_dir, "optimization_summary.csv")
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Method', 'Payload Fraction', 'Error', 'Time (s)'])
                for method, result in results.items():
                    writer.writerow([
                        method,
                        f"{result.get('payload_fraction', 0.0):.4f}",
                        f"{result.get('constraint_violation', 0.0):.4e}",  # Changed from 'error'
                        f"{result.get('execution_time', 0.0):.2f}"
                    ])
            logger.info(f"Summary results written to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to write summary CSV: {str(e)}")
            summary_path = None
        
        # Write detailed stage results
        try:
            detailed_path = os.path.join(output_dir, "stage_results.csv")
            with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Stage', 'Delta-V (m/s)', 'Stage Ratio (lambda)', 'Delta-V Contribution (%)', 'Method'])
                for method, result in results.items():
                    if not result.get('dv') or not result.get('stage_ratios'):  # Check for required data
                        logger.warning(f"Missing required data for {method}")
                        continue
                        
                    dv_values = result['dv']
                    lambda_values = result['stage_ratios']
                    total_dv = sum(dv_values)
                    
                    for i, (dv, lambda_ratio) in enumerate(zip(dv_values, lambda_values)):
                        writer.writerow([
                            i + 1,
                            f"{dv:.2f}",
                            f"{lambda_ratio:.4f}",
                            f"{(dv/total_dv)*100:.1f}",
                            method
                        ])
            logger.info(f"Stage results written to {detailed_path}")
        except Exception as e:
            logger.error(f"Failed to write detailed CSV: {str(e)}")
            detailed_path = None
            
        return summary_path, detailed_path
        
    except Exception as e:
        logger.error(f"Error in write_results_to_csv: {e}")
        return None, None