"""CSV report generation for optimization results."""
import os
import csv
from ..utils.config import OUTPUT_DIR, logger

def write_results_to_csv(results, stages, output_dir=OUTPUT_DIR):
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
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Method', 'Payload Fraction', 'Error', 'Time (s)'])
                for method, result in results.items():
                    if not all(k in result for k in ['payload_fraction', 'error', 'execution_time']):
                        logger.warning(f"Skipping incomplete result for {method}")
                        continue
                    writer.writerow([
                        method,
                        f"{result['payload_fraction']:.4f}",
                        f"{result['error']:.4e}",
                        f"{result['execution_time']:.2f}"
                    ])
            logger.info(f"Summary results written to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to write summary CSV: {str(e)}")
            summary_path = None
        
        # Write detailed stage results
        try:
            detailed_path = os.path.join(output_dir, "stage_results.csv")
            with open(detailed_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Stage', 'Delta-V (m/s)', 'Mass Ratio', 'Contribution (%)', 'Method'])
                for method, result in results.items():
                    if not all(k in result for k in ['dv', 'stage_ratios']):
                        logger.warning(f"Skipping incomplete stage data for {method}")
                        continue
                    total_dv = sum(result['dv'])
                    for i, (dv, ratio) in enumerate(zip(result['dv'], result['stage_ratios'])):
                        writer.writerow([
                            f"Stage {i+1}",
                            f"{dv:.2f}",
                            f"{ratio:.4f}",
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