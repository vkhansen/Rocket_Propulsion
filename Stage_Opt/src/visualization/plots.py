"""Plotting functions for optimization results."""
import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils.config import logger, OUTPUT_DIR

def plot_dv_breakdown(results, filename="dv_breakdown.png"):
    """Plot DV breakdown for each optimization method."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Convert results dict to list if necessary
        if isinstance(results, dict):
            # Create a list of tuples (solver_name, result_dict)
            results_list = [(solver_name, result) for solver_name, result in results.items()]
        else:
            results_list = results if isinstance(results, list) else []
            
        # Skip if no valid results
        if not results_list:
            logger.warning("No valid results to plot DV breakdown")
            return
            
        # Get number of methods
        n_methods = len(results_list)
        method_positions = np.arange(n_methods)
        bar_width = 0.35
        
        # Create stacked bars for each method
        bottom = np.zeros(n_methods)
        colors = ['dodgerblue', 'orange', 'green', 'red', 'purple']  # Colors for up to 5 stages
        
        # Get method names for x-axis labels
        method_names = []
        for item in results_list:
            if isinstance(item, tuple) and len(item) == 2:
                # If it's a (solver_name, result_dict) tuple from a dictionary
                solver_name, _ = item
                method_names.append(solver_name)
            elif isinstance(item, dict):
                # If it's just a result dictionary
                method_names.append(item.get('solver_name', 'Unknown'))
            else:
                method_names.append('Unknown')
        
        # Determine max number of stages across all methods
        max_stages = 0
        for item in results_list:
            result = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if isinstance(result, dict):
                stages = result.get('stages', [])
                max_stages = max(max_stages, len(stages))
        
        if max_stages == 0:
            logger.warning("No stages found in results")
            return
            
        # Plot each stage
        for stage_idx in range(max_stages):
            # Extract DV values and ratios for this stage across all methods
            stage_dvs = []
            stage_ratios = []
            
            for item in results_list:
                # Extract the result dict from tuple if necessary
                result = item[1] if isinstance(item, tuple) and len(item) == 2 else item
                
                if not isinstance(result, dict):
                    stage_dvs.append(0.0)
                    stage_ratios.append(0.0)
                    continue
                    
                stages = result.get('stages', [])
                if stage_idx < len(stages):
                    stage = stages[stage_idx]
                    stage_dvs.append(float(stage.get('delta_v', 0.0)))
                    stage_ratios.append(float(stage.get('Lambda', 0.0)))
                else:
                    stage_dvs.append(0.0)
                    stage_ratios.append(0.0)
            
            stage_dvs = np.array(stage_dvs)
            stage_ratios = np.array(stage_ratios)
            
            # Plot bars for this stage
            plt.bar(method_positions, stage_dvs, bar_width,
                   bottom=bottom, color=colors[stage_idx % len(colors)],
                   label=f'Stage {stage_idx+1}')
            
            # Add text labels with DV and Lambda values
            for i, (dv, lambda_ratio) in enumerate(zip(stage_dvs, stage_ratios)):
                if dv > 0:  # Only add text if there's a bar
                    # Add black text with white background for better visibility
                    plt.text(i, float(bottom[i]) + float(dv)/2,
                            f"{float(dv):.0f} m/s\nL={float(lambda_ratio):.3f}",
                            ha='center', va='center',
                            color='black', fontweight='bold',
                            fontsize=10, bbox=dict(
                                facecolor='white',
                                alpha=0.7,
                                edgecolor='none',
                                pad=1
                            ))
            
            bottom += stage_dvs
        
        # Customize plot
        plt.xlabel('Optimization Method')
        plt.ylabel('Delta-V (m/s)')
        plt.title('Stage Delta-V Breakdown by Method')
        plt.xticks(method_positions, method_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Delta-V breakdown plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting Delta-V breakdown: {str(e)}")

def plot_execution_time(results, filename="execution_time.png"):
    """Plot execution time for each optimization method."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Convert results dict to list if necessary
        if isinstance(results, dict):
            # Create a list of tuples (solver_name, result_dict)
            results_list = [(solver_name, result) for solver_name, result in results.items()]
        else:
            results_list = results if isinstance(results, list) else []
            
        # Skip if no valid results
        if not results_list:
            logger.warning("No valid results to plot execution time")
            return
            
        # Extract execution times and method names
        times = []
        method_names = []
        
        for item in results_list:
            # Extract the result dict from tuple if necessary
            solver_name = None
            result = None
            
            if isinstance(item, tuple) and len(item) == 2:
                # If it's a (solver_name, result_dict) tuple from a dictionary
                solver_name, result = item
            else:
                result = item
                solver_name = result.get('solver_name', 'Unknown') if isinstance(result, dict) else 'Unknown'
                
            method_names.append(solver_name)
            
            # Extract execution time from execution_metrics
            if isinstance(result, dict):
                execution_metrics = result.get('execution_metrics', {})
                times.append(float(execution_metrics.get('execution_time', 0.0)))
            else:
                times.append(0.0)
        
        # Create bar plot
        plt.bar(method_names, times, color='dodgerblue')
        
        # Add value labels on top of bars
        for i, time in enumerate(times):
            plt.text(i, time, f'{time:.2f}s',
                    ha='center', va='bottom')
        
        # Customize plot
        plt.xlabel('Optimization Method')
        plt.ylabel('Execution Time (s)')
        plt.title('Execution Time by Method')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Execution time plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting execution time: {str(e)}")

def plot_payload_fraction(results, filename="payload_fraction.png"):
    """Plot payload fraction for each optimization method."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Convert results dict to list if necessary
        if isinstance(results, dict):
            # Create a list of tuples (solver_name, result_dict)
            results_list = [(solver_name, result) for solver_name, result in results.items()]
        else:
            results_list = results if isinstance(results, list) else []
            
        # Skip if no valid results
        if not results_list:
            logger.warning("No valid results to plot payload fraction")
            return
            
        # Extract payload fractions and method names
        payload_fractions = []
        method_names = []
        
        for item in results_list:
            # Extract the result dict from tuple if necessary
            solver_name = None
            result = None
            
            if isinstance(item, tuple) and len(item) == 2:
                # If it's a (solver_name, result_dict) tuple from a dictionary
                solver_name, result = item
            else:
                result = item
                solver_name = result.get('solver_name', 'Unknown') if isinstance(result, dict) else 'Unknown'
                
            method_names.append(solver_name)
            
            # Extract payload fraction
            if isinstance(result, dict):
                payload_fractions.append(float(result.get('payload_fraction', 0.0)))
            else:
                payload_fractions.append(0.0)
        
        # Create bar plot
        plt.bar(method_names, payload_fractions, color='dodgerblue')
        
        # Add value labels on top of bars
        for i, pf in enumerate(payload_fractions):
            plt.text(i, pf, f'{pf:.6f}',
                    ha='center', va='bottom')
        
        # Customize plot
        plt.xlabel('Optimization Method')
        plt.ylabel('Payload Fraction')
        plt.title('Payload Fraction by Method')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Payload fraction plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting payload fraction: {str(e)}")

def plot_results(results):
    """Generate all plots."""
    try:
        if not results:
            logger.warning("No results to plot")
            return
            
        plot_dv_breakdown(results)
        plot_execution_time(results)
        plot_payload_fraction(results)
        logger.info("All plots generated successfully")
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
