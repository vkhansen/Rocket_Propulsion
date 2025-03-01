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
            results_list = list(results.values())
        else:
            results_list = results if isinstance(results, list) else []
            
        # Skip if no valid results
        if not results_list or not isinstance(results_list[0], dict):
            logger.warning("No valid results to plot DV breakdown")
            return
            
        # Get number of methods
        n_methods = len(results_list)
        method_positions = np.arange(n_methods)
        bar_width = 0.35
        
        # Create stacked bars for each method
        bottom = np.zeros(n_methods)
        colors = ['dodgerblue', 'orange', 'green']  # Colors for up to 3 stages
        
        # Plot each stage
        n_stages = len(results_list[0].get('stages', []))
        for stage_idx in range(n_stages):
            # Extract DV values and ratios for this stage across all methods
            stage_dvs = []
            stage_ratios = []
            
            for result in results_list:
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
        plt.xticks(method_positions, [result.get('method', f'Method {i}') if isinstance(result, dict) else f'Method {i}'
                                    for i, result in enumerate(results_list)])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
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
            results_list = list(results.values())
        else:
            results_list = results if isinstance(results, list) else []
            
        # Skip if no valid results
        if not results_list or not isinstance(results_list[0], dict):
            logger.warning("No valid results to plot execution time")
            return
            
        # Extract execution times and method names
        times = [float(result.get('execution_time', 0.0)) if isinstance(result, dict) else 0.0 
                for result in results_list]
        methods = [result.get('method', f'Method {i}') if isinstance(result, dict) else f'Method {i}'
                  for i, result in enumerate(results_list)]
        
        # Create bar plot
        plt.bar(methods, times, color='dodgerblue')
        
        # Add value labels on top of bars
        for i, time in enumerate(times):
            plt.text(i, time, f'{time:.2f}s',
                    ha='center', va='bottom')
        
        # Customize plot
        plt.xlabel('Optimization Method')
        plt.ylabel('Execution Time (s)')
        plt.title('Execution Time by Method')
        plt.grid(True, alpha=0.3)
        
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
            results_list = list(results.values())
        else:
            results_list = results if isinstance(results, list) else []
            
        # Skip if no valid results
        if not results_list or not isinstance(results_list[0], dict):
            logger.warning("No valid results to plot payload fraction")
            return
            
        # Extract payload fractions and method names
        payload_fractions = [float(result.get('payload_fraction', 0.0)) if isinstance(result, dict) else 0.0
                           for result in results_list]
        methods = [result.get('method', f'Method {i}') if isinstance(result, dict) else f'Method {i}'
                  for i, result in enumerate(results_list)]
        
        # Create bar plot
        plt.bar(methods, payload_fractions, color='dodgerblue')
        
        # Add value labels on top of bars
        for i, pf in enumerate(payload_fractions):
            plt.text(i, pf, f'{pf:.3f}',
                    ha='center', va='bottom')
        
        # Customize plot
        plt.xlabel('Optimization Method')
        plt.ylabel('Payload Fraction')
        plt.title('Payload Fraction by Method')
        plt.grid(True, alpha=0.3)
        
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
