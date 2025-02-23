"""Plotting functions for optimization results."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from ..utils.config import OUTPUT_DIR, logger

def plot_dv_breakdown(results, filename="dv_breakdown.png"):
    """Plot ΔV breakdown for each optimization method."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Convert results dict to list if necessary
        if isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results
            
        # Get number of methods
        n_methods = len(results_list)
        method_positions = np.arange(n_methods)
        bar_width = 0.35
        
        # Create stacked bars for each method
        bottom = np.zeros(n_methods)
        colors = ['dodgerblue', 'orange', 'green']  # Colors for up to 3 stages
        
        # Plot each stage
        n_stages = len(results_list[0]['dv'])
        for stage in range(n_stages):
            # Extract ΔV values and ratios for this stage across all methods
            stage_dvs = np.array([result['dv'][stage] for result in results_list])
            stage_ratios = np.array([result['stage_ratios'][stage] for result in results_list])
            
            # Plot bars for this stage
            plt.bar(method_positions, stage_dvs, bar_width,
                   bottom=bottom, color=colors[stage % len(colors)],
                   label=f'Stage {stage+1}')
            
            # Add text labels with ΔV and λ values
            for i, (dv, ratio) in enumerate(zip(stage_dvs, stage_ratios)):
                plt.text(i, float(bottom[i]) + float(dv)/2,
                        f"{float(dv):.0f} m/s\nλ={float(ratio):.3f}",
                        ha='center', va='center',
                        color='white', fontweight='bold',
                        fontsize=9)
            
            # Update bottom for next stack
            bottom = bottom + stage_dvs
        
        # Add total text above each bar
        for i, total in enumerate(bottom):
            plt.text(i, float(total) + 100,
                    f"Total: {float(total):.0f} m/s",
                    ha='center', va='bottom',
                    fontweight='bold')
        
        # Add horizontal line for total mission ΔV
        total_dv = float(np.sum(results_list[0]['dv']))  # Use first result's total
        plt.axhline(y=total_dv, color='red', linestyle='--',
                   label=f'Required ΔV = {total_dv} m/s')
        
        plt.ylabel('Delta-V (m/s)')
        plt.xlabel('Optimization Method')
        plt.title('Delta-V Solution per Solver')
        plt.xticks(method_positions, [result['method'] for result in results_list])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in plotting Delta-V breakdown: {e}")

def plot_execution_time(results, filename="execution_time.png"):
    """Plot execution time for each optimization method."""
    try:
        plt.figure(figsize=(10, 5))
        
        # Convert results dict to list if necessary
        if isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results
        
        # Extract method names and times
        method_names = [result['method'] for result in results_list]
        times = [result.get('execution_time', result.get('time', 0)) for result in results_list]
        
        plt.bar(method_names, times)
        plt.title('Solver Execution Time')
        plt.xlabel('Solver Method')
        plt.ylabel('Time (s)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Execution time plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in plotting execution time: {e}")

def plot_payload_fraction(results, filename="payload_fraction.png"):
    """Plot payload fraction for each optimization method."""
    try:
        plt.figure(figsize=(10, 5))
        
        # Convert results dict to list if necessary
        if isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results
        
        # Extract method names and payload fractions
        method_names = [result['method'] for result in results_list]
        payload_fractions = [result['payload_fraction'] for result in results_list]
        
        bars = plt.bar(method_names, payload_fractions)
        plt.title('Payload Mass Fraction per Solver')
        plt.xlabel('Solver Method')
        plt.ylabel('Payload Mass Fraction')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Payload fraction plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in plotting payload fraction: {e}")

def plot_results(results):
    """Generate all plots."""
    try:
        plot_dv_breakdown(results)
        plot_execution_time(results)
        plot_payload_fraction(results)
    except Exception as e:
        logger.error(f"Error in plotting results: {e}")
