"""Plotting functions for optimization results."""
import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils.config import OUTPUT_DIR, logger

def plot_dv_breakdown(results, filename="dv_breakdown.png"):
    """Plot ΔV breakdown for each optimization method."""
    try:
        if not results:
            logger.error("No results to plot for ΔV breakdown")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Get number of methods
        n_methods = len(results)
        method_positions = np.arange(n_methods)
        bar_width = 0.35
        
        # Create stacked bars for each method
        bottom = np.zeros(n_methods)
        colors = ['dodgerblue', 'orange', 'green']  # Colors for up to 3 stages
        
        # Plot each stage
        n_stages = len(next(iter(results.values()))['optimal_dv'])
        for stage in range(n_stages):
            # Extract ΔV values and ratios for this stage across all methods
            stage_dvs = [result['optimal_dv'][stage] for result in results.values()]
            stage_ratios = [result['stage_ratios'][stage] for result in results.values()]
            
            plt.bar(method_positions, stage_dvs, bar_width,
                   bottom=bottom, color=colors[stage % len(colors)],
                   label=f'Stage {stage+1}')
            
            # Add text labels
            for i, (dv, ratio) in enumerate(zip(stage_dvs, stage_ratios)):
                plt.text(i, bottom[i] + dv/2,
                        f"{dv:.0f} m/s\n(λ={ratio:.2f})",
                        ha='center', va='center',
                        color='white', fontweight='bold',
                        fontsize=9)
            
            bottom += np.array(stage_dvs)
        
        # Add total text above each bar
        for i, total in enumerate(bottom):
            plt.text(i, total + 100,
                    f"Total: {total:.0f} m/s",
                    ha='center', va='bottom',
                    fontweight='bold')
        
        plt.ylabel('ΔV (m/s)')
        plt.xlabel('Optimization Method')
        plt.title('ΔV Solution per Solver')
        plt.xticks(method_positions, list(results.keys()))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in plotting ΔV breakdown: {e}")

def plot_execution_time(results, filename="execution_time.png"):
    """Plot execution time for each optimization method."""
    try:
        if not results:
            logger.error("No results to plot for execution time")
            return
            
        plt.figure(figsize=(10, 6))
        
        methods = list(results.keys())
        times = [result['time'] for result in results.values()]
        
        plt.bar(methods, times, color='dodgerblue')
        plt.ylabel('Execution Time (s)')
        plt.xlabel('Optimization Method')
        plt.title('Solver Execution Time Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, time in enumerate(times):
            plt.text(i, time, f"{time:.3f}s",
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in plotting execution time: {e}")

def plot_payload_fraction(results, filename="payload_fraction.png"):
    """Plot payload fraction for each optimization method."""
    try:
        if not results:
            logger.error("No results to plot for payload fraction")
            return
            
        plt.figure(figsize=(10, 6))
        
        methods = list(results.keys())
        payload_fractions = [result['payload_fraction'] for result in results.values()]
        
        plt.bar(methods, payload_fractions, color='dodgerblue')
        plt.ylabel('Payload Fraction')
        plt.xlabel('Optimization Method')
        plt.title('Payload Fraction Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, pf in enumerate(payload_fractions):
            plt.text(i, pf, f"{pf:.3f}",
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in plotting payload fraction: {e}")

def plot_results(results):
    """Generate all plots."""
    try:
        if not results:
            logger.error("No results to plot")
            return
            
        plot_dv_breakdown(results)
        plot_execution_time(results)
        plot_payload_fraction(results)
        
    except Exception as e:
        logger.error(f"Error in plotting results: {e}")
