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
            
        # Filter out failed optimizations
        valid_results = {k: v for k, v in results.items() if v is not None and 'optimal_dv' in v and 'stage_ratios' in v}
        if not valid_results:
            logger.error("No valid optimization results to plot")
            return
            
        plt.figure(figsize=(14, 7))
        
        # Get number of methods
        n_methods = len(valid_results)
        method_positions = np.arange(n_methods)
        bar_width = 0.5
        
        # Create stacked bars for each method
        bottom = np.zeros(n_methods)
        colors = ['dodgerblue', 'orange', 'green', 'red', 'purple']  # More colors for stages
        
        # Plot each stage
        n_stages = len(next(iter(valid_results.values()))['optimal_dv'])
        for stage in range(n_stages):
            try:
                # Extract ΔV values and ratios for this stage across all methods
                stage_dvs = []
                stage_ratios = []
                for result in valid_results.values():
                    try:
                        dv = result['optimal_dv'][stage]
                        ratio = result['stage_ratios'][stage]
                        stage_dvs.append(float(dv))
                        stage_ratios.append(float(ratio))
                    except (IndexError, KeyError, TypeError) as e:
                        logger.error(f"Error extracting stage {stage} data: {e}")
                        stage_dvs.append(0)
                        stage_ratios.append(0)
                
                plt.bar(method_positions, stage_dvs, bar_width,
                       bottom=bottom, color=colors[stage % len(colors)],
                       label=f'Stage {stage+1}')
                
                # Add text labels
                for i, (dv, ratio) in enumerate(zip(stage_dvs, stage_ratios)):
                    if dv > 0:  # Only add label if there's a meaningful value
                        plt.text(i, bottom[i] + dv/2,
                                f"{dv:.0f} m/s\n(λ={ratio:.2f})",
                                ha='center', va='center',
                                color='white', fontweight='bold',
                                fontsize=9)
                
                bottom += np.array(stage_dvs)
            except Exception as e:
                logger.error(f"Error plotting stage {stage}: {e}")
                continue
        
        # Add total text above each bar
        for i, total in enumerate(bottom):
            if total > 0:  # Only add total if there's a meaningful value
                plt.text(i, total + 100,
                        f"Total: {total:.0f} m/s",
                        ha='center', va='bottom',
                        fontweight='bold')
        
        plt.ylabel('ΔV (m/s)')
        plt.xlabel('Optimization Method')
        plt.title('ΔV Solution per Solver')
        plt.xticks(method_positions, list(valid_results.keys()), rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
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
