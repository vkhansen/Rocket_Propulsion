"""LaTeX report generation."""
import os
import csv
import json
from datetime import datetime
from ..utils.config import OUTPUT_DIR, logger
import numpy as np

def write_results_to_csv(results, stages, output_dir=OUTPUT_DIR):
    """Write optimization results to CSV files."""
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
                    
                    # Calculate mass ratios using correct formula
                    for i, (dv, stage) in enumerate(zip(result['dv'], stages)):
                        # Correct mass ratio formula: λ = exp(-ΔV/(g₀·ISP)) - ε
                        ratio = np.exp(-dv / (9.81 * stage['ISP'])) - stage['EPSILON']
                        contribution = (dv / total_dv * 100) if total_dv > 0 else 0
                        writer.writerow([
                            i + 1,
                            f"{dv:.1f}",
                            f"{ratio:.4f}",
                            f"{contribution:.1f}",
                            method
                        ])
            logger.info(f"Stage results written to {detailed_path}")
        except Exception as e:
            logger.error(f"Failed to write stage results CSV: {str(e)}")
            detailed_path = None
            
    except Exception as e:
        logger.error(f"Error in CSV generation: {str(e)}")
    
    return summary_path, detailed_path

def compile_latex_to_pdf(tex_path):
    """Compile LaTeX file to PDF using pdflatex."""
    try:
        # Get directory containing the tex file
        tex_dir = os.path.dirname(tex_path)
        tex_file = os.path.basename(tex_path)
        
        # Run pdflatex twice to ensure references are properly resolved
        for _ in range(2):
            cmd = f'pdflatex -interaction=nonstopmode "{tex_file}"'
            result = os.system(f'cd "{tex_dir}" && {cmd}')
            
            if result != 0:
                logger.error(f"Error compiling LaTeX file: {tex_file}")
                return None
        
        pdf_path = tex_path.replace('.tex', '.pdf')
        if os.path.exists(pdf_path):
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
        else:
            logger.error(f"PDF file not found after compilation: {pdf_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error during PDF compilation: {e}")
        return None

def generate_report(results, stages, output_dir=OUTPUT_DIR):
    """Generate a LaTeX report with optimization results and compile to PDF."""
    try:
        if not results:
            logger.error("No results to include in report")
            return None
            
        # Filter out failed optimizations
        valid_results = {k: v for k, v in results.items() if v is not None and 'dv' in v and 'stage_ratios' in v}
        if not valid_results:
            logger.error("No valid optimization results for report")
            return None
            
        # Write results to CSV
        write_results_to_csv(valid_results, stages, output_dir)
            
        report_path = os.path.join(output_dir, "optimization_report.tex")
        
        # Extract global parameters
        g0 = float(stages[0].get('G0', 9.81))  # Standard gravity
        total_dv = float(stages[0].get('TOTAL_DELTA_V', 0))  # Total required ΔV
        
        # Get current date
        current_date = datetime.now().strftime("%B %d, %Y")
        
        report_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{siunitx}
\title{Rocket Stage Optimization Results}
\author{Generated by Stage\_Opt}
\date{""" + current_date + r"""}

\begin{document}
\maketitle

\section{Introduction}
This report presents the results of optimizing a multi-stage rocket using various optimization methods. The objective was to mazimize the payload mass fraction while satisfying the total delta-V requirement.

\section{Input Assumptions}
\subsection{Global Parameters}
\begin{table}[H]
\centering
\caption{Global Parameters}
\begin{tabular}{lS[table-format=4.2]}
\toprule
Parameter & {Value} \\
\midrule
Gravitational Acceleration ($G_0$) & \SI{""" + f"{g0:.2f}" + r"""}{\meter\per\second\squared} \\
Total $\Delta V$ Required & \SI{""" + f"{total_dv:.1f}" + r"""}{\meter\per\second} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Stage Parameters}
\begin{table}[H]
\centering
\caption{Stage Parameters and Assumptions}
\begin{tabular}{cS[table-format=3.0]S[table-format=1.3]}
\toprule
Stage & {ISP (\si{\second})} & {Mass Fraction ($\epsilon$)} \\
\midrule
"""
        # Add stage data rows
        for i, stage in enumerate(stages):
            stage_isp = float(stage.get('ISP', 0))
            stage_epsilon = float(stage.get('EPSILON', 0))
            report_content += f"{i+1} & {stage_isp:.0f} & {stage_epsilon:.3f} \\\\\n"
            
        report_content += r"""\bottomrule
\end{tabular}
\end{table}

\section{Optimization Methods}
The following optimization methods were evaluated:
\begin{itemize}
"""
        # Add optimization methods
        for method in valid_results.keys():
            report_content += f"\\item {method}\n"

        report_content += r"""\end{itemize}

\section{Optimization Results}
\subsection{Performance Visualization}
\begin{figure}[H]
\centering
\includegraphics[width=1.2\textwidth]{dv_breakdown.png}
\caption{$\Delta V$ Distribution Across Stages}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{execution_time.png}
\caption{Solver Execution Time Comparison}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{payload_fraction.png}
\caption{Payload Fraction Comparison}
\end{figure}

\section{Final Results Summary}
\begin{table}[H]
\centering
\caption{Optimization Results Summary}
\begin{tabular}{lS[table-format=1.4]S[table-format=1.4e-1]S[table-format=1.2]}
\toprule
Method & {Payload Fraction} & {Error} & {Time (\si{\second})} \\
\midrule
"""
        # Add solver results with consistent spacing
        for method, result in sorted(valid_results.items(), key=lambda x: x[1]['payload_fraction'], reverse=True):
            try:
                payload = result['payload_fraction']
                time = result['execution_time']
                error = result.get('error', 0)
                # Pad method name for alignment
                method_padded = f"{method:<12}"
                report_content += f"{method_padded} & {payload:.4f} & {error:.4e} & {time:.2f} \\\\\n"
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping incomplete result for {method}: {e}")
                continue
                
        report_content += r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Stage-by-Stage Analysis}

% Individual Stage Comparisons
"""
        # Get number of stages from first result
        num_stages = len(next(iter(valid_results.values()))['dv'])
        
        # Generate comparison table for each stage
        for stage in range(num_stages):
            report_content += f"""
\\begin{{table}}[H]
\\centering
\\caption{{Stage {stage + 1} Comparison Across Methods}}
\\begin{{tabular}}{{lS[table-format=4.1]S[table-format=1.4]S[table-format=3.1]}}
\\toprule
Method & {{$\\Delta V$ (\\si{{\\meter\\per\\second}})}} & {{Mass Ratio ($\\lambda$)}} & {{Contribution (\\%)}} \\\\
\\midrule
"""
            # Add data for each method
            for method, result in sorted(valid_results.items(), key=lambda x: x[1]['payload_fraction'], reverse=True):
                dv = result['dv'][stage]
                ratio = result['stage_ratios'][stage]
                contribution = (dv / sum(result['dv']) * 100)
                report_content += f"{method:<12} & {dv:.1f} & {ratio:.4f} & {contribution:.1f} \\\\\n"
            
            report_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

        # Overall Stage Distribution Analysis
        report_content += r"""
% Overall Stage Distribution Analysis
\begin{table}[H]
\centering
\caption{Stage Distribution Summary}
\begin{tabular}{l"""

        # Add columns for each stage plus total lambda
        for _ in range(num_stages):
            report_content += "S[table-format=4.1]"
        report_content += "S[table-format=1.4]}\n\\toprule\nMethod"

        # Add headers for each stage
        for i in range(num_stages):
            report_content += f" & {{Stage {i+1} (\\%)}}"
        report_content += " & {Total $\\lambda$} \\\\\n\\midrule\n"

        # Add data for each method
        for method, result in sorted(valid_results.items(), key=lambda x: x[1]['payload_fraction'], reverse=True):
            total_dv = sum(result['dv'])
            report_content += f"{method:<12}"
            
            # Add percentage for each stage
            for dv in result['dv']:
                stage_percent = (dv / total_dv * 100)
                report_content += f" & {stage_percent:.1f}"
            
            # Add total lambda
            report_content += f" & {result['payload_fraction']:.4f} \\\\\n"

        report_content += r"""\bottomrule
\end{tabular}
\end{table}

\paragraph{Key Observations:}
\begin{itemize}
"""
        # Add observations about stage distribution patterns
        even_split_value = 100.0 / num_stages
        even_split_threshold = 5.0  # 5% threshold for considering distribution even
        
        even_split_methods = [m for m, r in valid_results.items() 
                            if all(abs(dv/sum(r['dv'])*100 - even_split_value) < even_split_threshold 
                                 for dv in r['dv'])]
        uneven_methods = [m for m, r in valid_results.items() 
                         if any(abs(dv/sum(r['dv'])*100 - even_split_value) >= even_split_threshold 
                              for dv in r['dv'])]
        
        if even_split_methods:
            # Create the split text pattern (e.g., "33.3/33.3/33.3")
            split_pattern = "/".join([f"{even_split_value:.1f}"] * num_stages)
            report_content += r"\item Methods with even $\Delta$V distribution ($\approx" + split_pattern + "$): " + ", ".join(even_split_methods) + "\n"
        if uneven_methods:
            report_content += r"\item " + f"Methods with uneven distribution: {', '.join(uneven_methods)}\n"
            
        # Find method with best mass ratio for each stage
        for i in range(num_stages):
            best_stage = max(valid_results.items(), key=lambda x: x[1]['stage_ratios'][i])[0]
            report_content += r"\item " + f"Best Stage {i+1} mass ratio: {best_stage}\n"
        
        report_content += r"""\end{itemize}

"""
        report_content += r"\end{document}"

        # Write the report with UTF-8 encoding
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                # Replace Unicode Delta with LaTeX Delta
                report_content = report_content.replace('Δ', '$\\Delta$')
                f.write(report_content)
            
            logger.info(f"LaTeX report generated: {report_path}")
            
            # Compile the LaTeX file to PDF
            pdf_path = compile_latex_to_pdf(report_path)
            return pdf_path if pdf_path else report_path
            
        except Exception as e:
            logger.error(f"Error generating LaTeX report: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error generating LaTeX report: {e}")
        return None
