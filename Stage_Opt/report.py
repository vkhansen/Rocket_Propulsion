# report.py
class ReportGenerator:
    def __init__(self, results, input_data, assumptions, plot_filenames, output_file='results_summary.tex'):
        self.results = results
        self.input_data = input_data
        self.assumptions = assumptions
        self.plot_filenames = plot_filenames
        self.output_file = output_file

    def create_latex_summary(self):
        latex_str = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\usepackage{longtable}
\begin{document}

\section*{Results Summary}

\subsection*{Key Input Data and Assumptions}
\begin{table}[ht]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
"""
        for key, value in self.input_data.items():
            value_str = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
            latex_str += f"{key} & {value_str} \\\\\n"

        latex_str += r"""\bottomrule
\end{tabular}
\caption{Key Input Parameters}
\end{table}

\subsubsection*{Assumptions}
""" + self.assumptions + "\n"

        latex_str += r"""
\subsection*{Solver Performance Summary}
\begin{table}[ht]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Solver} & \textbf{Time (s)} & \textbf{Final Error} & \textbf{$\Delta V$ Mismatch (m/s)} & \textbf{Solution Vector} \\
\midrule
"""
        for res in self.results:
            sol_str = ', '.join([f"{x:.3f}" for x in res["solution"]])
            latex_str += f"{res['name']} & {res['time']:.4f} & {res['error']:.2e} & {res['mismatch']:.2f} & {sol_str} \\\\\n"

        latex_str += r"""\bottomrule
\end{tabular}
\caption{Summary of Solver Performance}
\end{table}
"""

        latex_str += r"""
\subsection*{Figures}
"""
        for fig_caption, filename in self.plot_filenames.items():
            latex_str += r"""
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{""" + filename + r"""}
\caption{""" + fig_caption + r"""}
\end{figure}
"""
        latex_str += r"\end{document}"

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"LaTeX summary written to {self.output_file}")
