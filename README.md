# Payload Optimization for Multi-Stage Rockets

This repository contains a collection of Python scripts designed to optimize the allocation of a rocket's total available ΔV (delta-V) among its multiple stages. The goal is to maximize the overall payload fraction, taking into account the rocket’s physical constraints and stage-specific parameters. The code leverages several optimization solvers to address this non-linear, constrained optimization problem.

---

## Overview

The payload optimization problem is based on a modified version of the Tsiolkovsky rocket equation. For each stage, the **stage ratio** is defined as:

  **f₍ᵢ₎ = exp(–ΔV₍ᵢ₎ / (G₀ · Isp₍ᵢ₎)) – ε₍ᵢ₎**

where:  
- **ΔV₍ᵢ₎** is the delta-V allocated to stage *i*,  
- **G₀** is the standard gravitational acceleration,  
- **Isp₍ᵢ₎** is the specific impulse of stage *i*, and  
- **ε₍ᵢ₎** represents structural ratio for stage *i*.

The overall payload fraction is computed as the product of the stage ratios. Since optimization routines generally minimize functions, the code minimizes the **negative product** of the stage ratios while ensuring that the sum of ΔV allocations equals the total available ΔV. Each stage's ΔV is also constrained by a maximum value determined by the condition that the stage ratio remains positive.

---

## File Structure

- **payload_optimization.py**  
  Contains the main routines for reading input data, computing the objective function, and performing the optimization. It supports multiple solver methods:
  - **SLSQP** (Sequential Least Squares Programming) via SciPy’s `minimize` with equality constraints.
  - **Differential Evolution** via SciPy’s `differential_evolution`, using a penalty-based approach.
  - **Genetic Algorithm (GA)** via the [pymoo](https://pymoo.org/) library, which uses an evolutionary approach.
  
  citeturn0file1

- **test_payload_optimization.py**  
  Contains unit tests that verify the correct behavior of the CSV input reader, objective function, and each optimization method. This ensures that any modifications do not break the underlying logic.  
  citeturn0file0

- **input_data.csv**  
  A sample CSV file that provides the global parameters and stage data in the following format:
  - **Section 1 (Global Parameters):**  
    ```
    Parameter,Value
    G0,9.81
    TOTAL_DELTA_V,9500
    ```
  - **Section 2 (Stage Data):**  
    ```
    stage,ISP,EPSILON
    1,300,0.1
    2,320,0.08
    ```

- **Optimization Output Files:**  
  The main script generates several output files upon execution:
  - **optimization_results.csv** – Contains the detailed results of the optimization.
  - **stage_data.csv** and **input_variables.csv** – Contain the parsed input data.
  - **report.tex** – A LaTeX report that integrates plots and tabulated results.
  - **dv_breakdown.png**, **execution_time.png**, and **payload_fraction.png** – Plots visualizing the ΔV allocation breakdown, solver execution times, and achieved payload fractions.

- **test_debug.log**  
  Logs from running the unit tests are stored here, helping in debugging and verifying test execution.  
  citeturn0file2

---

## Theory

The core idea of this project stems from the classical Tsiolkovsky rocket equation, which relates the change in velocity (ΔV) to the mass ratio of the rocket. In a multi-stage rocket, the payload fraction is determined by the product of the stage ratios. However, real-world considerations such as structural losses (modeled via the ε parameter) require a modified formulation:

1. **Stage Ratio Calculation:**  
   Each stage’s performance is modeled as:  
   > f₍ᵢ₎ = exp(–ΔV₍ᵢ₎ / (G₀ · Isp₍ᵢ₎)) – ε₍ᵢ₎  
   This captures the idealized mass loss (via the exponential term) and subtracts a penalty for inefficiencies.

2. **Optimization Objective:**  
   The overall payload fraction is the product of the stage ratios:  
   > Payload Fraction = ∏ f₍ᵢ₎  
   Since optimization routines are designed for minimization, the code minimizes the negative product, turning the maximization into a minimization problem.

3. **Constraints:**  
   - **Equality Constraint:** The sum of ΔV allocated to each stage must equal the total available ΔV.  
   - **Inequality Constraints:** Each ΔV allocation is bounded between zero and a maximum value, computed as:  
     > max_ΔV = –G₀ · Isp · ln(ε)  
     This ensures that the stage ratio remains positive.

---

## Optimization Solvers

The project implements three different optimization strategies to solve the payload allocation problem:

1. **SLSQP (Sequential Least Squares Programming):**  
   - A gradient-based optimization method available in SciPy's `minimize` function.
   - Directly handles equality constraints (i.e., the sum of ΔV must equal TOTAL_DELTA_V).
   
2. **Differential Evolution:**  
   - A stochastic, population-based optimization algorithm available in SciPy.
   - Uses a penalty function to enforce the equality constraint by penalizing deviations from the total ΔV.

3. **Genetic Algorithm (GA):**  
   - An evolutionary algorithm implemented via the pymoo library.
   - A custom problem class is defined to evaluate the objective with a penalty, making it suitable for the non-linear, constrained nature of the problem.

Each solver has its strengths, and the repository provides a framework for comparing their performance in terms of execution time and the resulting payload fraction.

---

## Installation

Ensure you have Python 3 installed. Then, install the required packages using pip:

```bash
pip install numpy scipy matplotlib pandas pymoo
```

---

## Usage

### Running the Optimization

To run the optimization using the CSV input file:

```bash
python payload_optimization.py input_data.csv
```

This will:
- Parse the input CSV.
- Optimize the ΔV allocation for each solver (SLSQP, Differential Evolution, and GA).
- Generate plots, CSV reports, and a LaTeX report summarizing the results.

### Running the Unit Tests

To run the unit tests:

```bash
python -m unittest test_payload_optimization.py
```

This ensures that all functions behave as expected and that the optimization routines work correctly.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

This project leverages robust scientific libraries such as NumPy, SciPy, and pymoo to address a real-world optimization problem in aerospace engineering. Contributions and feedback are welcome!

---

Feel free to open issues or submit pull requests for improvements and additional features.

Happy optimizing!
