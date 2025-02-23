# Rocket Stage Optimization

## Overview
This project focuses on optimizing the staging of a multi-stage rocket to maximize payload fraction while satisfying mission constraints. The optimization process leverages **Tsiolkovsky's Rocket Equation** and multiple numerical solvers to distribute the velocity change (**Delta-V**) among stages efficiently.

## Theoretical Background
The fundamental equation governing rocket staging is:

```
ΔV = I_{sp} \cdot g_0 \cdot \ln \left( \frac{m_0}{m_f} \right)
```

Where:
- `ΔV` is the total velocity change required for the mission.
- `I_{sp}` is the **specific impulse** of each stage, a measure of fuel efficiency.
- `g_0` is the standard gravitational acceleration (9.81 m/s²).
- `m_0` is the initial mass (including fuel and structure).
- `m_f` is the final mass after stage separation.

The mass fraction can also be expressed in terms of the **structural mass ratio** (`ε`):

```
\frac{m_f}{m_0} = 1 - ε
```

where `ε` represents the fraction of the initial mass that is structural rather than propellant. The optimization seeks to minimize `ε` while ensuring sufficient Delta-V for the mission.



## Implementation Details

### 1. **Input Handling**
- **Configuration File (`config.json`)**:
  - Defines solver parameters including penalty coefficients, iteration limits, and solver-specific configurations (e.g., genetic algorithm settings, PSO parameters).
  - Example:
    ```json
    {
        "optimization": {
            "tolerance": 1e-6,
            "max_iterations": 200,
            "ga": { "population_size": 100, "n_generations": 200 },
            "pso": { "n_particles": 50, "n_iterations": 200 }
        }
    }
    ```

- **Mission Data (`input_data.json`)**:
  - Specifies the mission’s global parameters and stage details.
  - Example:
    ```json
    {
        "parameters": {
            "G0": 9.81,
            "TOTAL_DELTA_V": 9300.0
        },
        "stages": [
            {"stage": 1, "ISP": 280, "EPSILON": 0.15},
            {"stage": 2, "ISP": 348, "EPSILON": 0.04}
        ]
    }
    ```

- **Function (`load_input_data`)**:
  - Loads input data and sorts stages.
  - **Inputs:** JSON filename
  - **Outputs:** `parameters` dictionary, `stages` list

### 2. **Optimization Algorithms** (`solvers.py`)
This project implements multiple numerical solvers:
- **SLSQP (Sequential Least Squares Quadratic Programming)**: Gradient-based method.
- **Basin-Hopping**: Stochastic global search algorithm.
- **Genetic Algorithm (GA & Adaptive GA)**: Evolutionary approach mimicking natural selection.
- **Differential Evolution (DE)**: Population-based optimization.
- **Particle Swarm Optimization (PSO)**: Simulated swarming behavior to find optima.

Each solver iterates over different Delta-V distributions to find the best combination that maximizes payload fraction while respecting constraints.

### 3. **Execution Flow (`main.py`)**
- Reads input files.
- Defines initial conditions and constraints.
- Iterates through multiple solvers to optimize staging.
- Logs performance and saves results.

### 4. **Function Breakdown**

#### **Objective Functions (`objective.py`)**
- `payload_fraction_objective(dv, G0, ISP, EPSILON)`:
  - Computes the payload fraction for a given Delta-V distribution.
  - **Inputs:** Delta-V distribution, gravity, ISP, structural mass ratio.
  - **Outputs:** Negative payload fraction (for minimization).

- `objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)`:
  - Adds constraint violations to the objective function.
  - **Inputs:** Delta-V distribution, mission parameters.
  - **Outputs:** Objective function value with penalties.

#### **Data Processing (`data.py`)**
- `calculate_mass_ratios(dv, ISP, EPSILON, G0)`:
  - Computes stage mass ratios using the rocket equation.
  - **Inputs:** Delta-V values, ISP, structural mass ratios, gravity.
  - **Outputs:** Mass ratios per stage.

- `calculate_payload_fraction(mass_ratios)`:
  - Computes overall payload fraction as a product of stage ratios.
  - **Inputs:** Mass ratios array.
  - **Outputs:** Payload fraction value.

#### **Report Generation (`latex.py`)**
- `generate_report(results, stages, output_dir)`:
  - Creates a LaTeX report with optimization results.
  - **Inputs:** Optimization results dictionary, stage details.
  - **Outputs:** LaTeX file summarizing optimization.

#### **Visualization (`plots.py`)**
- `plot_dv_breakdown(results, filename)`:
  - Generates a bar chart showing Delta-V distribution.
  - **Inputs:** Optimization results dictionary.
  - **Outputs:** PNG file with stacked bars per stage.

- `plot_execution_time(results, filename)`:
  - Plots solver execution time.
  - **Inputs:** Optimization results dictionary.
  - **Outputs:** PNG bar plot.

- `plot_payload_fraction(results, filename)`:
  - Compares payload fractions among solvers.
  - **Inputs:** Optimization results dictionary.
  - **Outputs:** PNG bar plot.

### 5. **Output & Visualization**
- **Plots (`plots.py`)**: Generates breakdowns of Delta-V allocation, execution time, and payload fraction.
- **LaTeX Report (`latex.py`)**: Summarizes results in a structured document.
- **Output Directory (`output/`)**:
  - `dv_breakdown.png`: Shows Delta-V allocation per stage.
  - `execution_time.png`: Solver performance comparison.
  - `payload_fraction.png`: Impact of staging on payload.
  - `optimization_report.tex`: Generated LaTeX report.

## Running the Project

### Dependencies
Ensure you have Python 3 and install required packages:
```bash
pip install -r requirements.txt
```

### Running the Optimization
```bash
python main.py input_data.json
```
This will execute the optimization using all solvers and store results in `output/`.

## Conclusion
This project provides a comprehensive framework for optimizing multi-stage rockets using various numerical techniques. It enables engineers to explore different staging strategies and analyze their impact on payload capacity efficiently.

# Rocket Stage Optimization (Stage_Opt)

This module performs multi-stage rocket optimization to maximize payload fraction while satisfying delta-V requirements. It implements various optimization algorithms and provides detailed analysis through CSV outputs and LaTeX reports.

## Theory

The optimization is based on the rocket equation and mass ratio calculations:

### Mass Ratio (λ)
For each stage i:

```
λᵢ = exp(-ΔVᵢ/(g₀·ISPᵢ)) - εᵢ
```

Where:
- `ΔVᵢ` = Delta-V contribution of stage i (m/s)
- `g₀` = Standard gravity (9.81 m/s²)
- `ISPᵢ` = Specific impulse of stage i (seconds)
- `εᵢ` = Mass fraction of stage i (dimensionless)

### Payload Fraction
The total payload fraction is the product of all stage mass ratios:

```
Payload Fraction = ∏ᵢ λᵢ
```

### Total Delta-V Constraint
The sum of individual stage delta-Vs must meet the mission requirement:

```
∑ᵢ ΔVᵢ = Total Required ΔV
```

## Directory Structure

```
Stage_Opt/
├── input/                  # Input configuration files
├── output/                # Generated results and reports
├── src/                   # Source code
│   ├── optimization/     # Optimization algorithms
│   ├── reporting/       # CSV and LaTeX report generation
│   ├── utils/          # Utility functions and configuration
│   └── visualization/  # Plotting and visualization tools
└── tests/              # Test suite
```

## Key Components

### 1. Optimization Algorithms (`src/optimization/solvers.py`)
- **SLSQP**: Sequential Least Squares Programming
- **Basin-Hopping**: Global optimization with local minimization
- **Genetic Algorithm (GA)**: Standard and adaptive variants
- **Differential Evolution (DE)**: Population-based optimization
- **Particle Swarm Optimization (PSO)**: Swarm intelligence approach

### 2. Input/Output

#### Input (`input_data.json`)
```json
{
    "parameters": {
        "G0": 9.81,
        "TOTAL_DELTA_V": 9300.0
    },
    "stages": [
        {
            "stage": 1,
            "ISP": 300,
            "EPSILON": 0.06
        },
        {
            "stage": 2,
            "ISP": 348,
            "EPSILON": 0.04
        }
    ]
}
```

#### Output Files
- `stage_results.csv`: Detailed results for each stage
  - Stage number
  - Delta-V contribution
  - Mass ratio (λ)
  - Percentage contribution
  - Optimization method

- `optimization_summary.csv`: Overall performance metrics
  - Method name
  - Final payload fraction
  - Optimization error
  - Execution time

### 3. Analysis Tools

- **CSV Generation** (`src/reporting/latex.py`):
  - Generates structured CSV files for result analysis
  - Calculates mass ratios and payload fractions
  - Tracks optimization performance metrics

- **LaTeX Reports**:
  - Professional-grade PDF reports
  - Performance comparisons across methods
  - Detailed stage-by-stage analysis

## Usage

1. **Setup Input**:
   ```bash
   # Modify input_data.json with your parameters
   {
       "parameters": {"TOTAL_DELTA_V": your_delta_v},
       "stages": [
           {"stage": 1, "ISP": isp1, "EPSILON": epsilon1},
           {"stage": 2, "ISP": isp2, "EPSILON": epsilon2}
       ]
   }
   ```

2. **Run Optimization**:
   ```bash
   python main.py [input_file.json]
   ```

3. **View Results**:
   - Check `output/stage_results.csv` for detailed stage analysis
   - Check `output/optimization_summary.csv` for method comparison
   - View `output/optimization_report.tex` for full LaTeX report

## Testing

Comprehensive test suite in `test_payload_optimization.py`:
```bash
python -m pytest test_payload_optimization.py -v
```

Tests cover:
- Input data loading and validation
- Mass ratio calculations
- Payload fraction optimization
- CSV output structure and consistency
- Optimization algorithm performance

## Dependencies

- NumPy: Numerical computations
- SciPy: Optimization algorithms
- Pymoo: Genetic algorithm implementation
- Pandas: Data manipulation
- LaTeX: Report generation

## Performance Notes

1. **Algorithm Selection**:
   - SLSQP: Fast, good for well-behaved problems
   - GA/DE: Better for avoiding local minima
   - PSO: Good balance of exploration/exploitation

2. **Convergence**:
   - All methods should achieve similar payload fractions
   - Execution times vary significantly
   - Error values indicate constraint satisfaction

3. **Validation**:
   - Mass ratios follow theoretical calculations
   - Total delta-V constraint is enforced
   - Stage contributions are properly balanced

## Contributing

When contributing:
1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Verify optimization results against theory
