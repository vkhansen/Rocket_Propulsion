# Rocket Stage Optimization (Stage_Opt)

This module performs multi-stage rocket optimization to maximize payload fraction while satisfying delta-V requirements. It implements various optimization algorithms and provides detailed analysis through CSV outputs and LaTeX reports.

## Table of Contents
1. [Theoretical Background](#theoretical-background)
2. [Implementation](#implementation)
3. [Usage Guide](#usage-guide)
4. [Testing](#testing)
5. [Contributing](#contributing)

## Theoretical Background
*Author: V H, February 2025*

### Core Definitions

1. **Payload Fraction (λ)**:
λ = m_payload/m_0
Where `m_payload` is the payload mass and `m_0` is the total initial mass.

2. **Total Mass Ratio (R)**:
R = m_0/m_f
Where `m_f` is the final mass after propellant burn.

3. **Rocket Equation (Single Stage)**:
ΔV = v_e · ln(R) = v_e · ln(m_0/m_f)
Where `v_e = I_sp · g_0` is the exhaust velocity, and `g_0 = 9.81 m/s²`.

### Multi-Stage Rockets

4. **Stage Mass Ratio**:
R_i = m_0,i/m_f,i
Where `m_0,i` and `m_f,i` are the initial and final masses of stage i.

5. **Delta-V per Stage**:
ΔV_i = v_e,i · ln(R_i)

6. **Total Delta-V**:
ΔV_total = Σ(i=1 to n) ΔV_i = Σ(i=1 to n) v_e,i · ln(R_i)
Where `n` is the number of stages.

7. **Multi-Stage Payload Fraction**:
λ = m_payload/m_0,total = ∏(i=1 to n) m_f,i/m_0,i = ∏(i=1 to n) 1/R_i

8. **Structural Mass Ratio**:
ε_i = m_structure,i/m_0,i

### Optimization Problem

The solver maximizes payload fraction (λ) subject to:
- Total required Delta-V constraint
- Structural mass fraction limits
- Stage mass ratio constraints
- Physical feasibility conditions

## Implementation

### Directory Structure
```
Stage_Opt/
├── input/                 # Configuration files
├── output/               # Results and reports
├── src/                  # Source code
│   ├── optimization/    # Optimization algorithms
│   ├── reporting/      # CSV and LaTeX generation
│   ├── utils/         # Utility functions
│   └── visualization/ # Plotting tools
└── tests/             # Test suite
```

### Optimization Algorithms
1. **SLSQP**: Sequential Least Squares Programming
   - Fast convergence for well-behaved problems
   - Local optimization with constraints

2. **Genetic Algorithm (GA)**:
   - Population-based global optimization
   - Adaptive variant available for better convergence

3. **Particle Swarm (PSO)**:
   - Swarm intelligence approach
   - Good balance of exploration/exploitation

4. **Basin Hopping**:
   - Global optimization with local minimization
   - Temperature-controlled exploration

5. **Differential Evolution**:
   - Population-based optimization
   - Good for avoiding local minima

### Configuration

The `config.json` file controls optimization parameters:

```json
{
    "optimization": {
        "penalty_coefficient": 1000.0,
        "tolerance": 1e-6,
        "max_iterations": 200,
        "bounds": {
            "min_dv": 0.0,
            "max_dv_factor": 1.0
        },
        "ga": {
            "population_size": 100,
            "n_generations": 200,
            "crossover_prob": 0.9,
            "mutation_prob": 0.2
        },
        "pso": {
            "n_particles": 50,
            "n_iterations": 200,
            "c1": 0.5,
            "c2": 0.3,
            "w": 0.9
        }
    }
}
```

### Input Data
Mission parameters are defined in `input_data.json`:
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

## Usage Guide

1. **Setup**:
   ```bash
   # Configure mission parameters
   edit input_data.json
   # Adjust optimization settings
   edit config.json
   ```

2. **Run**:
   ```bash
   python main.py
   ```

3. **Results**:
   - Check `output/stage_results.csv` for stage analysis
   - View `output/optimization_summary.csv` for method comparison
   - See `output/optimization_report.tex` for detailed report

## Testing
```bash
python -m pytest test_payload_optimization.py -v
```

Tests cover:
- Mass ratio calculations
- Payload fraction optimization
- CSV output validation
- Algorithm performance

## Contributing
1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Verify optimization results
