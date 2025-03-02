"""Configuration utilities for optimization solvers."""

def get_solver_config(config, solver_name):
    """Get solver-specific configuration with defaults.
    
    Args:
        config: Main configuration dictionary
        solver_name: Name of the solver (e.g., 'ga', 'adaptive_ga', 'pso')
        
    Returns:
        dict: Solver configuration with defaults applied
    """
    # Get optimization section with defaults
    opt_config = config.get('optimization', {})
    
    # Get solver specific config from solvers section
    solver_config = opt_config.get('solvers', {}).get(solver_name, {})
    
    # Get constraints from main optimization config
    constraints = opt_config.get('constraints', {})
    
    # Common defaults
    defaults = {
        'penalty_coefficient': opt_config.get('penalty_coefficient', 1e3),
        'constraints': constraints,  # Include constraints from main config
        'solver_specific': {
            'population_size': 50,
            'n_generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }
    }
    
    # Solver-specific defaults
    solver_defaults = {
        'ga': {
            'solver_specific': {
                'population_size': 100,
                'n_generations': 200,
                'mutation': {
                    'eta': 20,
                    'prob': 0.1
                },
                'crossover': {
                    'eta': 15,
                    'prob': 0.8
                }
            }
        },
        'adaptive_ga': {
            'solver_specific': {
                'population_size': 100,
                'n_generations': 200,
                'initial_mutation_rate': 0.1,
                'initial_crossover_rate': 0.8,
                'min_mutation_rate': 0.01,
                'max_mutation_rate': 0.5,
                'min_crossover_rate': 0.5,
                'max_crossover_rate': 0.95,
                'adaptation_rate': 0.1
            }
        },
        'pso': {
            'solver_specific': {
                'n_particles': 50,
                'n_iterations': 100,
                'w': 0.7,  # Inertia weight
                'c1': 1.5,  # Cognitive parameter
                'c2': 1.5   # Social parameter
            }
        }
    }
    
    # Get solver-specific defaults
    solver_default = solver_defaults.get(solver_name, defaults)
    
    # Deep merge configs
    result = defaults.copy()
    result.update(solver_default)
    if solver_config:
        for key, value in solver_config.items():
            if isinstance(value, dict) and key in result:
                result[key].update(value)
            else:
                result[key] = value
    
    return result
