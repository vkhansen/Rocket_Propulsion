"""Particle Swarm Optimization solver implementation."""
import numpy as np
from .base_solver import BaseSolver
from ...utils.config import logger

class ParticleSwarmOptimizer(BaseSolver):
    """Particle Swarm Optimization solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize PSO solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def _initialize_swarm(self, n_particles, bounds):
        """Initialize particle positions and velocities."""
        n_dim = len(bounds)
        
        # Initialize positions randomly within bounds
        positions = np.zeros((n_particles, n_dim))
        for i in range(n_dim):
            positions[:, i] = np.random.uniform(
                bounds[i][0], bounds[i][1], n_particles
            )
            
        # Initialize velocities as random values between -|bound_range| and |bound_range|
        velocities = np.zeros((n_particles, n_dim))
        for i in range(n_dim):
            bound_range = bounds[i][1] - bounds[i][0]
            velocities[:, i] = np.random.uniform(
                -bound_range, bound_range, n_particles
            )
            
        return positions, velocities
        
    def _evaluate_swarm(self, positions):
        """Evaluate fitness for all particles."""
        n_particles = positions.shape[0]
        fitness = np.zeros(n_particles)
        
        for i in range(n_particles):
            penalty = self.enforce_constraints(positions[i])
            payload_fraction = self.calculate_fitness(positions[i])
            penalty_coeff = self.solver_config.get('penalty_coefficient', 1e3)
            fitness[i] = payload_fraction - penalty_coeff * penalty
            
        return fitness
        
    def solve(self, initial_guess, bounds):
        """Solve using Particle Swarm Optimization."""
        try:
            # Get solver parameters
            n_particles = self.solver_specific.get('n_particles', 50)
            n_iterations = self.solver_specific.get('n_iterations', 100)
            w = self.solver_specific.get('w', 0.7)  # Inertia weight
            c1 = self.solver_specific.get('c1', 1.5)  # Cognitive parameter
            c2 = self.solver_specific.get('c2', 1.5)  # Social parameter
            
            # Initialize swarm
            positions, velocities = self._initialize_swarm(n_particles, bounds)
            
            # Initialize best positions and fitness
            fitness = self._evaluate_swarm(positions)
            personal_best_pos = positions.copy()
            personal_best_fitness = fitness.copy()
            
            global_best_idx = np.argmax(fitness)
            global_best_pos = positions[global_best_idx].copy()
            global_best_fitness = fitness[global_best_idx]
            
            n_dim = len(bounds)
            n_evals = n_particles
            
            # Main PSO loop
            for iteration in range(n_iterations):
                # Update velocities
                r1 = np.random.random((n_particles, n_dim))
                r2 = np.random.random((n_particles, n_dim))
                
                cognitive_velocity = c1 * r1 * (personal_best_pos - positions)
                social_velocity = c2 * r2 * (global_best_pos - positions)
                
                velocities = w * velocities + cognitive_velocity + social_velocity
                
                # Update positions
                positions += velocities
                
                # Enforce bounds
                for i in range(n_dim):
                    positions[:, i] = np.clip(
                        positions[:, i], bounds[i][0], bounds[i][1]
                    )
                    
                # Evaluate new positions
                fitness = self._evaluate_swarm(positions)
                n_evals += n_particles
                
                # Update personal bests
                improved = fitness > personal_best_fitness
                personal_best_pos[improved] = positions[improved]
                personal_best_fitness[improved] = fitness[improved]
                
                # Update global best
                current_best = np.argmax(fitness)
                if fitness[current_best] > global_best_fitness:
                    global_best_pos = positions[current_best].copy()
                    global_best_fitness = fitness[current_best]
                    
            # Calculate final results
            stage_ratios, stage_info = calculate_stage_ratios(
                global_best_pos, self.G0, self.ISP, self.EPSILON
            )
            
            return {
                'success': True,
                'message': "Optimization completed",
                'payload_fraction': global_best_fitness,
                'stages': stage_info,
                'n_iterations': n_iterations,
                'n_function_evals': n_evals
            }
            
        except Exception as e:
            logger.error(f"Error in PSO solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'stages': [],
                'n_iterations': 0,
                'n_function_evals': 0
            }
