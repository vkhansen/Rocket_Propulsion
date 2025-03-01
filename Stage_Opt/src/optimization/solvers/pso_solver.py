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
            fitness[i] = self.objective_with_penalty(positions[i])
            
        return fitness
        
    def solve(self, initial_guess, bounds):
        """Solve using Particle Swarm Optimization.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Starting PSO optimization...")
            
            # Get solver parameters
            n_particles = self.solver_specific.get('n_particles', 50)
            n_iterations = self.solver_specific.get('n_iterations', 100)
            w = self.solver_specific.get('inertia_weight', 0.7)
            c1 = self.solver_specific.get('cognitive_coeff', 1.5)
            c2 = self.solver_specific.get('social_coeff', 1.5)
            
            # Initialize swarm
            positions, velocities = self._initialize_swarm(n_particles, bounds)
            
            # Initialize best positions and fitness
            fitness = self._evaluate_swarm(positions)
            personal_best_pos = positions.copy()
            personal_best_fit = fitness.copy()
            
            # Global best
            global_best_idx = np.argmax(fitness)
            global_best_pos = positions[global_best_idx].copy()
            global_best_fit = fitness[global_best_idx]
            
            # Optimization loop
            for iteration in range(n_iterations):
                # Update velocities
                r1, r2 = np.random.rand(2, n_particles, len(bounds))
                cognitive = c1 * r1 * (personal_best_pos - positions)
                social = c2 * r2 * (global_best_pos - positions)
                velocities = w * velocities + cognitive + social
                
                # Update positions
                positions += velocities
                
                # Enforce bounds
                for i in range(len(bounds)):
                    positions[:, i] = np.clip(
                        positions[:, i], bounds[i][0], bounds[i][1]
                    )
                
                # Evaluate new positions
                fitness = self._evaluate_swarm(positions)
                
                # Update personal bests
                improved = fitness > personal_best_fit
                personal_best_pos[improved] = positions[improved]
                personal_best_fit[improved] = fitness[improved]
                
                # Update global best
                current_best = np.argmax(fitness)
                if fitness[current_best] > global_best_fit:
                    global_best_pos = positions[current_best].copy()
                    global_best_fit = fitness[current_best]
            
            # Process results
            stage_ratios, mass_ratios = self.calculate_stage_ratios(global_best_pos)
            payload_fraction = self.calculate_fitness(global_best_pos)
            
            return {
                'success': True,
                'x': global_best_pos.tolist(),
                'fun': float(-global_best_fit),  # Convert maximization to minimization
                'payload_fraction': float(payload_fraction),
                'stage_ratios': stage_ratios.tolist(),
                'mass_ratios': mass_ratios.tolist(),
                'stages': self.create_stage_results(global_best_pos, stage_ratios),
                'n_iterations': n_iterations,
                'n_function_evals': n_iterations * n_particles
            }
            
        except Exception as e:
            logger.error(f"Error in PSO solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error in PSO solver: {str(e)}"
            }
