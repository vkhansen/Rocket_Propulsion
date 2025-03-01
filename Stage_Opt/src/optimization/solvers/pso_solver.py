"""Particle Swarm Optimization solver implementation."""
import numpy as np
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

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
        positions = np.random.uniform(
            low=bounds[:, 0],
            high=bounds[:, 1],
            size=(n_particles, n_dim)
        )
        
        # Initialize velocities as random values between -1 and 1
        velocities = np.random.uniform(
            low=-1,
            high=1,
            size=(n_particles, n_dim)
        )
        
        return positions, velocities
        
    def _evaluate_swarm(self, positions):
        """Evaluate fitness for all particles."""
        return np.array([
            objective_with_penalty(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON,
                TOTAL_DELTA_V=self.TOTAL_DELTA_V
            )
            for x in positions
        ])
    
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
            n_particles = int(self.solver_specific.get('n_particles', 50))
            n_iterations = int(self.solver_specific.get('n_iterations', 100))
            w = float(self.solver_specific.get('w', 0.7))  # Inertia weight
            c1 = float(self.solver_specific.get('c1', 2.0))  # Cognitive parameter
            c2 = float(self.solver_specific.get('c2', 2.0))  # Social parameter
            
            # Convert bounds to numpy array
            bounds = np.array(bounds)
            
            # Initialize swarm
            positions, velocities = self._initialize_swarm(n_particles, bounds)
            
            # Initialize best positions and fitness
            fitness = self._evaluate_swarm(positions)
            personal_best_pos = positions.copy()
            personal_best_fitness = fitness.copy()
            
            # Initialize global best
            global_best_idx = np.argmin(personal_best_fitness)
            global_best_pos = personal_best_pos[global_best_idx].copy()
            global_best_fitness = personal_best_fitness[global_best_idx]
            
            # Main optimization loop
            n_evals = n_particles
            for iteration in range(n_iterations):
                # Update velocities
                r1, r2 = np.random.rand(2)
                velocities = (w * velocities +
                            c1 * r1 * (personal_best_pos - positions) +
                            c2 * r2 * (global_best_pos - positions))
                
                # Update positions
                positions = positions + velocities
                
                # Clip positions to bounds
                positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
                
                # Evaluate new positions
                fitness = self._evaluate_swarm(positions)
                n_evals += n_particles
                
                # Update personal bests
                improved = fitness < personal_best_fitness
                personal_best_pos[improved] = positions[improved]
                personal_best_fitness[improved] = fitness[improved]
                
                # Update global best
                min_idx = np.argmin(personal_best_fitness)
                if personal_best_fitness[min_idx] < global_best_fitness:
                    global_best_pos = personal_best_pos[min_idx].copy()
                    global_best_fitness = personal_best_fitness[min_idx]
            
            return self.process_results(
                x=global_best_pos,
                success=True,
                message="PSO optimization completed",
                n_iterations=n_iterations,
                n_function_evals=n_evals,
                time=0.0
            )
            
        except Exception as e:
            logger.error(f"Error in PSO solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
