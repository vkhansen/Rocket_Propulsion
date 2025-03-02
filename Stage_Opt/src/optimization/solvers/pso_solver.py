"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class ParticleSwarmOptimizer(BaseSolver):
    """Particle Swarm Optimization solver implementation."""

    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, swarm_size=50, max_iterations=100, inertia=0.5, cognitive_factor=0.5, social_factor=0.5):
        """Initialize PSO solver with direct problem parameters and PSO-specific settings.

        Args:
            G0: Gravitational constant
            ISP: List of specific impulse values for each stage
            EPSILON: List of structural coefficients for each stage
            TOTAL_DELTA_V: Required total delta-v
            bounds: List of (min, max) bounds for each variable
            swarm_size: Number of particles in the swarm
            max_iterations: Maximum number of iterations
            inertia: Inertia coefficient
            cognitive_factor: Cognitive coefficient
            social_factor: Social coefficient
        """
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds)
        
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive_factor = cognitive_factor
        self.social_factor = social_factor
        
        # Initialize swarm positions and velocities
        self.swarm = None
        self.velocities = None
        
        logger.debug(
            f"Initialized {self.name} with parameters: swarm_size={swarm_size}, max_iterations={max_iterations}, "
            f"inertia={inertia}, cognitive_factor={cognitive_factor}, social_factor={social_factor}"
        )

    def initialize_swarm(self):
        """Initialize particle positions and velocities within bounds."""
        try:
            n_vars = len(self.bounds)
            self.swarm = np.zeros((self.swarm_size, n_vars))
            self.velocities = np.zeros((self.swarm_size, n_vars))
            for i in range(n_vars):
                lower, upper = self.bounds[i]
                self.swarm[:, i] = np.random.uniform(lower, upper, self.swarm_size)
                self.velocities[:, i] = np.random.uniform(-abs(upper-lower), abs(upper-lower), self.swarm_size)
            return True
        except Exception as e:
            logger.error(f"Error initializing swarm: {str(e)}")
            return False

    def evaluate_swarm(self):
        """Evaluate the fitness of each particle in the swarm."""
        try:
            fitness_values = np.zeros(self.swarm_size)
            for i in range(self.swarm_size):
                try:
                    fitness = objective_with_penalty(
                        dv=self.swarm[i],
                        G0=self.G0,
                        ISP=self.ISP,
                        EPSILON=self.EPSILON,
                        TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                        return_tuple=False
                    )
                    fitness_values[i] = fitness if fitness is not None else float('-inf')
                except Exception as inner_e:
                    logger.error(f"Error evaluating particle {i}: {str(inner_e)}")
                    fitness_values[i] = float('-inf')
            return fitness_values
        except Exception as e:
            logger.error(f"Error evaluating swarm: {str(e)}")
            return None

    def update_velocities_and_positions(self, personal_best, global_best):
        """Update velocities and positions of the swarm."""
        try:
            r1 = np.random.rand(self.swarm_size, len(self.bounds))
            r2 = np.random.rand(self.swarm_size, len(self.bounds))
            cognitive_velocity = self.cognitive_factor * r1 * (personal_best - self.swarm)
            social_velocity = self.social_factor * r2 * (global_best - self.swarm)
            self.velocities = self.inertia * self.velocities + cognitive_velocity + social_velocity
            self.swarm = self.swarm + self.velocities
            
            # Ensure particles are within bounds
            for i in range(len(self.bounds)):
                lower, upper = self.bounds[i]
                self.swarm[:, i] = np.clip(self.swarm[:, i], lower, upper)
            return True
        except Exception as e:
            logger.error(f"Error updating velocities and positions: {str(e)}")
            return False

    def solve(self, initial_guess, bounds):
        """Solve using Particle Swarm Optimization."""
        try:
            logger.info("Starting PSO optimization...")
            
            if not self.initialize_swarm():
                raise ValueError("Failed to initialize swarm")
            
            # Initialize personal bests and global best
            personal_best = self.swarm.copy()
            personal_best_fitness = self.evaluate_swarm()
            global_best_idx = np.argmax(personal_best_fitness)
            global_best = personal_best[global_best_idx].copy()
            global_best_fitness = personal_best_fitness[global_best_idx]
            
            start_time = time.time()
            
            for iteration in range(self.max_iterations):
                current_fitness = self.evaluate_swarm()
                
                # Update personal bests
                for i in range(self.swarm_size):
                    if current_fitness[i] > personal_best_fitness[i]:
                        personal_best[i] = self.swarm[i].copy()
                        personal_best_fitness[i] = current_fitness[i]
                        
                        # Update global best
                        if current_fitness[i] > global_best_fitness:
                            global_best = self.swarm[i].copy()
                            global_best_fitness = current_fitness[i]
                
                if not self.update_velocities_and_positions(personal_best, global_best):
                    raise ValueError("Error updating swarm positions")
                
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations} - Global Best Fitness: {global_best_fitness:.6f}")
            
            execution_time = time.time() - start_time
            
            return self.process_results(
                x=global_best,
                success=True,
                message="PSO Optimization completed successfully",
                n_iterations=self.max_iterations,
                n_function_evals=self.max_iterations * self.swarm_size,
                time=execution_time
            )
        except Exception as e:
            logger.error(f"Error in PSO solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
