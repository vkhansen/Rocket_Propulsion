"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from typing import Dict, List, Tuple
from ...utils.config import logger
from .base_solver import BaseSolver

class ParticleSwarmOptimizer(BaseSolver):
    """PSO solver implementation."""
    
    def __init__(self, G0: float, ISP: List[float], EPSILON: List[float], 
                 TOTAL_DELTA_V: float, bounds: List[Tuple[float, float]], config: Dict,
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0):
        """Initialize PSO solver with problem parameters and PSO-specific settings."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        
        # PSO-specific parameters
        self.w = float(w)  # Inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = float(c1)  # Cognitive parameter
        self.c2 = float(c2)  # Social parameter
        
    def initialize_swarm(self):
        """Initialize particle swarm with positions and velocities."""
        # Initialize positions using LHS
        positions = self.initialize_population_lhs()
        
        # Initialize velocities relative to stage ranges
        velocities = np.zeros((self.population_size, self.n_stages), dtype=np.float64)
        for i in range(self.population_size):
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                range_j = float(upper - lower)
                velocities[i,j] = np.random.uniform(-0.1, 0.1) * range_j
                
        return positions, velocities
        
    def update_velocity(self, velocity, position, p_best, g_best, iteration):
        """Update particle velocity with improved stability and adaptive parameters."""
        r1, r2 = np.random.random(2)
        
        # Adaptive inertia weight
        w = self.w - (self.w - self.w_min) * (iteration / self.max_iterations)
        
        # Calculate cognitive and social components
        cognitive = self.c1 * r1 * (p_best - position)
        social = self.c2 * r2 * (g_best - position)
        
        # Scale velocity components by stage ranges
        new_velocity = np.zeros_like(velocity, dtype=np.float64)
        for j in range(self.n_stages):
            lower, upper = self.bounds[j]
            range_j = float(upper - lower)
            
            # Apply stage-specific velocity updates
            new_velocity[j] = (w * velocity[j] + 
                             cognitive[j] * range_j * 0.1 +
                             social[j] * range_j * 0.1)
            
            # Velocity clamping relative to stage range
            max_velocity = 0.2 * range_j
            new_velocity[j] = np.clip(new_velocity[j], -max_velocity, max_velocity)
        
        return new_velocity
        
    def solve(self, initial_guess, bounds):
        """Solve using Particle Swarm Optimization."""
        try:
            logger.info("Starting PSO optimization...")
            start_time = time.time()
            
            # Initialize swarm
            positions, velocities = self.initialize_swarm()
            
            # Initialize personal and global best
            p_best_pos = positions.copy()
            p_best_scores = np.full(self.population_size, float('inf'), dtype=np.float64)
            g_best_pos = positions[0].copy()
            g_best_score = float('inf')
            
            stall_count = 0
            for iteration in range(self.max_iterations):
                improved = False
                
                # Evaluate all particles
                for i in range(self.population_size):
                    # Project position to feasible space
                    positions[i] = self.iterative_projection(positions[i])
                    
                    # Check feasibility and evaluate
                    is_feasible, violation = self.check_feasibility(positions[i])
                    score = self.evaluate_solution(positions[i])  # Remove [0] indexing
                    
                    # Update personal best
                    if score < p_best_scores[i]:
                        p_best_pos[i] = positions[i].copy()
                        p_best_scores[i] = score
                        
                        # Update global best if better feasible solution found
                        if self.update_best_solution(positions[i], score, is_feasible, violation):
                            g_best_pos = positions[i].copy()
                            g_best_score = score
                            improved = True
                            
                # Update velocities and positions
                for i in range(self.population_size):
                    velocities[i] = self.update_velocity(
                        velocities[i], positions[i], p_best_pos[i], g_best_pos, iteration
                    )
                    positions[i] += velocities[i]
                    
                # Check convergence
                if not improved:
                    stall_count += 1
                    if stall_count >= self.stall_limit:
                        break
                else:
                    stall_count = 0
                    
            execution_time = time.time() - start_time
            
            # Return best feasible solution found
            if self.best_feasible is not None:
                return self.process_results(
                    x=self.best_feasible,
                    success=True,
                    message="PSO optimization completed successfully",
                    n_iterations=iteration + 1,
                    n_function_evals=self.population_size * (iteration + 1),
                    time=execution_time
                )
            else:
                return self.process_results(
                    x=g_best_pos,  # Return best position even if infeasible
                    success=False,
                    message="No feasible solution found",
                    n_iterations=iteration + 1,
                    n_function_evals=self.population_size * (iteration + 1),
                    time=execution_time
                )
                
        except Exception as e:
            logger.error(f"PSO optimization failed: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
