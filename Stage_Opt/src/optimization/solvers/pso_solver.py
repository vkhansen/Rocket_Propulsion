"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class ParticleSwarmOptimizer(BaseSolver):
    """PSO solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config=None, n_particles=50, n_iterations=100,
                 w=0.7, c1=2.0, c2=2.0):
        """Initialize PSO solver with parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive weight
        self.c2 = c2  # Social weight
        self.n_stages = len(bounds)
        
    def project_to_feasible(self, x):
        """Project solution to feasible space with high precision."""
        x_proj = np.array(x, dtype=np.float64)  # Higher precision
        
        # Ensure bounds constraints
        for i in range(self.n_stages):
            lower, upper = self.bounds[i]
            x_proj[i] = np.clip(x_proj[i], lower, upper)
        
        # Normalize to total ΔV
        total = np.sum(x_proj)
        if total > 0:
            x_proj *= self.TOTAL_DELTA_V / total
            
            # Verify and adjust for exact constraint
            error = np.abs(np.sum(x_proj) - self.TOTAL_DELTA_V)
            if error > 1e-10:
                # Distribute remaining error proportionally
                adjustment = (self.TOTAL_DELTA_V - np.sum(x_proj)) / self.n_stages
                x_proj += adjustment
                
                # Final bounds check after adjustment
                for i in range(self.n_stages):
                    lower, upper = self.bounds[i]
                    x_proj[i] = np.clip(x_proj[i], lower, upper)
                
                # One final normalization if needed
                total = np.sum(x_proj)
                if np.abs(total - self.TOTAL_DELTA_V) > 1e-10:
                    x_proj *= self.TOTAL_DELTA_V / total
        
        return x_proj
        
    def initialize_swarm(self):
        """Initialize particle swarm with feasible solutions."""
        positions = np.zeros((self.n_particles, self.n_stages))
        velocities = np.zeros((self.n_particles, self.n_stages))
        
        for i in range(self.n_particles):
            # Generate random fractions that sum to 1
            fractions = np.random.random(self.n_stages)
            fractions /= np.sum(fractions)
            
            # Scale by total ΔV and project to feasible space
            positions[i] = self.project_to_feasible(fractions * self.TOTAL_DELTA_V)
            
            # Initialize velocities as small random values
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                range_j = upper - lower
                velocities[i,j] = np.random.uniform(-0.1, 0.1) * range_j
                
        return positions, velocities
        
    def update_velocity(self, velocity, position, p_best, g_best):
        """Update particle velocity with damping for stability."""
        r1, r2 = np.random.random(2)
        
        # Calculate new velocity
        new_velocity = (self.w * velocity + 
                       self.c1 * r1 * (p_best - position) +
                       self.c2 * r2 * (g_best - position))
        
        # Apply velocity damping for stability
        max_velocity = 0.1 * self.TOTAL_DELTA_V  # Max 10% of total ΔV
        new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)
        
        return new_velocity
        
    def check_constraints(self, x):
        """Check if solution satisfies constraints."""
        is_feasible = True
        violation = 0
        
        # Check bounds constraints
        for i in range(self.n_stages):
            lower, upper = self.bounds[i]
            if x[i] < lower or x[i] > upper:
                is_feasible = False
                violation += 1
        
        # Check total ΔV constraint
        total = np.sum(x)
        if np.abs(total - self.TOTAL_DELTA_V) > 1e-10:
            is_feasible = False
            violation += 1
        
        return is_feasible, violation
        
    def solve(self, initial_guess=None, bounds=None):
        """Solve using PSO."""
        try:
            logger.info("Starting PSO optimization...")
            start_time = time.time()
            
            # Initialize swarm
            positions, velocities = self.initialize_swarm()
            
            # Initialize personal and global bests
            p_best_positions = positions.copy()
            p_best_fitness = np.array([float('-inf')] * self.n_particles)
            g_best_position = None
            g_best_fitness = float('-inf')
            
            # Main optimization loop
            for iteration in range(self.n_iterations):
                # Evaluate all particles
                for i in range(self.n_particles):
                    # Project position to feasible space
                    positions[i] = self.project_to_feasible(positions[i])
                    
                    # Check constraints
                    is_feasible, violation = self.check_constraints(positions[i])
                    if not is_feasible:
                        continue  # Skip infeasible solutions
                    
                    # Evaluate fitness only for feasible solutions
                    fitness = objective_with_penalty(
                        dv=positions[i],
                        G0=self.G0,
                        ISP=self.ISP,
                        EPSILON=self.EPSILON,
                        TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                        return_tuple=False
                    )
                    
                    # Update personal best (only if feasible)
                    if fitness > p_best_fitness[i]:
                        p_best_fitness[i] = fitness
                        p_best_positions[i] = positions[i].copy()
                        
                        # Update global best
                        if fitness > g_best_fitness:
                            g_best_fitness = fitness
                            g_best_position = positions[i].copy()
                
                # Update velocities and positions
                for i in range(self.n_particles):
                    # Update velocity
                    velocities[i] = self.update_velocity(
                        velocities[i],
                        positions[i],
                        p_best_positions[i],
                        g_best_position if g_best_position is not None else positions[i]
                    )
                    
                    # Update position
                    positions[i] += velocities[i]
                    
                    # Project to feasible space
                    positions[i] = self.project_to_feasible(positions[i])
            
            execution_time = time.time() - start_time
            
            # Final check of best solution
            if g_best_position is not None:
                g_best_position = self.project_to_feasible(g_best_position)
                is_feasible, violation = self.check_constraints(g_best_position)
                
                if is_feasible:
                    return self.process_results(
                        x=g_best_position,
                        success=True,
                        message="Optimization completed successfully",
                        n_iterations=self.n_iterations,
                        n_function_evals=self.n_iterations * self.n_particles,
                        time=execution_time
                    )
            
            # If we get here, no feasible solution was found
            return self.process_results(
                x=initial_guess,
                success=False,
                message="Failed to find feasible solution",
                n_iterations=self.n_iterations,
                n_function_evals=self.n_iterations * self.n_particles,
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
