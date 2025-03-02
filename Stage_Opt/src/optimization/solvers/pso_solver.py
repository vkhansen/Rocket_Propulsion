"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class ParticleSwarmOptimizer(BaseSolver):
    """PSO solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config, 
                 n_particles=50, maxiter=100, w=0.6, c1=1.0, c2=1.0):
        """Initialize PSO solver with problem parameters and PSO-specific settings."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.n_particles = n_particles
        self.maxiter = maxiter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        
    def project_to_feasible(self, x):
        """Project solution to feasible space with high precision."""
        x_proj = np.array(x, dtype=np.float64)  # Higher precision
        
        # First ensure bounds constraints
        for i in range(self.n_stages):
            lower, upper = self.bounds[i]
            x_proj[i] = np.clip(x_proj[i], lower, upper)
        
        # Iterative projection to handle numerical precision
        max_iterations = 10
        for _ in range(max_iterations):
            # Normalize to total ΔV
            total = np.sum(x_proj)
            if total > 0:
                x_proj *= self.TOTAL_DELTA_V / total
                
                # Verify constraint
                error = np.abs(np.sum(x_proj) - self.TOTAL_DELTA_V)
                if error <= 1e-10:  # Strict convergence check
                    break
                    
                # Distribute remaining error proportionally
                adjustment = (self.TOTAL_DELTA_V - np.sum(x_proj)) / self.n_stages
                x_proj += adjustment
                
                # Re-check bounds after adjustment
                for i in range(self.n_stages):
                    lower, upper = self.bounds[i]
                    x_proj[i] = np.clip(x_proj[i], lower, upper)
        
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
        
        # Calculate new velocity with reduced cognitive/social terms
        new_velocity = (self.w * velocity + 
                       0.5 * self.c1 * r1 * (p_best - position) +  # Reduced cognitive
                       0.5 * self.c2 * r2 * (g_best - position))   # Reduced social
        
        # Apply stronger velocity damping for stability
        max_velocity = 0.05 * self.TOTAL_DELTA_V  # Reduced to 5% of total ΔV
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
        
    def solve(self, initial_guess, bounds):
        """Solve using Particle Swarm Optimization."""
        try:
            logger.info("Starting PSO optimization...")
            start_time = time.time()
            
            # Initialize particles with feasible solutions
            positions = np.zeros((self.n_particles, self.n_stages))
            velocities = np.zeros_like(positions)
            
            # Generate feasible initial population
            for i in range(self.n_particles):
                positions[i] = self.project_to_feasible(
                    np.random.uniform(low=[b[0] for b in bounds], 
                                    high=[b[1] for b in bounds], 
                                    size=self.n_stages)
                )
                velocities[i] = np.random.uniform(-0.1, 0.1, self.n_stages)
            
            # Initialize personal and global best
            p_best_pos = positions.copy()
            p_best_scores = np.array([float('inf')] * self.n_particles)
            g_best_pos = positions[0].copy()
            g_best_score = float('inf')
            
            # Main optimization loop
            best_violation = float('inf')
            best_x = None
            
            for iteration in range(self.maxiter):
                # Update each particle
                for i in range(self.n_particles):
                    # Project current position
                    curr_pos = self.project_to_feasible(positions[i])
                    positions[i] = curr_pos
                    
                    # Evaluate objective
                    score = objective_with_penalty(
                        dv=curr_pos,
                        G0=self.G0,
                        ISP=self.ISP,
                        EPSILON=self.EPSILON,
                        TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                        return_tuple=False
                    )
                    _, violation = self.check_constraints(curr_pos)
                    
                    # Update personal best
                    if violation < 1e-8 and score < p_best_scores[i]:
                        p_best_pos[i] = curr_pos.copy()
                        p_best_scores[i] = score
                        
                        # Update global best
                        if score < g_best_score:
                            g_best_pos = curr_pos.copy()
                            g_best_score = score
                    
                    # Track best solution
                    if violation < best_violation:
                        best_violation = violation
                        best_x = curr_pos.copy()
                    
                    # Update velocity with reduced cognitive/social terms
                    velocities[i] = self.update_velocity(
                        velocities[i], positions[i], 
                        p_best_pos[i], g_best_pos
                    )
                    
                    # Update position
                    positions[i] += velocities[i]
                    
                    # Project to feasible space
                    positions[i] = self.project_to_feasible(positions[i])
            
            execution_time = time.time() - start_time
            
            # Return best solution found
            if best_violation < 1e-4:
                return self.process_results(
                    x=best_x,
                    success=True,
                    message="Optimization successful",
                    n_iterations=self.maxiter,
                    n_function_evals=self.maxiter * self.n_particles,
                    time=execution_time
                )
            else:
                return self.process_results(
                    x=initial_guess,
                    success=False,
                    message=f"Failed to find feasible solution (violation={best_violation:.2e})",
                    n_iterations=self.maxiter,
                    n_function_evals=self.maxiter * self.n_particles,
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
