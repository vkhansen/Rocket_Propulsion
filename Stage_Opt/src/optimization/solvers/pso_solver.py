"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class ParticleSwarmOptimizer(BaseSolver):
    """PSO solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config, 
                 n_particles=100, maxiter=200, w=0.7, c1=1.5, c2=1.5):
        """Initialize PSO solver with problem parameters and PSO-specific settings."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.n_particles = n_particles  # Increased for better coverage
        self.maxiter = maxiter  # More iterations for convergence
        self.w = w  # Higher inertia for exploration
        self.c1 = c1  # Stronger cognitive component
        self.c2 = c2  # Stronger social component
        self.n_stages = len(bounds)
        
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
        try:
            from scipy.stats import qmc
            
            # Use Latin Hypercube Sampling for better coverage
            sampler = qmc.LatinHypercube(d=self.n_stages)
            samples = sampler.random(n=self.n_particles)
            
            positions = np.zeros((self.n_particles, self.n_stages))
            velocities = np.zeros((self.n_particles, self.n_stages))
            
            for i in range(self.n_particles):
                # Scale samples to bounds
                for j in range(self.n_stages):
                    lower, upper = self.bounds[j]
                    positions[i,j] = samples[i,j] * (upper - lower) + lower
                
                # Project to feasible space using base class method
                positions[i] = super().project_to_feasible(positions[i])
                
                # Initialize velocities based on problem scale
                for j in range(self.n_stages):
                    lower, upper = self.bounds[j]
                    range_j = upper - lower
                    velocities[i,j] = np.random.uniform(-0.2, 0.2) * range_j
                    
            return positions, velocities
            
        except Exception as e:
            logger.warning(f"LHS initialization failed: {str(e)}, using uniform random")
            return self._uniform_random_init()
            
    def _uniform_random_init(self):
        """Fallback uniform random initialization."""
        positions = np.zeros((self.n_particles, self.n_stages))
        velocities = np.zeros((self.n_particles, self.n_stages))
        
        for i in range(self.n_particles):
            # Generate random position
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                positions[i,j] = np.random.uniform(lower, upper)
            
            # Project to feasible space
            positions[i] = super().project_to_feasible(positions[i])
            
            # Initialize velocities
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                range_j = upper - lower
                velocities[i,j] = np.random.uniform(-0.2, 0.2) * range_j
                
        return positions, velocities
        
    def update_velocity(self, velocity, position, p_best, g_best):
        """Update particle velocity with improved stability."""
        r1, r2 = np.random.random(2)
        
        # Calculate new velocity with balanced exploration/exploitation
        new_velocity = (self.w * velocity + 
                       self.c1 * r1 * (p_best - position) +  # Cognitive
                       self.c2 * r2 * (g_best - position))   # Social
        
        # Apply velocity clamping
        max_velocity = 0.2 * self.TOTAL_DELTA_V  # Increased to 20% of total ΔV
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
            
            # Initialize swarm
            positions, velocities = self.initialize_swarm()
            
            # Initialize personal and global best
            p_best_pos = positions.copy()
            p_best_scores = np.array([float('inf')] * self.n_particles)
            g_best_pos = positions[0].copy()
            g_best_score = float('inf')
            
            # Track best solution
            best_feasible_pos = None
            best_feasible_score = float('inf')
            best_violation = float('inf')
            stall_count = 0
            
            for iteration in range(self.maxiter):
                improved = False
                
                # Update each particle
                for i in range(self.n_particles):
                    # Project current position
                    curr_pos = super().project_to_feasible(positions[i])
                    positions[i] = curr_pos
                    
                    # Evaluate with components
                    obj, dv_const, phys_const = self.evaluate_solution(curr_pos, return_components=True)
                    total_violation = dv_const + phys_const
                    score = obj + total_violation
                    
                    # Update personal best if better score and feasible
                    if score < p_best_scores[i]:
                        p_best_pos[i] = curr_pos.copy()
                        p_best_scores[i] = score
                        improved = True
                        
                        # Update global best if this is the best so far
                        if score < g_best_score:
                            g_best_pos = curr_pos.copy()
                            g_best_score = score
                    
                    # Track best feasible solution
                    if total_violation <= self.feasibility_threshold:
                        if obj < best_feasible_score:
                            best_feasible_pos = curr_pos.copy()
                            best_feasible_score = obj
                            improved = True
                    elif total_violation < best_violation:
                        best_violation = total_violation
                    
                    # Update velocity and position
                    velocities[i] = self.update_velocity(
                        velocities[i], positions[i],
                        p_best_pos[i], g_best_pos
                    )
                    positions[i] += velocities[i]
                    
                # Update stall count
                if not improved:
                    stall_count += 1
                else:
                    stall_count = 0
                    
                # Early stopping if stalled
                if stall_count >= 30:  # Increased from previous value
                    break
            
            execution_time = time.time() - start_time
            
            # Return best feasible solution if found
            if best_feasible_pos is not None:
                return self.process_results(
                    x=best_feasible_pos,
                    success=True,
                    message="Optimization successful",
                    n_iterations=iteration + 1,
                    n_function_evals=(iteration + 1) * self.n_particles,
                    time=execution_time
                )
            else:
                # Return best solution found with violation info
                return self.process_results(
                    x=g_best_pos,
                    success=False,
                    message=f"Solution violates constraints (violation={best_violation:.2e})",
                    n_iterations=iteration + 1,
                    n_function_evals=(iteration + 1) * self.n_particles,
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
