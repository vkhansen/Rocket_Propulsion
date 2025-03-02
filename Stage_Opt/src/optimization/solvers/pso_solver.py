"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class ParticleSwarmOptimizer(BaseSolver):
    """PSO solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config, 
                 n_particles=150, maxiter=300, w=0.9, c1=2.0, c2=2.0):
        """Initialize PSO solver with problem parameters and PSO-specific settings."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.n_particles = n_particles
        self.maxiter = maxiter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_stages = len(bounds)
        self.w_min = 0.4
        self.precision_threshold = 1e-6
        self.max_projection_iterations = 10
        
    def iterative_projection(self, x):
        """Project solution to feasible space using iterative refinement."""
        x_proj = np.array(x, dtype=np.float64)
        
        for _ in range(self.max_projection_iterations):
            # First ensure bounds constraints
            for i in range(self.n_stages):
                lower, upper = self.bounds[i]
                x_proj[i] = np.clip(x_proj[i], lower, upper)
            
            # Check total ΔV constraint using relative error
            total = np.sum(x_proj)
            rel_error = abs(total - self.TOTAL_DELTA_V) / self.TOTAL_DELTA_V
            
            if rel_error <= self.precision_threshold:
                break
                
            # Scale to match total ΔV
            x_proj *= self.TOTAL_DELTA_V / total
            
            # Re-check bounds after scaling
            for i in range(self.n_stages):
                lower, upper = self.bounds[i]
                x_proj[i] = np.clip(x_proj[i], lower, upper)
                
            # Distribute any remaining error proportionally
            remaining = self.TOTAL_DELTA_V - np.sum(x_proj)
            if abs(remaining) / self.TOTAL_DELTA_V > self.precision_threshold:
                # Scale adjustment by stage values
                total_stage_values = np.sum(x_proj)
                if total_stage_values > 0:
                    adjustments = (remaining * x_proj) / total_stage_values
                    x_proj += adjustments
                
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
            
            # Scale samples to stage-specific ranges
            for i in range(self.n_particles):
                for j in range(self.n_stages):
                    lower, upper = self.bounds[j]
                    positions[i,j] = lower + samples[i,j] * (upper - lower)
                
                # Project to feasible space
                positions[i] = self.iterative_projection(positions[i])
                
                # Initialize velocities relative to stage ranges
                for j in range(self.n_stages):
                    lower, upper = self.bounds[j]
                    range_j = upper - lower
                    velocities[i,j] = np.random.uniform(-0.1, 0.1) * range_j
                    
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
            positions[i] = self.iterative_projection(positions[i])
            
            # Initialize velocities relative to stage ranges
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                range_j = upper - lower
                velocities[i,j] = np.random.uniform(-0.1, 0.1) * range_j
                
        return positions, velocities
        
    def update_velocity(self, velocity, position, p_best, g_best, iteration):
        """Update particle velocity with improved stability and adaptive parameters."""
        r1, r2 = np.random.random(2)
        
        # Adaptive inertia weight
        w = self.w - (self.w - self.w_min) * (iteration / self.maxiter)
        
        # Calculate cognitive and social components
        cognitive = self.c1 * r1 * (p_best - position)
        social = self.c2 * r2 * (g_best - position)
        
        # Scale velocity components by stage ranges
        new_velocity = np.zeros_like(velocity)
        for j in range(self.n_stages):
            lower, upper = self.bounds[j]
            range_j = upper - lower
            
            # Apply stage-specific velocity updates
            new_velocity[j] = (w * velocity[j] + 
                             cognitive[j] * range_j * 0.1 +
                             social[j] * range_j * 0.1)
            
            # Velocity clamping relative to stage range
            max_velocity = 0.2 * range_j
            new_velocity[j] = np.clip(new_velocity[j], -max_velocity, max_velocity)
        
        return new_velocity
        
    def check_feasibility(self, x):
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
        if np.abs(total - self.TOTAL_DELTA_V) / self.TOTAL_DELTA_V > self.precision_threshold:
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
                # Evaluate all particles
                for i in range(self.n_particles):
                    # Project position to feasible space
                    positions[i] = self.iterative_projection(positions[i])
                    
                    # Check feasibility and evaluate
                    is_feasible, violation = self.check_feasibility(positions[i])
                    score = self.evaluate_solution(positions[i])[0]
                    
                    # Update personal best
                    if score < p_best_scores[i]:
                        if is_feasible or violation < best_violation:
                            p_best_pos[i] = positions[i].copy()
                            p_best_scores[i] = score
                            
                            # Update global best if better feasible solution found
                            if is_feasible and score < best_feasible_score:
                                best_feasible_pos = positions[i].copy()
                                best_feasible_score = score
                                g_best_pos = positions[i].copy()
                                g_best_score = score
                                stall_count = 0
                            elif not is_feasible and violation < best_violation:
                                best_violation = violation
                                
                # Update velocities and positions
                for i in range(self.n_particles):
                    velocities[i] = self.update_velocity(
                        velocities[i], positions[i], p_best_pos[i], g_best_pos, iteration
                    )
                    positions[i] += velocities[i]
                    
                # Check convergence
                stall_count += 1
                if stall_count >= 30:  # No improvement in 30 iterations
                    break
                    
            # Return best feasible solution found
            if best_feasible_pos is not None:
                return {
                    'x': best_feasible_pos,
                    'success': True,
                    'message': f"Found feasible solution after {iteration} iterations",
                    'execution_time': time.time() - start_time
                }
            else:
                return {
                    'x': g_best_pos,
                    'success': False, 
                    'message': f"No feasible solution found after {iteration} iterations",
                    'execution_time': time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"PSO optimization failed: {str(e)}")
            return {
                'x': np.zeros(self.n_stages),
                'success': False,
                'message': str(e),
                'execution_time': time.time() - start_time
            }
