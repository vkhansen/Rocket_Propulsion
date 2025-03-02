"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from typing import Dict, List, Tuple
from ...utils.config import logger as global_logger
from .base_solver import BaseSolver
from .solver_logging import setup_solver_logger
from .stage_constraints import enforce_stage_allocation

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
        self.logger.debug("Initializing PSO swarm...")
        
        positions = np.zeros((self.population_size, self.n_stages))
        
        # Use more balanced Dirichlet distribution
        alpha = np.ones(self.n_stages) * 10.0  # Reduced from 15.0 for more variation
        
        # More relaxed minimum stage fraction - allow some stages to be smaller
        min_stage_fraction = 0.05  # 5% minimum per stage
        max_retries = 100  # Prevent infinite loops
        
        self.logger.debug(f"Using min_stage_fraction={min_stage_fraction:.4f} for {self.n_stages} stages")
        
        for i in range(self.population_size):
            # Generate balanced proportions using Dirichlet
            props = np.random.dirichlet(alpha)
            
            # Allow more variation but prevent extremely small allocations
            retry_count = 0
            while np.any(props < min_stage_fraction) and retry_count < max_retries:
                props = np.random.dirichlet(alpha)
                retry_count += 1
                if retry_count % 10 == 0:
                    self.logger.debug(f"Retrying initialization {retry_count} times for particle {i}")
            
            # If we hit max retries, enforce constraints
            positions[i] = props * self.TOTAL_DELTA_V
            positions[i] = enforce_stage_allocation(positions[i], logger=self.logger)
            
            if i < 3:  # Log first few particles
                self.logger.debug(f"Initial position {i}: {positions[i]}")
        
        # Initialize velocities with smaller scale for better stability
        velocities = np.zeros_like(positions)
        for i in range(self.population_size):
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                range_j = float(upper - lower)
                velocities[i, j] = np.random.uniform(-0.1, 0.1) * range_j
        
        return positions, velocities

    def update_velocity(self, velocity, position, p_best, g_best, iteration):
        """Update particle velocity with improved stability and adaptive parameters."""
        r1, r2 = np.random.random(2)
        
        # Adaptive inertia weight with nonlinear decay
        progress = iteration / self.max_iterations
        w = self.w_min + (self.w - self.w_min) * (1 - progress)**2
        
        # Calculate cognitive and social components with stage-specific scaling
        cognitive = np.zeros_like(velocity, dtype=np.float64)
        social = np.zeros_like(velocity, dtype=np.float64)
        
        # Progressive velocity clamping factors
        v_clamp = 0.5 + 0.3 * (1 - progress)  # Reduces velocity bounds over time
        
        for j in range(self.n_stages):
            lower, upper = self.bounds[j]
            range_j = float(upper - lower)
            
            # Scale cognitive and social components by stage range
            cognitive[j] = self.c1 * r1 * (p_best[j] - position[j]) / range_j
            social[j] = self.c2 * r2 * (g_best[j] - position[j]) / range_j
            
            # Stage-specific velocity adjustments
            if j == 0:
                # Reduce first stage velocity to maintain constraints
                cognitive[j] *= 0.7
                social[j] *= 0.7
            else:
                # Balance velocities for other stages
                cognitive[j] *= 0.9
                social[j] *= 0.9
        
        # Update velocity with better numerical stability
        new_velocity = w * velocity + cognitive + social
        
        # Apply progressive velocity clamping
        for j in range(self.n_stages):
            lower, upper = self.bounds[j]
            range_j = float(upper - lower)
            v_max = v_clamp * range_j
            new_velocity[j] = np.clip(new_velocity[j], -v_max, v_max)
        
        return new_velocity

    def solve(self, initial_guess, bounds):
        """Solve using Particle Swarm Optimization."""
        try:
            # Setup logger at the start of solve() for multiprocessing support
            self.logger = setup_solver_logger("PSO")
            self.logger.info("Starting PSO optimization...")
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
                stage_means = np.mean(positions, axis=0)
                stage_stds = np.std(positions, axis=0)
                self.logger.debug(f"\nIteration {iteration + 1}:")
                self.logger.debug(f"Stage means: {stage_means}")
                self.logger.debug(f"Stage stddevs: {stage_stds}")
                self.logger.debug(f"Current best score: {g_best_score:.6f}")
                self.logger.debug(f"Current best position: {g_best_pos}")
                
                # Evaluate all particles
                for i in range(self.population_size):
                    # Update position with velocity
                    new_position = positions[i] + velocities[i]
                    
                    # Enforce stage allocation constraints
                    new_position = enforce_stage_allocation(new_position, logger=self.logger)
                    
                    # Update position and evaluate
                    positions[i] = new_position
                    score = self.evaluate_solution(positions[i])
                    
                    # Update personal best
                    if score < p_best_scores[i]:
                        p_best_scores[i] = score
                        p_best_pos[i] = positions[i].copy()
                        
                        # Update global best
                        if score < g_best_score:
                            self.logger.debug(f"New best solution found:")
                            self.logger.debug(f"Old best: {g_best_pos} (score: {g_best_score:.6f})")
                            self.logger.debug(f"New best: {positions[i]} (score: {score:.6f})")
                            g_best_score = score
                            g_best_pos = positions[i].copy()
                            improved = True
                            stall_count = 0
                
                # Update velocities
                for i in range(self.population_size):
                    velocities[i] = self.update_velocity(
                        velocities[i], positions[i], p_best_pos[i], g_best_pos, iteration
                    )
                
                if not improved:
                    stall_count += 1
                    if stall_count >= self.stall_limit:
                        self.logger.info(f"PSO converged after {iteration + 1} iterations")
                        break
                
                # Log progress periodically
                if (iteration + 1) % 10 == 0:
                    self.logger.info(f"PSO iteration {iteration + 1}/{self.max_iterations}, "
                                  f"best score: {g_best_score:.6f}")
            
            execution_time = time.time() - start_time
            return self.process_results(
                g_best_pos,
                success=True,
                message="PSO optimization completed successfully",
                n_iterations=iteration + 1,
                n_function_evals=(iteration + 1) * self.population_size,
                time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in PSO optimization: {str(e)}")
            return self.process_results(
                initial_guess,
                success=False,
                message=f"PSO optimization failed: {str(e)}",
                time=time.time() - start_time
            )
