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
        # Initialize positions using Dirichlet distribution for better balance
        positions = np.zeros((self.population_size, self.n_stages), dtype=np.float64)
        # Increase alpha for more uniform distribution
        alpha = np.ones(self.n_stages) * 15.0  # Increased from 5.0 to encourage more even distribution
        
        # Minimum fraction of total delta-v per stage based on equal distribution
        min_stage_fraction = 1.0 / self.n_stages  # Equal distribution baseline
        
        for i in range(self.population_size):
            # Generate balanced proportions using Dirichlet
            props = np.random.dirichlet(alpha)
            
            # Ensure minimum stage allocation
            while np.any(props < min_stage_fraction):
                props = np.random.dirichlet(alpha)
            
            positions[i] = props * self.TOTAL_DELTA_V
            
            # Enforce first stage constraints (15-80% of total)
            first_stage_min = 0.15 * self.TOTAL_DELTA_V
            first_stage_max = 0.80 * self.TOTAL_DELTA_V
            if positions[i, 0] < first_stage_min:
                excess = first_stage_min - positions[i, 0]
                positions[i, 0] = first_stage_min
                # Redistribute excess proportionally to other stages
                remaining_props = positions[i, 1:] / positions[i, 1:].sum()
                positions[i, 1:] -= excess * remaining_props
            elif positions[i, 0] > first_stage_max:
                excess = positions[i, 0] - first_stage_max
                positions[i, 0] = first_stage_max
                # Redistribute excess proportionally to other stages
                remaining_props = positions[i, 1:] / positions[i, 1:].sum()
                positions[i, 1:] += excess * remaining_props
            
            # Project to feasible space while preserving proportions
            positions[i] = self.iterative_projection(positions[i])
        
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
                    # Update position with velocity
                    new_position = positions[i] + velocities[i]
                    
                    # Project position to feasible space using improved method
                    new_position = self.iterative_projection(new_position)
                    
                    # Enforce stage balance constraints
                    total_dv = np.sum(new_position)
                    max_stage_dv = 0.8 * total_dv  # No stage should exceed 80% of total
                    
                    # Check and rebalance if any stage exceeds limit
                    max_stage = np.max(new_position)
                    if max_stage > max_stage_dv:
                        excess = max_stage - max_stage_dv
                        max_idx = np.argmax(new_position)
                        new_position[max_idx] = max_stage_dv
                        
                        # Redistribute excess to other stages with safety check for zero-sum
                        other_stages = list(range(self.n_stages))
                        other_stages.remove(max_idx)
                        other_sum = np.sum(new_position[other_stages])
                        if other_sum > 1e-10:  # Use small threshold to avoid division by zero
                            props = new_position[other_stages] / other_sum
                        else:
                            # If other stages have near-zero sum, distribute equally
                            props = np.ones(len(other_stages)) / len(other_stages)
                        new_position[other_stages] += excess * props
                        
                        # Re-project to ensure constraints
                        new_position = self.iterative_projection(new_position)
                    
                    # Update position and evaluate
                    positions[i] = new_position
                    score = self.evaluate_solution(positions[i])
                    
                    # Update personal best
                    if score < p_best_scores[i]:
                        p_best_scores[i] = score
                        p_best_pos[i] = positions[i].copy()
                        
                        # Update global best
                        if score < g_best_score:
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
                        logger.info(f"PSO converged after {iteration + 1} iterations")
                        break
                
                # Log progress periodically
                if (iteration + 1) % 10 == 0:
                    logger.info(f"PSO iteration {iteration + 1}/{self.max_iterations}, "
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
            logger.error(f"Error in PSO optimization: {str(e)}")
            return self.process_results(
                initial_guess,
                success=False,
                message=f"PSO optimization failed: {str(e)}",
                time=time.time() - start_time
            )
