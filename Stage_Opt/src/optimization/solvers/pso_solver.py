"""Particle Swarm Optimization solver implementation."""
import time
import numpy as np
from typing import Dict, List, Tuple
from ...utils.config import logger as global_logger
from .base_solver import BaseSolver
from .solver_logging import setup_solver_logger

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
        
    def initialize_swarm(self, other_solver_results=None):
        """Initialize particle swarm with positions and velocities.
        
        Args:
            other_solver_results: Optional dictionary of solutions from other solvers
            
        Returns:
            Tuple of (positions, velocities) arrays
        """
        self.logger.debug("Initializing PSO swarm...")
        
        positions = np.zeros((self.population_size, self.n_stages))
        
        # Use more balanced Dirichlet distribution
        alpha = np.ones(self.n_stages) * 10.0  # Reduced from 15.0 for more variation
        
        # More relaxed minimum stage fraction - allow some stages to be smaller
        min_stage_fraction = 0.5 / self.n_stages  # Reduced from 1.0/n_stages
        max_retries = 100  # Prevent infinite loops
        
        self.logger.debug(f"Using min_stage_fraction={min_stage_fraction:.4f} for {self.n_stages} stages")
        
        # Check if we have solutions from other solvers to bootstrap
        bootstrap_solutions = []
        best_bootstrap_solution = None
        best_bootstrap_fitness = float('inf')
        
        # Handle both dictionary format and list format for other_solver_results
        if other_solver_results is not None:
            if isinstance(other_solver_results, dict):
                # Handle dictionary format (key: solver_name, value: result dict)
                self.logger.info(f"Found {len(other_solver_results)} other solver results for bootstrapping (dict format)")
                for solver_name, result in other_solver_results.items():
                    if 'x' in result and np.all(np.isfinite(result['x'])) and len(result['x']) == self.n_stages:
                        bootstrap_solutions.append(result['x'])
                        
                        # Track best bootstrap solution
                        fitness = result.get('fitness', float('inf'))
                        if fitness < best_bootstrap_fitness:
                            best_bootstrap_solution = result['x'].copy()
                            best_bootstrap_fitness = fitness
                            
                        self.logger.debug(f"Added bootstrap solution from {solver_name}: {result['x']}")
            elif isinstance(other_solver_results, list):
                # Handle list format (list of dicts with 'solution' key)
                self.logger.info(f"Found {len(other_solver_results)} other solver results for bootstrapping (list format)")
                for result in other_solver_results:
                    if isinstance(result, dict) and 'solution' in result:
                        solution = result['solution']
                        if np.all(np.isfinite(solution)) and len(solution) == self.n_stages:
                            bootstrap_solutions.append(solution)
                            
                            # Track best bootstrap solution
                            fitness = result.get('fitness', float('inf'))
                            if fitness < best_bootstrap_fitness:
                                best_bootstrap_solution = solution.copy()
                                best_bootstrap_fitness = fitness
                                
                            solver_name = result.get('solver_name', 'unknown')
                            self.logger.debug(f"Added bootstrap solution from {solver_name}: {solution}")
            
            # Store the best bootstrap solution in the base solver for solution rejection
            if best_bootstrap_solution is not None:
                self.best_bootstrap_solution = best_bootstrap_solution
                self.best_bootstrap_fitness = best_bootstrap_fitness
                self.logger.info(f"Stored best bootstrap solution with fitness {best_bootstrap_fitness}")
                
            self.logger.info(f"Using {len(bootstrap_solutions)} valid bootstrap solutions")
        
        # Determine how many particles to initialize from bootstrap solutions
        n_bootstrap = min(len(bootstrap_solutions), self.population_size // 3)
        
        # Initialize first part of swarm with bootstrap solutions
        for i in range(n_bootstrap):
            if i == 0 and len(bootstrap_solutions) > 0:
                # Keep the best solution exactly as is without any noise
                best_solution = bootstrap_solutions[0]
                positions[i] = best_solution.copy()
                self.logger.info(f"Preserving best bootstrap solution exactly: {positions[i]}")
            else:
                # Select a random bootstrap solution
                idx = np.random.randint(0, len(bootstrap_solutions))
                bootstrap_solution = bootstrap_solutions[idx]
                
                # Add some noise to avoid exact copies (reduced from 5% to 0.5% variation)
                noise = np.random.uniform(-0.005, 0.005, self.n_stages)
                noisy_solution = bootstrap_solution * (1 + noise)
                
                # Ensure the solution is valid (sums to TOTAL_DELTA_V)
                noisy_solution = noisy_solution * (self.TOTAL_DELTA_V / np.sum(noisy_solution))
                
                positions[i] = noisy_solution
                self.logger.debug(f"Initialized particle {i} with bootstrap solution: {positions[i]}")
        
        # Initialize remaining particles with Dirichlet distribution
        for i in range(n_bootstrap, self.population_size):
            # Generate balanced proportions using Dirichlet
            props = np.random.dirichlet(alpha)
            
            # Allow more variation but prevent extremely small allocations
            retry_count = 0
            while np.any(props < min_stage_fraction) and retry_count < max_retries:
                props = np.random.dirichlet(alpha)
                retry_count += 1
                if retry_count % 10 == 0:
                    self.logger.debug(f"Retrying initialization {retry_count} times for particle {i}")
            
            # If we hit max retries, redistribute remaining delta-V
            if retry_count >= max_retries:
                self.logger.warning(f"Hit max retries for particle {i}, redistributing stages")
                props = np.clip(props, min_stage_fraction, 1.0)
                props = props / np.sum(props)  # Renormalize
            
            positions[i] = props * self.TOTAL_DELTA_V
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

    def solve(self, initial_guess, bounds, other_solver_results=None):
        """Solve using Particle Swarm Optimization.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            other_solver_results: Optional dictionary of solutions from other solvers
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Setup logger at the start of solve() for multiprocessing support
            self.logger = setup_solver_logger("PSO")
            self.logger.info("Starting PSO optimization...")
            start_time = time.time()
            
            # Initialize swarm
            positions, velocities = self.initialize_swarm(other_solver_results)
            
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
                    
                    # Enforce minimum delta-v value for each stage (increased from 10.0 to 50.0)
                    min_dv_value = 50.0  # 50 m/s minimum to avoid very small stages
                    for j in range(self.n_stages):
                        if new_position[j] < min_dv_value:
                            new_position[j] = min_dv_value
                    
                    # Project position to feasible space using improved method with multiple iterations
                    new_position = self.iterative_projection(new_position, max_iterations=5)
                    
                    # Enforce stage balance constraints
                    total_dv = np.sum(new_position)
                    max_stage_dv = 0.8 * total_dv  # No stage should exceed 80% of total
                    
                    # Check and rebalance if any stage exceeds limit
                    max_stage = np.max(new_position)
                    if max_stage > max_stage_dv:
                        self.logger.debug(f"Rebalancing particle {i} - Stage exceeds limit:")
                        self.logger.debug(f"Before rebalance: {new_position}")
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
                            self.logger.debug("Using equal redistribution due to near-zero sum")
                        new_position[other_stages] += excess * props
                        self.logger.debug(f"After rebalance: {new_position}")
                    
                    # Final check for minimum delta-v values
                    for j in range(self.n_stages):
                        if new_position[j] < min_dv_value:
                            # If a stage is still below minimum, try to redistribute from largest stage
                            largest_idx = np.argmax(new_position)
                            if largest_idx != j and new_position[largest_idx] > min_dv_value * 2:
                                # Only redistribute if largest stage has enough to spare
                                deficit = min_dv_value - new_position[j]
                                new_position[j] = min_dv_value
                                new_position[largest_idx] -= deficit
                                # Final projection to ensure constraints
                                new_position = self.project_to_feasible(new_position)
                    
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
