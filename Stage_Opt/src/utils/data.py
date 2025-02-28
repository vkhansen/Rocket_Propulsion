"""Data loading and processing utilities."""
import json
import numpy as np
from .config import logger, OUTPUT_DIR

def load_input_data(filename):
    """Load input data from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Extract global parameters
        parameters = data['parameters']
        
        # Sort stages by stage number
        stages = sorted(data['stages'], key=lambda x: x['stage'])
        
        return parameters, stages
        
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise

def calculate_mass_ratios(dv, ISP, EPSILON, G0=9.81):
    """Calculate stage ratios (Λ) for each stage.
    
    For each stage i, the stage ratio Λᵢ is calculated as:
    Λᵢ = rᵢ/(1 + εᵢ)
    
    where:
    - rᵢ is the mass ratio from the rocket equation: exp(-ΔVᵢ/(g₀*ISPᵢ))
    - εᵢ is the structural coefficient for stage i
    """
    try:
        dv = np.asarray(dv).flatten()
        stage_ratios = []
        
        # Calculate stage ratios for each stage
        for i in range(len(dv)):
            # Calculate mass ratio using rocket equation
            mass_ratio = np.exp(-dv[i] / (G0 * ISP[i]))
            
            # Calculate stage ratio consistently for all stages
            stage_ratio = mass_ratio / (1.0 + EPSILON[i])
            stage_ratios.append(float(stage_ratio))
                
        return np.array(stage_ratios)
    except Exception as e:
        logger.error(f"Error calculating stage ratios: {e}")
        return np.array([float('inf')] * len(dv))

def calculate_payload_fraction(stage_ratios):
    """Calculate payload fraction as the product of stage ratios.
    
    The payload fraction is the product of all stage ratios:
    PF = ∏ᵢ Λᵢ = ∏ᵢ (rᵢ/(1 + εᵢ))
    """
    try:
        if any(r <= 0 for r in stage_ratios):
            return 0.0
        return float(np.prod(stage_ratios))
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {e}")
        return 0.0

class OptimizationCache:
    """Cache for optimization results."""
    
    def __init__(self, max_size=1000):
        """Initialize cache with maximum size."""
        self.max_size = max_size
        self.fitness_cache = {}  # Maps solution tuple to fitness
        self.best_solutions = []  # List of best solutions found
        self.hit_count = 0
        logger.info(f"Loaded {len(self.fitness_cache)} cached fitness values and {len(self.best_solutions)} best solutions")
    
    def clear(self):
        """Clear all cached data."""
        self.fitness_cache.clear()
        self.best_solutions.clear()
        self.hit_count = 0
    
    def add(self, x, fitness):
        """Add or update a solution in the cache."""
        x_tuple = tuple(x.flatten())
        self.fitness_cache[x_tuple] = fitness
        
        # Update best solutions list
        if len(self.best_solutions) < self.max_size:
            self.best_solutions.append(x)
        else:
            # Replace worst solution if this one is better
            worst_idx = np.argmax([self.fitness_cache.get(tuple(s.flatten()), float('inf')) 
                                 for s in self.best_solutions])
            if fitness < self.fitness_cache.get(tuple(self.best_solutions[worst_idx].flatten()), float('inf')):
                self.best_solutions[worst_idx] = x.copy()
        
        # Maintain cache size
        if len(self.fitness_cache) > self.max_size:
            # Remove oldest entries that aren't in best_solutions
            best_tuples = {tuple(s.flatten()) for s in self.best_solutions}
            to_remove = []
            for k in self.fitness_cache:
                if k not in best_tuples:
                    to_remove.append(k)
                    if len(self.fitness_cache) - len(to_remove) <= self.max_size:
                        break
            for k in to_remove:
                del self.fitness_cache[k]
    
    def has_cached_solution(self, x):
        """Check if solution exists in cache."""
        return tuple(x.flatten()) in self.fitness_cache
    
    def get_cached_fitness(self, x):
        """Get cached fitness for a solution."""
        x_tuple = tuple(x.flatten())
        if x_tuple in self.fitness_cache:
            self.hit_count += 1
            return self.fitness_cache[x_tuple]
        return None
    
    def get_best_solutions(self, n=None):
        """Get n best solutions from cache."""
        if not self.best_solutions:
            return []
        
        if n is None:
            n = len(self.best_solutions)
            
        # Sort by fitness
        sorted_solutions = sorted(
            self.best_solutions,
            key=lambda x: self.fitness_cache.get(tuple(x.flatten()), float('inf'))
        )
        return sorted_solutions[:n]
