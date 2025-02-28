"""Cache and persistence utilities for optimization."""
import os
import pickle
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Dict
from ..utils.config import OUTPUT_DIR, logger

class OptimizationCache:
    """Cache for optimization results and fitness evaluations."""
    
    def __init__(self, cache_file: str = "ga_cache.pkl"):
        """Initialize the cache.
        
        Args:
            cache_file: Name of the file to store cached solutions
        """
        self.cache_file = os.path.join(OUTPUT_DIR, cache_file)
        self.fitness_cache: Dict[Tuple, float] = {}
        self.best_solutions: List[np.ndarray] = []
        self.load_cache()
    
    def load_cache(self):
        """Load cached solutions from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Convert string keys back to tuples
                    self.fitness_cache = {}
                    for k, v in data.get('fitness_cache', {}).items():
                        if isinstance(k, str):
                            # Handle different string formats
                            k = k.replace('np.float64(', '').replace(')', '')  # Remove numpy type info
                            k = k.strip('()[]')  # Remove brackets
                            try:
                                k = tuple(float(x.strip()) for x in k.split(',') if x.strip())
                            except ValueError:
                                logger.warning(f"Skipping invalid cache key: {k}")
                                continue
                        self.fitness_cache[k] = v
                    self.best_solutions = data.get('best_solutions', [])
                logger.info(f"Loaded {len(self.fitness_cache)} cached fitness values "
                          f"and {len(self.best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.fitness_cache = {}
            self.best_solutions = []
    
    def save_cache(self):
        """Save cached solutions to file."""
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Limit cache size by keeping only the most recent solutions
            if len(self.best_solutions) > 1000:
                self.best_solutions = self.best_solutions[-1000:]
            
            # Convert keys to strings to avoid pickle issues
            safe_cache = {
                'fitness_cache': {str(k): v for k, v in self.fitness_cache.items()},
                'best_solutions': self.best_solutions
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(safe_cache, f, protocol=4)  # Use protocol 4 for better handling of large objects
            
            logger.info(f"Saved cache with {len(self.fitness_cache)} fitness values "
                       f"and {len(self.best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            # Try to save to a temporary location if main location fails
            try:
                temp_cache = os.path.join(os.path.dirname(self.cache_file), 
                                        'temp_' + os.path.basename(self.cache_file))
                with open(temp_cache, 'wb') as f:
                    pickle.dump(safe_cache, f, protocol=4)
                logger.info(f"Saved cache to temporary file: {temp_cache}")
            except Exception as e2:
                logger.warning(f"Failed to save to temporary cache: {e2}")
    
    def get_cached_fitness(self, solution: np.ndarray) -> float:
        """Get cached fitness value for a solution."""
        solution_key = tuple(float(x) for x in solution)
        return self.fitness_cache.get(solution_key)
    
    def cache_fitness(self, solution: np.ndarray, fitness: float):
        """Cache fitness value for a solution."""
        solution_key = tuple(float(x) for x in solution)
        self.fitness_cache[solution_key] = fitness
    
    def add_best_solution(self, solution: np.ndarray, max_solutions: int = 10):
        """Add a solution to the best solutions list."""
        self.best_solutions.append(solution.copy())
        if len(self.best_solutions) > max_solutions:
            self.best_solutions.pop(0)  # Remove oldest solution
    
    def get_best_solutions(self) -> List[np.ndarray]:
        """Get list of best solutions."""
        return self.best_solutions.copy()