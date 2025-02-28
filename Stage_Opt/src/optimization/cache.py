"""Cache and persistence utilities for optimization."""
import os
import pickle
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Dict
from ..utils.config import OUTPUT_DIR, logger

class OptimizationCache:
    """Cache for optimization results and fitness evaluations."""
    
    def __init__(self, cache_file: str = "ga_cache.pkl", max_size: int = 1000):
        """Initialize the cache.
        
        Args:
            cache_file: Name of the file to store cached solutions
            max_size: Maximum number of solutions to cache
        """
        self.cache_file = os.path.join(OUTPUT_DIR, cache_file)
        self.cache_fitness: Dict[Tuple, float] = {}
        self.cache_best_solutions: List[np.ndarray] = []
        self.max_size = max_size
        self.hit_count = 0
        self.load_cache()
    
    def load_cache(self):
        """Load cached solutions from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Convert string keys back to tuples
                    self.cache_fitness = {}
                    for k, v in data.get('cache_fitness', {}).items():
                        if isinstance(k, str):
                            # Handle different string formats
                            k = k.replace('np.float64(', '').replace(')', '')  # Remove numpy type info
                            k = k.strip('()[]')  # Remove brackets
                            try:
                                k = tuple(float(x.strip()) for x in k.split(',') if x.strip())
                            except ValueError:
                                logger.warning(f"Skipping invalid cache key: {k}")
                                continue
                        self.cache_fitness[k] = v
                    self.cache_best_solutions = data.get('cache_best_solutions', [])
                logger.info(f"Loaded {len(self.cache_fitness)} cached fitness values "
                          f"and {len(self.cache_best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache_fitness = {}
            self.cache_best_solutions = []
            
    def clear(self):
        """Clear all cached data."""
        self.cache_fitness = {}
        self.cache_best_solutions = []
        self.hit_count = 0
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                logger.info("Cache file removed")
            except OSError as e:
                logger.warning(f"Failed to remove cache file: {e}")
    
    def save_cache(self):
        """Save cached solutions to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                data = {
                    'cache_fitness': self.cache_fitness,
                    'cache_best_solutions': self.cache_best_solutions
                }
                pickle.dump(data, f)
            logger.info(f"Saved {len(self.cache_fitness)} cached fitness values "
                      f"and {len(self.cache_best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def add(self, x: np.ndarray, fitness: float):
        """Add or update a solution in the cache.
        
        Args:
            x: Solution vector
            fitness: Fitness value
        """
        x_tuple = tuple(x.flatten())
        self.cache_fitness[x_tuple] = fitness
        
        # Update best solutions list
        if len(self.cache_best_solutions) < self.max_size:
            self.cache_best_solutions.append(x.copy())
        else:
            # Replace worst solution if this one is better
            worst_idx = np.argmax([self.cache_fitness.get(tuple(s.flatten()), float('inf')) 
                                 for s in self.cache_best_solutions])
            if fitness < self.cache_fitness.get(tuple(self.cache_best_solutions[worst_idx].flatten()), float('inf')):
                self.cache_best_solutions[worst_idx] = x.copy()
        
        # Maintain cache size
        if len(self.cache_fitness) > self.max_size:
            # Remove oldest entries that aren't in best_solutions
            best_tuples = {tuple(s.flatten()) for s in self.cache_best_solutions}
            to_remove = []
            for k in self.cache_fitness:
                if k not in best_tuples:
                    to_remove.append(k)
                    if len(self.cache_fitness) - len(to_remove) <= self.max_size:
                        break
            for k in to_remove:
                del self.cache_fitness[k]
    
    def has_cached_solution(self, x: np.ndarray) -> bool:
        """Check if solution exists in cache."""
        return tuple(x.flatten()) in self.cache_fitness
    
    def get_cached_fitness(self, x: np.ndarray) -> float:
        """Get cached fitness for a solution."""
        x_tuple = tuple(x.flatten())
        fitness = self.cache_fitness.get(x_tuple)
        if fitness is not None:
            self.hit_count += 1
        return fitness
    
    def get_best_solutions(self, n: int = None) -> List[np.ndarray]:
        """Get n best solutions from cache.
        
        Args:
            n: Number of solutions to return. If None, returns all solutions.
            
        Returns:
            List of best solutions sorted by fitness.
        """
        if not self.cache_best_solutions:
            return []
        
        if n is None:
            n = len(self.cache_best_solutions)
            
        # Sort by fitness
        sorted_solutions = sorted(
            self.cache_best_solutions,
            key=lambda x: self.cache_fitness.get(tuple(x.flatten()), float('inf'))
        )
        return sorted_solutions[:n]