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
        self.cache_fitness: Dict[str, float] = {}
        self.cache_best_solutions: List[np.ndarray] = []
        self.max_size = max_size
        self.hit_count = 0
        self.load_cache()
    
    def _get_key(self, x: np.ndarray) -> str:
        """Generate a cache key for a solution vector.
        
        Uses a string representation of the rounded array for better cache hits
        on nearly identical solutions.
        """
        try:
            # Handle NaN and inf values
            if np.any(~np.isfinite(x)):
                return None
                
            # Round to 6 decimal places for cache key
            rounded = np.round(x, decimals=6)
            return np.array2string(rounded, precision=6, separator=',', suppress_small=True)
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return None
            
    def load_cache(self):
        """Load cached solutions from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache_fitness = data.get('fitness', {})
                    self.cache_best_solutions = data.get('best_solutions', [])
                logger.info(f"Loaded {len(self.cache_fitness)} cached fitness values "
                          f"and {len(self.cache_best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache_fitness = {}
            self.cache_best_solutions = []
            
    def clear(self):
        """Clear all cached data."""
        self.cache_fitness.clear()
        self.cache_best_solutions.clear()
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
                pickle.dump({
                    'fitness': self.cache_fitness,
                    'best_solutions': self.cache_best_solutions
                }, f)
            logger.info(f"Saved {len(self.cache_fitness)} cached fitness values "
                      f"and {len(self.cache_best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def has_cached_solution(self, x: np.ndarray) -> bool:
        """Check if solution exists in cache."""
        key = self._get_key(x)
        return key is not None and key in self.cache_fitness
    
    def get_cached_fitness(self, x: np.ndarray) -> float:
        """Get cached fitness for a solution."""
        key = self._get_key(x)
        if key is not None and key in self.cache_fitness:
            self.hit_count += 1
            return self.cache_fitness[key]
        return None
    
    def add(self, x: np.ndarray, fitness: float):
        """Add or update a solution in the cache."""
        key = self._get_key(x)
        if key is None:
            return
            
        self.cache_fitness[key] = fitness
        
        # Update best solutions list
        if len(self.cache_best_solutions) < self.max_size:
            self.cache_best_solutions.append(x.copy())
        elif fitness > min(self.cache_fitness.values()):
            # Replace worst solution
            worst_idx = np.argmin([
                self.cache_fitness.get(self._get_key(s), float('-inf'))
                for s in self.cache_best_solutions
            ])
            self.cache_best_solutions[worst_idx] = x.copy()
        
        # Trim cache if too large
        if len(self.cache_fitness) > self.max_size:
            # Remove oldest entries
            keys = list(self.cache_fitness.keys())
            for old_key in keys[:-self.max_size]:
                del self.cache_fitness[old_key]
    
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
            key=lambda x: self.cache_fitness.get(self._get_key(x), float('inf'))
        )
        return sorted_solutions[:n]