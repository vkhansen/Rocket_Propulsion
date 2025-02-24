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
                    self.fitness_cache = data.get('fitness_cache', {})
                    self.best_solutions = data.get('best_solutions', [])
                logger.info(f"Loaded {len(self.fitness_cache)} cached fitness values "
                          f"and {len(self.best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def save_cache(self):
        """Save cached solutions to file."""
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'fitness_cache': self.fitness_cache,
                    'best_solutions': self.best_solutions
                }, f)
            logger.info(f"Saved cache with {len(self.fitness_cache)} fitness values "
                       f"and {len(self.best_solutions)} best solutions")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_cached_fitness(self, solution: np.ndarray) -> float:
        """Get cached fitness value for a solution."""
        solution_tuple = tuple(solution.flatten())
        return self.fitness_cache.get(solution_tuple)
    
    def cache_fitness(self, solution: np.ndarray, fitness: float):
        """Cache fitness value for a solution."""
        solution_tuple = tuple(solution.flatten())
        self.fitness_cache[solution_tuple] = fitness
    
    def add_best_solution(self, solution: np.ndarray, max_solutions: int = 10):
        """Add a solution to the best solutions list."""
        self.best_solutions.append(solution.copy())
        if len(self.best_solutions) > max_solutions:
            self.best_solutions.pop(0)  # Remove oldest solution
    
    def get_best_solutions(self) -> List[np.ndarray]:
        """Get list of best solutions."""
        return self.best_solutions.copy()