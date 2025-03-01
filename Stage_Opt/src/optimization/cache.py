"""Cache and persistence utilities for optimization."""
import os
import pickle
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
import re
from ..utils.config import OUTPUT_DIR, logger

class OptimizationCache:
    """Cache for optimization results and fitness evaluations."""
    
    def __init__(self, cache_file: str = "optimization_cache.pkl", max_size: int = 1000):
        """Initialize the cache.
        
        Args:
            cache_file: Name of the file to store cached solutions
            max_size: Maximum number of solutions to cache
        """
        # Sanitize cache filename to remove invalid characters
        cache_file = re.sub(r'[<>:"/\\|?*]', '_', cache_file)
        self.cache_file = os.path.join(OUTPUT_DIR, cache_file)
        self.max_size = max_size
        self.cache: Dict[Tuple[float, ...], float] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.debug(f"Loaded {len(self.cache)} cached solutions")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} solutions to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, key: Tuple[float, ...]) -> Optional[float]:
        """Get a cached fitness value.
        
        Args:
            key: Solution vector as a tuple
            
        Returns:
            float: Cached fitness value if found, None otherwise
        """
        return self.cache.get(key)
    
    def set(self, key: Tuple[float, ...], value: float):
        """Cache a fitness value.
        
        Args:
            key: Solution vector as a tuple
            value: Fitness value to cache
        """
        # Enforce maximum cache size
        if len(self.cache) >= self.max_size:
            # Remove random entry
            random_key = next(iter(self.cache))
            del self.cache[random_key]
        
        self.cache[key] = value
        
        # Periodically save cache to disk
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                logger.debug("Cache file removed")
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
    
    def __len__(self):
        """Get number of cached solutions."""
        return len(self.cache)
    
    def __contains__(self, key: Tuple[float, ...]):
        """Check if solution is in cache."""
        return key in self.cache