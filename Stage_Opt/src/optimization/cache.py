"""Cache implementation for optimization results."""
import os
import json
import time
from typing import Dict, Optional
import numpy as np
from ..utils.config import logger

class OptimizationCache:
    """Cache for optimization results to avoid redundant calculations."""
    
    def __init__(self):
        """Initialize the cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        
    def _hash_array(self, arr: np.ndarray) -> str:
        """Create a hash for a numpy array.
        
        Args:
            arr: Numpy array to hash
            
        Returns:
            String hash of the array
        """
        try:
            # Round to reduce floating point differences
            arr_rounded = np.round(arr, decimals=6)
            return hash(arr_rounded.tobytes())
        except Exception as e:
            logger.error(f"Error hashing array: {str(e)}")
            return None
            
    def add(self, x: np.ndarray, result: Dict) -> None:
        """Add a result to the cache.
        
        Args:
            x: Input array
            result: Dictionary containing optimization results
        """
        try:
            key = self._hash_array(x)
            if key is not None:
                self.cache[key] = {
                    'result': result,
                    'timestamp': time.time()
                }
        except Exception as e:
            logger.error(f"Error adding to cache: {str(e)}")
            
    def get(self, x: np.ndarray) -> Optional[Dict]:
        """Get a result from the cache if it exists.
        
        Args:
            x: Input array
            
        Returns:
            Cached result dictionary if found, None otherwise
        """
        try:
            key = self._hash_array(x)
            if key is not None and key in self.cache:
                self.hits += 1
                return self.cache[key]['result']
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
            
    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        
    def get_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }