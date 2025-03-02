"""Cache implementation for optimization results."""
import os
import json
import time
from typing import Dict, Optional
import numpy as np
from ..utils.config import logger

class OptimizationCache:
    """Cache for optimization results to avoid redundant calculations."""
    
    def __init__(self, cache_file=None, max_size=10000):
        """Initialize the cache.
        
        Args:
            cache_file: Optional file path to persist cache
            max_size: Maximum number of entries in cache
        """
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.cache_file = cache_file
        self.max_size = max_size
        
        # Load from file if it exists
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache from file: {str(e)}")
                
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
                
                # Save to file if specified
                if self.cache_file:
                    try:
                        with open(self.cache_file, 'w') as f:
                            json.dump(self.cache, f)
                    except Exception as e:
                        logger.error(f"Error saving cache to file: {str(e)}")
                        
                # Limit cache size
                if len(self.cache) > self.max_size:
                    self.cache = dict(list(self.cache.items())[-self.max_size:])
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
        
        # Remove file if it exists
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception as e:
                logger.error(f"Error removing cache file: {str(e)}")
                
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