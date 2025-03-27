"""Logging utilities for the optimization system."""

import logging
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

# Keeping these classes for backward compatibility, but they won't be used in console-only mode
class ThreadSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A thread-safe version of RotatingFileHandler."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        
    def emit(self, record):
        """Thread-safe emit with retry logic."""
        tries = 0
        while tries < 3:  # Retry up to 3 times
            try:
                with self.lock:
                    super().emit(record)
                break
            except Exception:
                tries += 1
                time.sleep(0.1)  # Small delay before retry

class AsyncLogQueue:
    """Asynchronous logging queue to handle high-volume logging."""
    
    def __init__(self, max_workers=2):
        self.queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def _process_queue(self):
        """Process queued log records."""
        while self.running:
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:  # Shutdown signal
                    break
                self.executor.submit(logging.getLogger().handle, record)
            except Queue.Empty:
                continue
                
    def stop(self):
        """Stop the logging queue processor."""
        self.running = False
        self.queue.put(None)  # Signal shutdown
        self.worker_thread.join()
        self.executor.shutdown()

class AsyncHandler(logging.Handler):
    """Asynchronous logging handler that uses a queue."""
    
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
    def emit(self, record):
        """Put the record in the queue instead of handling directly."""
        self.queue.queue.put(record)

def setup_logging(name, log_dir="logs"):
    """Set up logging with console output only for improved performance.
    
    Args:
        name: Logger name (usually module or class name)
        log_dir: Directory to store log files (not used with console-only logging)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO for performance
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create console formatter - simplified for better readability
    console_formatter = logging.Formatter(
        f'[{name}] %(levelname)s - %(message)s'
    )
    
    # Console handler only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger
