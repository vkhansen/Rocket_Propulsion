"""Logging utilities for the optimization system."""

import logging
import logging.handlers
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

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
    """Set up logging with both file and console output.
    
    Args:
        name: Logger name (usually module or class name)
        log_dir: Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create and configure handlers
    # File handler (thread-safe with rotation)
    file_handler = ThreadSafeRotatingFileHandler(
        filename=os.path.join(log_dir, f"{name}.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Create async queue for high-volume logging
    async_queue = AsyncLogQueue()
    
    # Create async handler for debug logs
    async_handler = AsyncHandler(async_queue)
    async_handler.setLevel(logging.DEBUG)
    async_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(async_handler)
    
    return logger
