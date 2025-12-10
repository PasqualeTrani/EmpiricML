import time
from functools import wraps
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import (
    List,
    Dict, 
    Union, 
    Tuple, 
    Callable, 
    Any
)

import polars as pl
import numpy as np

from empml.pipelines import Pipeline
from empml.base import Metric

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------------------------------
# DECORATORS 
# ------------------------------------------------------------------------------------------

def log_execution_time(func):
    """A decorator that logs start, end, and duration of a function."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # 1. Log start
        logging.info(f"[{func_name}] START execution.")
        start_time = time.perf_counter()
        
        # 2. Run the actual function
        result = func(*args, **kwargs)
        
        # 3. Log end and calculate duration
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        logging.info(f"[{func_name}] END execution. Duration: {duration:.4f} seconds.")
        
        return result

    return wrapper

def time_execution(func: Callable) -> Callable:
    """Decorator that returns function result and execution duration."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = round(time.perf_counter() - start, 2)
        return result, duration
    return wrapper


# ------------------------------------------------------------------------------------------
# FUNCTIONS 
# ------------------------------------------------------------------------------------------

@contextmanager
def log_step(step_name: str, verbose: bool):
    """Context manager for logging step execution."""
    if verbose:
        logging.info(f'Start {step_name}...')
    try:
        yield
    finally:
        if verbose:
            logging.info(f'✓ {step_name} ended.')
