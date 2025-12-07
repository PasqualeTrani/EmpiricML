import time
import functools
import logging
from pathlib import Path

import polars as pl  # type: ignore

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
    
    @functools.wraps(func)
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


# ------------------------------------------------------------------------------------------
# FUNCTIONS 
# ------------------------------------------------------------------------------------------