import time
import functools
import logging
from pathlib import Path
from typing import Dict, Union

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

def relative_performance(minimize : bool, x1 : float, x2 : float) -> float:
    """
    Compute the relative performance of a pipeline with score x2 with respect to another of score x1 (reference).
    The same function can be used to compute overfitting.
    """
    if minimize:
        performance = round(((x1 - x2)/(x1)) * 100 ,2)
    else:
        performance = round(((x2 - x1)/(x1)) * 100 ,2)

    return performance


@log_execution_time
def eval_pipeline(
    pipeline : Pipeline,
    train : pl.LazyFrame, 
    valid : pl.LazyFrame, 
    metric : Metric, 
    target : str,
    minimize : bool, 
    eval_overfitting : bool = True, 
    store_preds : bool = True, 
    verbose : bool = True
) -> Dict[str, Union[float, np.array]]:
    """
    Evalute pipeline performance by training on the train dataset and validate the prediction on valid dataset. 
    """
    
    # TRAINING
    
    if verbose:
        logging.info(f'Start training...')

    start_train = time.perf_counter()

    pipeline.fit(train)

    end_train = time.perf_counter()
    duration_train = round(end_train - start_train, 2)

    if verbose:
        logging.info(f'✓ Training ended.')

    # INFERENCE 

    if verbose:
        logging.info(f'Start inference...')

    start_inf = time.perf_counter()

    preds = pipeline.predict(valid)

    end_inf = time.perf_counter()
    duration_inf = round(end_inf - start_inf, 2)

    if verbose:
        logging.info(f'✓ Inference ended.')

    # PERFORMANCE ON VALIDATION 

    valid = valid.with_columns(pl.Series(preds).alias('preds'))
    score = metric.compute_metric(lf = valid, target = target, preds = 'preds')

    # INFERENCE ON TRAINING 
    if eval_overfitting:

        if verbose:
            logging.info(f'Start computing overfitting...')

        train = train.with_columns(pl.Series(pipeline.predict(train)).alias('preds'))
        score_on_train = metric.compute_metric(lf = train, target = target, preds = 'preds') 
        overfitting = relative_performance(minimize, score, score_on_train)

        if verbose:
            logging.info(f'✓ Compute overfitting ended.')

    else:

        score_on_train = np.nan
        overfitting = np.nan

    if not(store_preds):
        preds = np.nan

    results = {
        'validation_score' : score, 
        'train_score' : score_on_train, 
        'overfitting' : overfitting, 
        'duration_train' : duration_train, 
        'duration_inf' : duration_inf, 
        'preds' : preds
    }

    return results 