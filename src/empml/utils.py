import time
from functools import wraps
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import (
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


def train_pipeline(pipeline: Pipeline, train: pl.LazyFrame) -> Pipeline:
    """Train the pipeline on training data."""
    pipeline.fit(train)
    return pipeline


def predict_with_pipeline(pipeline: Pipeline, data: pl.LazyFrame) -> np.array:
    """Generate predictions using the pipeline."""
    return pipeline.predict(data)


def compute_score(
    data: pl.LazyFrame, 
    preds: np.array, 
    metric: Metric, 
    target: str
) -> float:
    """Compute metric score for predictions."""
    data_with_preds = data.with_columns(pl.Series(preds).alias('preds'))
    return metric.compute_metric(lf=data_with_preds, target=target, preds='preds')


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
    
    with log_step('Training', verbose):
        _, duration_train = time_execution(train_pipeline)(pipeline, train)
    
    with log_step('Inference', verbose):
        preds, duration_inf = time_execution(predict_with_pipeline)(pipeline, valid)
    
    score = compute_score(valid, preds, metric, target)
    
    if eval_overfitting:
        with log_step('Computing Overfitting', verbose):
            train_preds = predict_with_pipeline(pipeline, train)
            score_on_train = compute_score(train, train_preds, metric, target)
            overfitting = relative_performance(minimize, score, score_on_train)
    else:
        score_on_train = np.nan
        overfitting = np.nan
    
    return {
        'validation_score': score, 
        'train_score': score_on_train, 
        'overfitting': overfitting, 
        'duration_train': duration_train, 
        'duration_inf': duration_inf, 
        'preds': preds if store_preds else np.nan
    }