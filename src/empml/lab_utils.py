import polars as pl 
import numpy as np
import pandas as pd 
from datetime import datetime 
import pytz
from typing import Dict, List

# ANSI escape codes for colors in print and logging 
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ------------------------------------------------------------------------------------------
# Setting up functions
# ------------------------------------------------------------------------------------------

def setup_row_id_column(df: pl.DataFrame, row_id: str | None = None) -> tuple[pl.DataFrame, str]:
    """
    Setup row identifier column.
    
    Returns:
        Tuple of (modified_dataframe, row_id_column_name)
    """
    if row_id:
        return df, row_id
    else:
        df_with_id = df.with_row_index().rename({'index': 'row_id'})
        return df_with_id, 'row_id'


def create_results_schema() -> pl.DataFrame:
    """Initialize empty Polars DataFrame for tracking experiments."""
    return pl.DataFrame(
        schema={
            'experiment_id': pl.Int64,
            'name': pl.Utf8,
            'description': pl.Utf8,
            'cv_mean_score': pl.Float64,
            'train_mean_score': pl.Float64,
            'mean_overfitting_pct': pl.Float64,
            'cv_std_score': pl.Float64,
            'mean_train_time_s': pl.Float64,
            'mean_inference_time_s': pl.Float64,
            'is_completed': pl.Boolean,
            'timestamp_utc': pl.Datetime,
        }
    )


def create_results_details_schema() -> pl.DataFrame:
    """Initialize empty Polars DataFrame for tracking experiment details."""
    return pl.DataFrame(
        schema={
            'experiment_id': pl.Int64,
            'fold_number': pl.Int64,
            'validation_score': pl.Float64,
            'train_score': pl.Float64,
            'overfitting_pct': pl.Float64,
        }
    )


# ------------------------------------------------------------------------------------------
# Formatting results functions 
# ------------------------------------------------------------------------------------------

def format_experiment_results(
    eval: pl.DataFrame,
    experiment_id: int,
    is_completed : bool,
    description: str = '',
    name: str = ''
) -> pl.DataFrame:
    """
    Transform evaluation results into experiment summary format.
    
    Args:
        eval: DataFrame with fold evaluation results
        experiment_id: Unique identifier for the experiment
        description: Human-readable description of the experiment
        notes: Additional notes about the experiment
        
    Returns:
        DataFrame with aggregated experiment metrics
    """
    return (
        eval.drop('preds').mean()
        .rename({
            'validation_score': 'cv_mean_score',
            'train_score': 'train_mean_score',
            'overfitting': 'mean_overfitting_pct',
            'duration_train': 'mean_train_time_s',
            'duration_inf': 'mean_inference_time_s',
        })
        .with_columns(pl.lit(eval['validation_score'].std()).alias('cv_std_score'))
        .with_columns(
            pl.lit(experiment_id).alias('experiment_id'),
            pl.lit(description).alias('description'),
            pl.lit(name).alias('name'),
            pl.lit(is_completed).alias('is_completed'),
            pl.lit(datetime.now(pytz.timezone('UTC'))).dt.replace_time_zone(None).alias('timestamp_utc')
        )
    )


def format_experiment_details(eval: pl.DataFrame, experiment_id: int) -> pl.DataFrame:
    """
    Transform evaluation results into detailed fold-by-fold format.
    
    Args:
        eval: DataFrame with fold evaluation results
        experiment_id: Unique identifier for the experiment
        
    Returns:
        DataFrame with per-fold metrics
    """
    return (
        eval
        .drop(['preds', 'duration_train', 'duration_inf'])
        .with_row_index()
        .rename({'index': 'fold_number'})
        .with_columns(
            pl.col('fold_number') + 1,
            pl.lit(experiment_id).alias('experiment_id')
        )
        .rename({'overfitting': 'overfitting_pct'})
    )


def prepare_predictions_for_save(eval: pl.DataFrame) -> pl.DataFrame:
    """
    Transform evaluation results into predictions format for storage.
    
    Args:
        eval: DataFrame with fold evaluation results including predictions
        
    Returns:
        DataFrame with predictions exploded by fold
    """
    return (
        eval
        .select('preds')
        .drop_nans()
        .drop_nulls()
        .explode('preds')
    )


def format_log_performance(x : float, th : float, is_percentage : bool = True) -> str:
    """Format logging performance to highlight good and bad performance"""

    percentage_str : str = '%' if is_percentage else ''

    if x>th: # good performance in green
        return f"{BOLD}{GREEN}{str(x)}{percentage_str}{RESET}"
    else: # bad ones in red
        return f"{BOLD}{RED}{str(x)}{percentage_str}{RESET}"
    

def log_performance_against(comparison : Dict[str, float], n_folds_threshold : int):
    """Print performance stats for comparing two experiments."""

    print(f"\n{BOLD}{BLUE}Relative Performance Report Experiment B (Current) vs Experiment A (Chosen Baseline){RESET}")
    print(f"""
    {BOLD}{BLUE}Note: positive performances like reduction of overfitting or increment/decrement of the metrics over the CV are indicated in {RESET}{BOLD}{GREEN}GREEN{RESET}, 
    {BOLD}{BLUE}while negative performances are indicated in {BOLD}{RED}RED{RESET}\n
    """)

    print(f'Mean CV Score Experiment B vs A: {format_log_performance(comparison['mean_cv_performance'], 0)}')
    print(f'Std CV Score Experiment B vs A: {format_log_performance(comparison['std_cv_performance'], 0)}\n')

    for row in comparison['fold_performances'].iter_rows():
        print(f'Fold {row[0]} Score Performance Experiment B vs A: {format_log_performance(row[1], 0)}')
    print(f'Number of Folds Experiment B is Better then A: {format_log_performance(comparison['n_folds_better_performance'], comparison['n_folds'] - n_folds_threshold - 1, is_percentage = False)}\n')

    print(f'Mean % Overfitting Score Experiment B vs A: {format_log_performance(comparison['mean_cv_performance_overfitting'], 0)}')
    for row in comparison['fold_performances_overfitting'].iter_rows():
        print(f'\t - Fold {row[0]} % Overfitting Score Performance Experiment B vs A: {format_log_performance(row[1], 0)}')



def retrieve_predictions_from_path(lab_name : str, experiment_id : int) -> pl.Expr:

    expr = pl.read_parquet(f'./{lab_name}/predictions/predictions_{experiment_id}.parquet').to_series().alias(f'preds_{experiment_id}')
    
    if expr.shape[0] > 0:
        return expr 
    else:
        return pl.lit(np.nan).alias(f'preds_{experiment_id}')
    

def generate_params_list(
    params_list : Dict[str, List[float | int | str]], 
    search_type : str = 'grid', 
    num_samples : int = 64, 
    random_state : int = 0
) -> List[Dict[str, float]]:
    
    """Generate a list of paramaters to test in an HPO procedure starting from a dictionary with list of parameters."""

    params_df = pd.Series(params_list).reset_index().transpose()
    params_df.columns = params_df.iloc[0]
    params_df=params_df.iloc[1:]

    for c in params_df.columns:
        params_df = params_df.explode(c, ignore_index=True)

    if search_type == 'grid':
        sample = params_df
    elif search_type == 'random':
        sample = params_df.sample(n=num_samples, random_state=random_state)
    else:
        raise ValueError("search_type argument should be 'grid' or 'random'")

    return [dict(row) for i, row in sample.iterrows()]