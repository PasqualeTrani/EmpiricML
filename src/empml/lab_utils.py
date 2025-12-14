import polars as pl 
from datetime import datetime 
import pytz
from typing import Dict

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
            'description': pl.Utf8,
            'cv_mean_score': pl.Float64,
            'train_mean_score': pl.Float64,
            'mean_overfitting_pct': pl.Float64,
            'cv_std_score': pl.Float64,
            'mean_train_time_s': pl.Float64,
            'mean_inference_time_s': pl.Float64,
            'is_completed': pl.Boolean,
            'notes': pl.Utf8,
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
    notes: str = ''
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
            pl.lit(notes).alias('notes'),
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
        .with_row_index()
        .rename({'index': 'fold_number'})
        .with_columns(pl.col('fold_number') + 1)
        .explode('preds')
    )


def format_log_performance(x : float, th : float) -> str:
    """Format logging performance to highlight good and bad performance"""

    if x>th: # good performance in green
        return f"{BOLD}{GREEN}{str(x)}%{RESET}"
    else: # bad ones in red
        return f"{BOLD}{RED}{str(x)}%{RESET}"
    

def log_performance_against(comparison : Dict[str, float], threshold : float):
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
    print(f'Percentage of Folds Experiment B is Better then A: {format_log_performance(comparison['perc_of_folds_b_better_then_a'], threshold * 100)}\n')

    print(f'Mean % Overfitting Score Experiment B vs A: {format_log_performance(comparison['mean_cv_performance_overfitting'], 0)}')
    for row in comparison['fold_performances_overfitting'].iter_rows():
        print(f'\t - Fold {row[0]} % Overfitting Score Performance Experiment B vs A: {format_log_performance(row[1], 0)}')