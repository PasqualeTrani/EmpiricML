"""
Utility functions for the Lab experiment tracking system.

This module provides helper functions used by the Lab class to manage machine learning
experiments. Each ML pipeline tested is tracked as an experiment with comprehensive
metrics, predictions, and metadata stored in a structured format.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl

from empml.base import Metric

# streaming engine as the default for .collect()
pl.Config.set_engine_affinity(engine="streaming")

# ANSI escape codes for colored terminal output
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ------------------------------------------------------------------------------------------
# Setting up functions
# ------------------------------------------------------------------------------------------


def setup_row_id_column(
    df: pl.DataFrame, row_id: str | None = None
) -> tuple[pl.DataFrame, str]:
    """
    Ensure DataFrame has a row identifier for tracking predictions across folds.

    The Lab class requires a unique row identifier to properly align predictions
    from different CV folds back to the original dataset.

    Args:
        df: Input DataFrame
        row_id: Existing row ID column name, or None to auto-generate

    Returns:
        Tuple of (DataFrame with row ID, row ID column name)
    """
    if row_id:
        return df, row_id
    else:
        # Create 'row_id' column with sequential indices
        df_with_id = df.with_row_index().rename({"index": "row_id"})
        return df_with_id, "row_id"


def create_results_schema() -> pl.DataFrame:
    """
    Create empty schema for Lab's experiment results table.

    The Lab class maintains a results table where each row represents one experiment
    with aggregated performance metrics across all CV folds.

    Returns:
        Empty DataFrame with experiment summary columns
    """
    return pl.DataFrame(
        schema={
            "experiment_id": pl.Int64,
            "name": pl.Utf8,
            "description": pl.Utf8,
            "cv_mean_score": pl.Float64,  # Mean validation score across folds
            "train_mean_score": pl.Float64,  # Mean training score across folds
            "mean_overfitting_pct": pl.Float64,  # Average overfitting percentage
            "cv_std_score": pl.Float64,  # Standard deviation of validation scores
            "mean_train_time_s": pl.Float64,  # Average training time per fold
            "mean_inference_time_s": pl.Float64,  # Average inference time per fold
            "is_completed": pl.Boolean,  # Whether experiment finished successfully
            "timestamp_utc": pl.Datetime,  # When experiment was logged
        }
    )


def create_results_details_schema() -> pl.DataFrame:
    """
    Create empty schema for Lab's per-fold experiment details table.

    The Lab class maintains a detailed table where each row represents one fold
    of one experiment, allowing granular analysis of fold-level performance.

    Returns:
        Empty DataFrame with per-fold metric columns
    """
    return pl.DataFrame(
        schema={
            "experiment_id": pl.Int64,
            "fold_number": pl.Int64,
            "validation_score": pl.Float64,
            "train_score": pl.Float64,
            "overfitting_pct": pl.Float64,
        }
    )


def create_results_schema_multi(
    n_metrics: int,
) -> pl.DataFrame:
    """
    Create empty schema for multi-metric experiment results.

    Per-metric columns get _1, _2, ... suffixes. Shared columns
    (timing, metadata) remain unsuffixed.

    Args:
        n_metrics: Number of metrics

    Returns:
        Empty DataFrame with multi-metric experiment columns
    """
    schema: dict[str, pl.DataType] = {
        "experiment_id": pl.Int64,
        "name": pl.Utf8,
        "description": pl.Utf8,
    }
    for i in range(1, n_metrics + 1):
        schema[f"cv_mean_score_{i}"] = pl.Float64
        schema[f"train_mean_score_{i}"] = pl.Float64
        schema[f"mean_overfitting_pct_{i}"] = pl.Float64
        schema[f"cv_std_score_{i}"] = pl.Float64
    schema["mean_train_time_s"] = pl.Float64
    schema["mean_inference_time_s"] = pl.Float64
    schema["is_completed"] = pl.Boolean
    schema["timestamp_utc"] = pl.Datetime
    return pl.DataFrame(schema=schema)


def create_results_details_schema_multi(
    n_metrics: int,
) -> pl.DataFrame:
    """
    Create empty schema for multi-metric per-fold details.

    Per-metric columns get _1, _2, ... suffixes.

    Args:
        n_metrics: Number of metrics

    Returns:
        Empty DataFrame with multi-metric per-fold columns
    """
    schema: dict[str, pl.DataType] = {
        "experiment_id": pl.Int64,
        "fold_number": pl.Int64,
    }
    for i in range(1, n_metrics + 1):
        schema[f"validation_score_{i}"] = pl.Float64
        schema[f"train_score_{i}"] = pl.Float64
        schema[f"overfitting_pct_{i}"] = pl.Float64
    return pl.DataFrame(schema=schema)


# ------------------------------------------------------------------------------------------
# Formatting results functions
# ------------------------------------------------------------------------------------------


def format_experiment_results(
    eval: pl.DataFrame,
    experiment_id: int,
    is_completed: bool,
    description: str = "",
    name: str = "",
) -> pl.DataFrame:
    """
    Transform raw fold evaluation results into Lab's experiment summary format.

    Takes the per-fold evaluation metrics and aggregates them into a single row
    suitable for insertion into the Lab's results table.

    Args:
        eval: DataFrame with per-fold metrics (from CV evaluation)
        experiment_id: Unique identifier assigned by Lab
        is_completed: Whether experiment finished without errors
        description: Human-readable description of the experiment
        name: Short name for the experiment

    Returns:
        Single-row DataFrame matching create_results_schema()
    """
    return (
        eval.drop("preds")
        .mean()
        .rename(
            {
                "validation_score": "cv_mean_score",
                "train_score": "train_mean_score",
                "overfitting": "mean_overfitting_pct",
                "duration_train": "mean_train_time_s",
                "duration_inf": "mean_inference_time_s",
            }
        )
        .with_columns(pl.lit(eval["validation_score"].std()).alias("cv_std_score"))
        .with_columns(
            pl.lit(experiment_id).alias("experiment_id"),
            pl.lit(description).alias("description"),
            pl.lit(name).alias("name"),
            pl.lit(is_completed).alias("is_completed"),
            pl.lit(datetime.now(UTC)).dt.replace_time_zone(None).alias("timestamp_utc"),
        )
    )


def format_experiment_details(eval: pl.DataFrame, experiment_id: int) -> pl.DataFrame:
    """
    Transform raw fold evaluation results into Lab's detailed per-fold format.

    Expands the per-fold metrics into a format suitable for insertion into the
    Lab's results_details table for granular fold-level analysis.

    Args:
        eval: DataFrame with per-fold metrics (from CV evaluation)
        experiment_id: Unique identifier assigned by Lab

    Returns:
        DataFrame with one row per fold matching create_results_details_schema()
    """
    return (
        eval.drop(["preds", "duration_train", "duration_inf"])
        .with_row_index()
        .rename({"index": "fold_number"})
        .with_columns(
            pl.col("fold_number") + 1,  # 1-indexed fold numbers for readability
            pl.lit(experiment_id).alias("experiment_id"),
        )
        .rename({"overfitting": "overfitting_pct"})
    )


def format_experiment_results_multi(
    eval: pl.DataFrame,
    experiment_id: int,
    is_completed: bool,
    n_metrics: int,
    description: str = "",
    name: str = "",
) -> pl.DataFrame:
    """
    Aggregate multi-metric fold results into a summary row.

    For each metric i, computes mean/std of validation_score_i,
    train_score_i, overfitting_i into cv_mean_score_i, etc.

    Args:
        eval: DataFrame with per-fold multi-metric results
        experiment_id: Unique identifier assigned by Lab
        is_completed: Whether experiment completed all folds
        n_metrics: Number of metrics
        description: Human-readable experiment description
        name: Short experiment name

    Returns:
        Single-row DataFrame matching create_results_schema_multi
    """
    # Build rename map and std columns
    rename_map = {
        "duration_train": "mean_train_time_s",
        "duration_inf": "mean_inference_time_s",
    }
    for i in range(1, n_metrics + 1):
        rename_map[f"validation_score_{i}"] = f"cv_mean_score_{i}"
        rename_map[f"train_score_{i}"] = f"train_mean_score_{i}"
        rename_map[f"overfitting_{i}"] = f"mean_overfitting_pct_{i}"

    result = eval.drop("preds").mean().rename(rename_map)

    # Add std columns per metric
    std_cols = [
        pl.lit(eval[f"validation_score_{i}"].std()).alias(f"cv_std_score_{i}")
        for i in range(1, n_metrics + 1)
    ]
    result = result.with_columns(std_cols)

    # Add metadata
    result = result.with_columns(
        pl.lit(experiment_id).alias("experiment_id"),
        pl.lit(description).alias("description"),
        pl.lit(name).alias("name"),
        pl.lit(is_completed).alias("is_completed"),
        pl.lit(datetime.now(UTC)).dt.replace_time_zone(None).alias("timestamp_utc"),
    )
    return result


def format_experiment_details_multi(
    eval: pl.DataFrame,
    experiment_id: int,
    n_metrics: int,
) -> pl.DataFrame:
    """
    Format multi-metric per-fold details.

    Renames overfitting_i -> overfitting_pct_i, adds fold_number
    and experiment_id columns.

    Args:
        eval: DataFrame with per-fold multi-metric results
        experiment_id: Unique identifier assigned by Lab
        n_metrics: Number of metrics

    Returns:
        DataFrame with one row per fold
    """
    rename_map = {
        f"overfitting_{i}": f"overfitting_pct_{i}" for i in range(1, n_metrics + 1)
    }
    return (
        eval.drop(["preds", "duration_train", "duration_inf"])
        .with_row_index()
        .rename({"index": "fold_number"})
        .with_columns(
            pl.col("fold_number") + 1,
            pl.lit(experiment_id).alias("experiment_id"),
        )
        .rename(rename_map)
    )


def prepare_predictions_for_save(eval: pl.DataFrame) -> pl.DataFrame:
    """
    Extract predictions from evaluation results for Lab's prediction storage.

    Lab stores predictions separately in parquet files. This function flattens
    the nested predictions structure to prepare for saving.

    Args:
        eval: DataFrame with nested predictions per fold

    Returns:
        DataFrame with predictions unnested, one row per sample
    """
    return (
        eval.select("preds")
        .drop_nans()
        .drop_nulls()
        .explode("preds")  # Flatten nested predictions from all folds
    )


def format_log_performance(x: float, th: float, is_percentage: bool = True) -> str:
    """
    Format performance metric with color coding for Lab's console output.

    Used by Lab when comparing experiments to highlight improvements (green)
    and regressions (red) in terminal output.

    Args:
        x: Performance value
        th: Threshold for determining good/bad performance
        is_percentage: Whether to append '%' symbol

    Returns:
        ANSI-colored string (green if x > th, red otherwise)
    """
    percentage_str: str = "%" if is_percentage else ""

    if x > th:  # Improvement: green
        return f"{BOLD}{GREEN}{str(x)}{percentage_str}{RESET}"
    else:  # Regression: red
        return f"{BOLD}{RED}{str(x)}{percentage_str}{RESET}"


def log_performance_against(comparison: dict[str, float], n_folds_threshold: int):
    """
    Print comprehensive comparison report between two Lab experiments.

    Lab uses this to display detailed performance comparisons when evaluating
    whether a new experiment (B) improves upon a baseline experiment (A).

    Args:
        comparison: Dict with comparison metrics:
            - mean_cv_performance: Overall CV score difference
            - std_cv_performance: CV stability difference
            - fold_performances: Per-fold score differences
            - n_folds_better_performance: Count of folds where B beats A
            - mean_cv_performance_overfitting: Overall overfitting difference
            - fold_performances_overfitting: Per-fold overfitting differences
        n_folds_threshold: Minimum fold advantage needed to consider B better
    """
    print(
        f"\n{BOLD}{BLUE}Relative Performance Report Experiment B (Current) vs Experiment A (Chosen Baseline){RESET}"
    )
    print(f"""
    {BOLD}{BLUE}Note: positive performances like reduction of overfitting or increment/decrement of the metrics over the CV are indicated in {RESET}{BOLD}{GREEN}GREEN{RESET},
    {BOLD}{BLUE}while negative performances are indicated in {BOLD}{RED}RED{RESET}\n
    """)

    # Overall CV performance comparison
    print(
        f"Mean CV Score Experiment B vs A: {format_log_performance(comparison['mean_cv_performance'], 0)}"
    )
    print(
        f"Std CV Score Experiment B vs A: {format_log_performance(comparison['std_cv_performance'], 0)}\n"
    )

    # Per-fold breakdown
    for row in comparison["fold_performances"].iter_rows():
        print(
            f"Fold {row[0]} Score Performance Experiment B vs A: {format_log_performance(row[1], 0)}"
        )
    print(
        f"Number of Folds Experiment B is Better then A: {format_log_performance(comparison['n_folds_better_performance'], comparison['n_folds'] - n_folds_threshold - 1, is_percentage=False)}\n"
    )

    # Overfitting analysis
    print(
        f"Mean % Overfitting Score Experiment B vs A: {format_log_performance(comparison['mean_cv_performance_overfitting'], 0)}"
    )
    for row in comparison["fold_performances_overfitting"].iter_rows():
        print(
            f"\t - Fold {row[0]} % Overfitting Score Performance Experiment B vs A: {format_log_performance(row[1], 0)}"
        )


def log_performance_against_multi(
    comparisons: list[dict],
    n_folds_threshold: int,
    n_metrics: int,
) -> None:
    """
    Print comparison report for each metric independently.

    Iterates over comparison dicts and delegates per-metric
    logging to log_performance_against.

    Args:
        comparisons: List of comparison dicts, one per metric
        n_folds_threshold: Fold advantage threshold
        n_metrics: Number of metrics
    """
    for i, comparison in enumerate(comparisons, 1):
        print(f"\n{BOLD}{BLUE}{'=' * 60}\n  Metric {i}\n{'=' * 60}{RESET}")
        log_performance_against(
            comparison=comparison,
            n_folds_threshold=n_folds_threshold,
        )


def retrieve_predictions_from_path(lab_name: str, experiment_id: int) -> pl.Expr:
    """
    Load predictions from Lab's prediction storage for a specific experiment.

    Lab stores predictions in parquet files under ./{lab_name}/predictions/.
    Used when comparing or ensembling predictions from different experiments.

    Args:
        lab_name: Name of the Lab instance (directory name)
        experiment_id: Unique identifier for the experiment

    Returns:
        Polars expression with predictions, or NaN if file is empty/missing
    """
    expr = (
        pl.read_parquet(f"./{lab_name}/predictions/predictions_{experiment_id}.parquet")
        .to_series()
        .alias(f"preds_{experiment_id}")
    )

    if expr.shape[0] > 0:
        return expr
    else:
        return pl.lit(np.nan).alias(f"preds_{experiment_id}")


# ------------------------------------------------------------------------------------------
# HPO
# ------------------------------------------------------------------------------------------


def generate_params_list(
    params_list: dict[str, list[float | int | str]],
    search_type: str = "grid",
    num_samples: int = 64,
    random_state: int = 0,
) -> list[dict[str, float]]:
    """
    Generate hyperparameter configurations for Lab's HPO functionality.

    Lab can run hyperparameter optimization by testing multiple parameter
    configurations as separate experiments. This generates the search space.

    Args:
        params_list: Dict mapping parameter names to lists of candidate values
        search_type: 'grid' for exhaustive search, 'random' for sampling
        num_samples: Number of random samples (only used if search_type='random')
        random_state: Random seed for reproducible sampling

    Returns:
        List of parameter dictionaries, each representing one experiment to run

    Raises:
        ValueError: If search_type is not 'grid' or 'random'
    """
    # Convert to DataFrame for cartesian product computation
    params_df = pd.Series(params_list).reset_index().transpose()
    params_df.columns = params_df.iloc[0]
    params_df = params_df.iloc[1:]

    # Generate all combinations (grid) or subset (random)
    for c in params_df.columns:
        params_df = params_df.explode(c, ignore_index=True)

    if search_type == "grid":
        sample = params_df
    elif search_type == "random":
        sample = params_df.sample(n=num_samples, random_state=random_state)
    else:
        raise ValueError("search_type argument should be 'grid' or 'random'")

    # Convert to list of dicts for Lab to iterate over
    return [dict(row) for i, row in sample.iterrows()]


# ------------------------------------------------------------------------------------------
# Statistical tests
# ------------------------------------------------------------------------------------------


def generate_shuffle_preds(
    lf: pl.LazyFrame, preds_1: str, preds_2: str, random_state: int = 0
) -> pl.LazyFrame:
    """
    Create shuffled predictions for permutation testing in Lab.

    Lab can use permutation tests to assess whether performance differences
    between two experiments are statistically significant. This randomly swaps
    predictions between the two experiments to create null distribution.

    Args:
        lf: LazyFrame containing both sets of predictions
        preds_1: First experiment's prediction column
        preds_2: Second experiment's prediction column
        random_state: Random seed for reproducible permutations

    Returns:
        LazyFrame with 'shuffle_a' and 'shuffle_b' columns (randomly swapped)
    """
    transf_lz = (
        lf
        # Generate random binary mask (0 or 1) for each row
        .with_columns(
            rand_seq=(
                pl.int_range(0, pl.len()).sample(
                    fraction=1.0, with_replacement=True, seed=random_state
                )
                % 2
            )
        ).with_columns(
            # shuffle_a: takes preds_1 where mask=1, preds_2 where mask=0
            (
                (pl.col(preds_1) * pl.col("rand_seq"))
                + (pl.col(preds_2) * (1 - pl.col("rand_seq")))
            ).alias("shuffle_a"),
            # shuffle_b: inverse of shuffle_a
            (
                (pl.col(preds_2) * pl.col("rand_seq"))
                + (pl.col(preds_1) * (1 - pl.col("rand_seq")))
            ).alias("shuffle_b"),
        )
    )

    return transf_lz


def compute_anomaly(
    metric: Metric, lf: pl.LazyFrame, preds_1: str, preds_2: str, target: str
) -> float:
    """
    Compute test statistic for Lab's permutation testing.

    Calculates the absolute difference in metric scores between two experiments.
    Lab uses this as the test statistic when running permutation tests to assess
    significance of performance differences.

    Args:
        metric: Metric object with compute_metric method
        lf: LazyFrame containing predictions and target
        preds_1: First experiment's prediction column
        preds_2: Second experiment's prediction column
        target: Ground truth label column

    Returns:
        Absolute difference in metric scores (test statistic)
    """
    score_1 = metric.compute_metric(lf=lf, target=target, preds=preds_1)
    score_2 = metric.compute_metric(lf=lf, target=target, preds=preds_2)

    return abs(score_1 - score_2)
