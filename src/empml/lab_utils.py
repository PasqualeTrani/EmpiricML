"""
Utility functions for the Lab experiment tracking system.

This module keeps the row-identifier and prediction-storage helpers. Earlier
releases also defined results, comparison, and HPO helpers here; they now live
in ``empml.results``, ``empml.comparison``, ``empml.hpo``, and
``empml.artifacts``. The names below stay importable for backward compatibility.
"""

import numpy as np
import polars as pl

from empml.artifacts import LabArtifacts
from empml.comparison import (  # noqa: F401 - re-exported for compatibility
    compute_anomaly,
    format_log_performance,
    generate_shuffle_preds,
    log_performance_against,
    log_performance_against_multi,
)
from empml.hpo import generate_params_list  # noqa: F401 - re-exported for compatibility
from empml.results import (
    MetricColumns,
    format_details_rows,
    format_results_row,
    results_details_schema,
    results_schema,
)

# ------------------------------------------------------------------------------------------
# Setting up functions
# ------------------------------------------------------------------------------------------


def setup_row_id_column(
    df: pl.LazyFrame, row_id: str | None = None
) -> tuple[pl.LazyFrame, str]:
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
        schema = df.collect_schema()
        if row_id not in schema.names():
            raise ValueError(f"Row ID column {row_id!r} was not found.")
        missing_id = pl.col(row_id).is_null()
        if schema[row_id] in (pl.Float32, pl.Float64):
            missing_id = missing_id | pl.col(row_id).is_nan()
        id_stats = df.select(
            pl.len().alias("rows"),
            missing_id.sum().alias("nulls"),
            pl.col(row_id).n_unique().alias("unique"),
        ).collect()
        if id_stats["nulls"].item() > 0:
            raise ValueError(f"Row ID column {row_id!r} must not contain null values.")
        if id_stats["unique"].item() != id_stats["rows"].item():
            raise ValueError(f"Row ID column {row_id!r} must contain unique values.")
        return df, row_id
    else:
        # Create 'row_id' column with sequential indices
        df_with_id = df.with_row_index().rename({"index": "row_id"})
        return df_with_id, "row_id"


# ------------------------------------------------------------------------------------------
# Prediction storage
# ------------------------------------------------------------------------------------------


def prepare_predictions_for_save(
    eval: pl.DataFrame,
    validation_keys: list[pl.DataFrame],
    row_id: str,
) -> pl.DataFrame:
    """
    Extract predictions from evaluation results for Lab's prediction storage.

    Lab stores predictions separately in parquet files. This function flattens
    the nested predictions structure to prepare for saving.

    Args:
        eval: DataFrame with nested predictions per fold
        validation_keys: Row IDs and fold numbers in prediction order
        row_id: Name of the row identifier column

    Returns:
        DataFrame keyed by row ID and fold number, one row per evaluated sample
    """
    prediction_frames = []
    for fold_index, predictions in enumerate(eval["preds"].to_list()):
        if not isinstance(predictions, (list, np.ndarray)):
            continue
        keys = validation_keys[fold_index]
        if keys.height != len(predictions):
            raise ValueError(
                f"Fold {fold_index + 1} has {keys.height} row IDs but "
                f"{len(predictions)} predictions."
            )
        prediction_frames.append(keys.with_columns(pl.Series("preds", predictions)))

    if prediction_frames:
        return pl.concat(prediction_frames, how="vertical_relaxed")

    return (
        validation_keys[0]
        .head(0)
        .with_columns(pl.Series("preds", [], dtype=pl.Float64))
        .select(row_id, "fold_number", "preds")
    )


def retrieve_predictions_from_path(lab_name: str, experiment_id: int) -> pl.DataFrame:
    """
    Load predictions from Lab's prediction storage for a specific experiment.

    Returns:
        Stored prediction DataFrame. Legacy artifacts contain only ``preds``.
    """
    return LabArtifacts(lab_name).load_predictions(experiment_id)


# ------------------------------------------------------------------------------------------
# Compatibility wrappers over empml.results
# ------------------------------------------------------------------------------------------


def create_results_schema() -> pl.DataFrame:
    """Empty single-metric experiment summary table."""
    return results_schema(MetricColumns.single())


def create_results_details_schema() -> pl.DataFrame:
    """Empty single-metric per-fold table."""
    return results_details_schema(MetricColumns.single())


def create_results_schema_multi(
    n_metrics: int,
) -> pl.DataFrame:
    """Empty multi-metric experiment summary table (columns suffixed _1, _2, ...)."""
    return results_schema(MetricColumns.multi(n_metrics))


def create_results_details_schema_multi(
    n_metrics: int,
) -> pl.DataFrame:
    """Empty multi-metric per-fold table (columns suffixed _1, _2, ...)."""
    return results_details_schema(MetricColumns.multi(n_metrics))


def format_experiment_results(
    eval: pl.DataFrame,
    experiment_id: int,
    is_completed: bool,
    description: str = "",
    name: str = "",
) -> pl.DataFrame:
    """Aggregate single-metric fold results into one summary row."""
    return format_results_row(
        eval, experiment_id, is_completed, MetricColumns.single(), description, name
    )


def format_experiment_details(eval: pl.DataFrame, experiment_id: int) -> pl.DataFrame:
    """Format single-metric fold results as per-fold detail rows."""
    return format_details_rows(eval, experiment_id, MetricColumns.single())


def format_experiment_results_multi(
    eval: pl.DataFrame,
    experiment_id: int,
    is_completed: bool,
    n_metrics: int,
    description: str = "",
    name: str = "",
) -> pl.DataFrame:
    """Aggregate multi-metric fold results into one summary row."""
    return format_results_row(
        eval,
        experiment_id,
        is_completed,
        MetricColumns.multi(n_metrics),
        description,
        name,
    )


def format_experiment_details_multi(
    eval: pl.DataFrame,
    experiment_id: int,
    n_metrics: int,
) -> pl.DataFrame:
    """Format multi-metric fold results as per-fold detail rows."""
    return format_details_rows(eval, experiment_id, MetricColumns.multi(n_metrics))
