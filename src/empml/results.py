"""
Experiment results tables kept by the Lab.

A Lab keeps two tables: ``results`` has one summary row per experiment, and
``results_details`` has one row per fold. Per-metric columns are named by
``MetricColumns``, so single-metric and multi-metric Labs share one
implementation.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl


@dataclass(frozen=True)
class MetricColumns:
    """
    Name per-metric columns.

    A single-metric Lab keeps the historical unsuffixed names (``cv_mean_score``).
    A multi-metric Lab appends the 1-based metric position (``cv_mean_score_2``).
    """

    n_metrics: int
    suffixed: bool

    @classmethod
    def single(cls) -> "MetricColumns":
        return cls(n_metrics=1, suffixed=False)

    @classmethod
    def multi(cls, n_metrics: int) -> "MetricColumns":
        return cls(n_metrics=n_metrics, suffixed=True)

    @property
    def indices(self) -> range:
        """1-based metric positions."""
        return range(1, self.n_metrics + 1)

    def name(self, base: str, index: int) -> str:
        """Column name of ``base`` for the metric at 1-based ``index``."""
        return f"{base}_{index}" if self.suffixed else base


def results_schema(columns: MetricColumns) -> pl.DataFrame:
    """Empty experiment summary table: one row per experiment."""
    schema: dict[str, Any] = {
        "experiment_id": pl.Int64,
        "name": pl.Utf8,
        "description": pl.Utf8,
    }
    for i in columns.indices:
        schema[columns.name("cv_mean_score", i)] = pl.Float64
        schema[columns.name("train_mean_score", i)] = pl.Float64
        schema[columns.name("mean_overfitting_pct", i)] = pl.Float64
        schema[columns.name("cv_std_score", i)] = pl.Float64
    schema["mean_train_time_s"] = pl.Float64
    schema["mean_inference_time_s"] = pl.Float64
    schema["is_completed"] = pl.Boolean
    schema["timestamp_utc"] = pl.Datetime
    return pl.DataFrame(schema=schema)


def results_details_schema(columns: MetricColumns) -> pl.DataFrame:
    """Empty per-fold table: one row per fold of each experiment."""
    schema: dict[str, Any] = {
        "experiment_id": pl.Int64,
        "fold_number": pl.Int64,
    }
    for i in columns.indices:
        schema[columns.name("validation_score", i)] = pl.Float64
        schema[columns.name("train_score", i)] = pl.Float64
        schema[columns.name("overfitting_pct", i)] = pl.Float64
    return pl.DataFrame(schema=schema)


def format_results_row(
    eval: pl.DataFrame,
    experiment_id: int,
    is_completed: bool,
    columns: MetricColumns,
    description: str = "",
    name: str = "",
) -> pl.DataFrame:
    """
    Aggregate per-fold evaluation results into one summary row.

    Args:
        eval: Per-fold results from CV evaluation
        experiment_id: Identifier assigned by the Lab
        is_completed: Whether every fold was evaluated
        columns: Per-metric column naming
        description: Human-readable experiment description
        name: Short experiment name
    """
    renames = {
        "duration_train": "mean_train_time_s",
        "duration_inf": "mean_inference_time_s",
    }
    for i in columns.indices:
        renames[columns.name("validation_score", i)] = columns.name("cv_mean_score", i)
        renames[columns.name("train_score", i)] = columns.name("train_mean_score", i)
        renames[columns.name("overfitting", i)] = columns.name(
            "mean_overfitting_pct", i
        )

    std_scores = [
        pl.lit(eval[columns.name("validation_score", i)].std()).alias(
            columns.name("cv_std_score", i)
        )
        for i in columns.indices
    ]
    return (
        eval.drop("preds")
        .mean()
        .rename(renames)
        .with_columns(std_scores)
        .with_columns(
            pl.lit(experiment_id).alias("experiment_id"),
            pl.lit(description).alias("description"),
            pl.lit(name).alias("name"),
            pl.lit(is_completed).alias("is_completed"),
            pl.lit(datetime.now(UTC)).dt.replace_time_zone(None).alias("timestamp_utc"),
        )
    )


def format_details_rows(
    eval: pl.DataFrame, experiment_id: int | None, columns: MetricColumns
) -> pl.DataFrame:
    """
    Turn per-fold evaluation results into per-fold detail rows.

    Fold numbers are 1-based. ``experiment_id`` is None for the partial results
    that CV evaluation compares while deciding whether to stop early.
    """
    renames = {
        columns.name("overfitting", i): columns.name("overfitting_pct", i)
        for i in columns.indices
    }
    return (
        eval.drop(["preds", "duration_train", "duration_inf"])
        .with_row_index("fold_number")
        .with_columns(
            pl.col("fold_number") + 1,
            pl.lit(experiment_id).alias("experiment_id"),
        )
        .rename(renames)
    )
