"""
Pipeline orchestration and evaluation for machine learning experiments.

This module provides the core Pipeline class for chaining transformers and estimators,
along with comprehensive evaluation functions for assessing pipeline performance in
cross-validation settings. Designed to work exclusively with Polars LazyFrames.

Key Components:
- Pipeline: Flexible ML pipeline supporting transformer chains and optional final estimators
- Single-fold evaluation: Train, predict, and score a pipeline on one fold
- Cross-validation evaluation: Evaluate pipelines across multiple CV folds with early stopping
- Performance comparison: Compare two pipeline experiments with detailed statistics

Used by the Lab class for tracking and comparing ML experiments.
"""

# base imports
from typing import Union

import numpy as np

# data wranglers
import polars as pl

# internal imports
from empml.base import BaseEstimator, BaseTransformer, Metric  # base classes
from empml.comparison import (  # noqa: F401 - re-exported for compatibility
    compare_experiments,
    compare_results_stats,
    relative_performance,
)
from empml.results import MetricColumns, format_details_rows
from empml.utils import log_execution_time, log_step, time_execution

# ------------------------------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------------------------------


class Pipeline:
    """
    Custom pipeline for chaining transformers and an optional final estimator.
    Works exclusively with Polars LazyFrames.

    Supports:
    - Transformer-only pipelines (returns transformed LazyFrame)
    - Transformer + Estimator pipelines (returns predictions)
    - Nested pipelines (a Pipeline can be a step in another Pipeline)

    Example (with estimator):
        pipeline = Pipeline([
            ('imputer', SimpleImputerTransformer(features=['col1', 'col2'])),
            ('scaler', StandardScalerTransformer(features=['col1', 'col2'])),
            ('model', lgbm_reg(features=['col1', 'col2'], target='target'))
        ])

        pipeline.fit(train_lf)
        predictions = pipeline.predict(test_lf)

    Example (transformer-only):
        preprocessing = Pipeline([
            ('imputer', SimpleImputerTransformer(features=['col1', 'col2'])),
            ('scaler', StandardScalerTransformer(features=['col1', 'col2']))
        ])

        preprocessing.fit(train_lf)
        transformed_lf = preprocessing.transform(test_lf)

    Example (nested pipelines):
        preprocessing = Pipeline([
            ('imputer', SimpleImputerTransformer(features=['col1', 'col2'])),
            ('scaler', StandardScalerTransformer(features=['col1', 'col2']))
        ])

        full_pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', lgbm_reg(features=['col1', 'col2'], target='target'))
        ])
    """

    def __init__(
        self,
        steps: list[tuple[str, Union[BaseTransformer, BaseEstimator, "Pipeline"]]],
        name: str = "",
        description: str = "",
    ):
        """
        Parameters:
        -----------
        steps : list of tuples
            List of (name, transformer/estimator/pipeline) tuples in the order they should be applied.
            If the last step is an estimator, the pipeline will support predict().
            If all steps are transformers (or pipelines acting as transformers), the pipeline
            will support transform().
        """
        self.steps = steps
        self._validate_steps()
        self._is_transformer_only = self._check_if_transformer_only()

        self.name = name
        self.description = description

    def _validate_steps(self):
        """Validate that steps are properly configured."""
        if len(self.steps) == 0:
            raise ValueError("Pipeline must have at least one step")

        # Check that all steps except possibly the last are transformers or pipelines
        for name, step in self.steps[:-1]:
            if not (isinstance(step, (BaseTransformer, Pipeline))):
                raise ValueError(
                    f"All steps except the last must be transformers or pipelines. "
                    f"'{name}' is neither."
                )

        # The last step can be a transformer, estimator, or pipeline
        last_name, last_step = self.steps[-1]
        if not isinstance(last_step, (BaseTransformer, BaseEstimator, Pipeline)):
            raise ValueError(
                f"Last step '{last_name}' must be a transformer, estimator, or pipeline."
            )

    def _check_if_transformer_only(self) -> bool:
        """Check if pipeline contains only transformers (no final estimator)."""
        last_name, last_step = self.steps[-1]

        # If last step is a Pipeline, check if it's transformer-only
        if isinstance(last_step, Pipeline):
            return last_step._is_transformer_only

        # Otherwise, check if it's a transformer
        return isinstance(last_step, BaseTransformer)

    def fit(self, lf: pl.LazyFrame, **fit_params):
        """
        Fit all transformers and the final estimator (if present).

        Parameters:
        -----------
        lf : pl.LazyFrame
            Training data
        **fit_params : dict
            Parameters to pass to the final estimator's fit method
        """
        # Apply transformers sequentially
        lf_transformed = lf
        for _name, step in self.steps[:-1]:
            lf_transformed = step.fit_transform(lf_transformed)

        # Fit the final step; transformers take no fit parameters
        _final_name, final_step = self.steps[-1]
        if isinstance(final_step, BaseTransformer):
            final_step.fit(lf_transformed)
        else:
            final_step.fit(lf_transformed, **fit_params)

        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply all transformers sequentially.
        Only available for transformer-only pipelines.

        Parameters:
        -----------
        lf : pl.LazyFrame
            Data to transform

        Returns:
        --------
        pl.LazyFrame
            Transformed data
        """
        if not self._is_transformer_only:
            raise ValueError(
                "transform() is only available for transformer-only pipelines. "
                "This pipeline has an estimator as the final step. Use predict() instead."
            )

        lf_transformed = lf
        for _name, step in self.steps:
            lf_transformed = step.transform(lf_transformed)

        return lf_transformed

    def fit_transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Fit and transform in one step.
        Only available for transformer-only pipelines.
        """
        if not self._is_transformer_only:
            raise ValueError(
                "fit_transform() is only available for transformer-only pipelines. "
                "This pipeline has an estimator as the final step. Use fit_predict() instead."
            )

        self.fit(lf)
        return self.transform(lf)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        """
        Apply all transformers and predict with the final estimator.
        Only available for pipelines with an estimator as the final step.

        Parameters:
        -----------
        lf : pl.LazyFrame
            Data to predict on

        Returns:
        --------
        np.ndarray
            Predictions
        """
        if self._is_transformer_only:
            raise ValueError(
                "predict() is only available for pipelines with an estimator. "
                "This pipeline contains only transformers. Use transform() instead."
            )

        # Apply transformers sequentially
        lf_transformed = lf
        for _name, step in self.steps[:-1]:
            lf_transformed = step.transform(lf_transformed)

        # Predict with the final estimator
        _final_name, final_estimator = self.steps[-1]
        return final_estimator.predict(lf_transformed)

    def fit_predict(self, lf: pl.LazyFrame, **fit_params) -> np.ndarray:
        """
        Fit the pipeline and return predictions on the same data.
        Only available for pipelines with an estimator.
        """
        if self._is_transformer_only:
            raise ValueError(
                "fit_predict() is only available for pipelines with an estimator. "
                "This pipeline contains only transformers. Use fit_transform() instead."
            )

        self.fit(lf, **fit_params)
        return self.predict(lf)

    def __getitem__(self, index: int | str):
        """Access a step by index or name."""
        if isinstance(index, int):
            return self.steps[index][1]

        for name, step in self.steps:
            if name == index:
                return step

        raise KeyError(f"Step '{index}' not found in pipeline")

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        steps_str = ",\n    ".join(
            [f"('{name}', {step!r})" for name, step in self.steps]
        )
        pipeline_type = (
            "transformer-only" if self._is_transformer_only else "with estimator"
        )
        return f"Pipeline({pipeline_type})[\n    {steps_str}\n]"


# ------------------------------------------------------------------------------------------
# PIPELINE EVALUATION
# ------------------------------------------------------------------------------------------
# One implementation serves single-metric and multi-metric evaluation; MetricColumns
# decides how per-metric result keys are named. The eval_pipeline_* functions keep
# their historical signatures and log names.


def train_pipeline(pipeline: Pipeline, train: pl.LazyFrame) -> Pipeline:
    """Train the pipeline on training data."""
    pipeline.fit(train)
    return pipeline


def predict_with_pipeline(pipeline: Pipeline, data: pl.LazyFrame) -> np.array:
    """Generate predictions using the pipeline."""
    return pipeline.predict(data)


def compute_scores(
    data: pl.LazyFrame,
    preds: np.ndarray,
    metrics: list[Metric],
    target: str,
) -> list[float]:
    """Compute multiple metric scores for predictions."""
    data_with_preds = data.with_columns(pl.Series(preds).alias("preds"))
    return [
        m.compute_metric(lf=data_with_preds, target=target, preds="preds")
        for m in metrics
    ]


def compute_score(
    data: pl.LazyFrame, preds: np.array, metric: Metric, target: str
) -> float:
    """Compute metric score for predictions."""
    return compute_scores(data, preds, [metric], target)[0]


def _evaluate_fold(
    pipeline: Pipeline,
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    metrics: list[Metric],
    target: str,
    minimize: list[bool],
    columns: MetricColumns,
    eval_overfitting: bool,
    store_preds: bool,
    verbose: bool,
) -> dict[str, float | list[float]]:
    """Train once, predict once, then score every metric."""
    with log_step("Training", verbose):
        _, duration_train = time_execution(train_pipeline)(pipeline, train)

    with log_step("Inference", verbose):
        preds, duration_inf = time_execution(predict_with_pipeline)(pipeline, valid)

    scores = compute_scores(valid, preds, metrics, target)

    if eval_overfitting:
        with log_step("Computing Overfitting", verbose):
            train_preds = predict_with_pipeline(pipeline, train)
            train_scores = compute_scores(train, train_preds, metrics, target)
            overfitting = [
                relative_performance(minimize_metric, score, train_score)
                for score, train_score, minimize_metric in zip(
                    scores, train_scores, minimize, strict=False
                )
            ]
    else:
        train_scores = [np.nan] * len(metrics)
        overfitting = [np.nan] * len(metrics)

    metric_results: dict[str, float | None] = {}
    for i, score, train_score, overfit in zip(
        columns.indices, scores, train_scores, overfitting, strict=False
    ):
        metric_results[columns.name("validation_score", i)] = score
        metric_results[columns.name("train_score", i)] = train_score
        metric_results[columns.name("overfitting", i)] = overfit

    shared_results = {
        "duration_train": duration_train,
        "duration_inf": duration_inf,
        "preds": list(preds) if store_preds else np.nan,
    }
    # Callers receive this dict, so each mode keeps its historical key order.
    if columns.suffixed:
        return {**shared_results, **metric_results}
    return {**metric_results, **shared_results}


@log_execution_time
def eval_pipeline_single_fold(
    pipeline: Pipeline,
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    metric: Metric,
    target: str,
    minimize: bool,
    eval_overfitting: bool = True,
    store_preds: bool = True,
    verbose: bool = True,
) -> dict[str, float | list[float]]:
    """
    Evalute pipeline performance by training on the train dataset and validate the prediction on valid dataset.
    """
    return _evaluate_fold(
        pipeline,
        train,
        valid,
        [metric],
        target,
        [minimize],
        MetricColumns.single(),
        eval_overfitting,
        store_preds,
        verbose,
    )


@log_execution_time
def eval_pipeline_single_fold_multi(
    pipeline: Pipeline,
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    metrics: list[Metric],
    target: str,
    minimize: list[bool],
    eval_overfitting: bool = True,
    store_preds: bool = True,
    verbose: bool = True,
) -> dict[str, float | list[float]]:
    """
    Evaluate pipeline on a single fold against multiple metrics.

    Trains once, predicts once, then scores each metric.
    Returns dict with suffixed keys (validation_score_1, etc.).
    """
    return _evaluate_fold(
        pipeline,
        train,
        valid,
        metrics,
        target,
        minimize,
        MetricColumns.multi(len(metrics)),
        eval_overfitting,
        store_preds,
        verbose,
    )


def evaluate_fold(
    pipeline: Pipeline,
    train: pl.LazyFrame,
    valid: pl.LazyFrame,
    metrics: list[Metric],
    target: str,
    minimize: list[bool],
    columns: MetricColumns,
    eval_overfitting: bool = True,
    store_preds: bool = True,
    verbose: bool = True,
) -> dict[str, float | list[float]]:
    """
    Evaluate a pipeline on one train/validation split.

    Delegates to the public evaluator for the naming mode, so the logged
    function name stays the same as before.
    """
    options = {
        "eval_overfitting": eval_overfitting,
        "store_preds": store_preds,
        "verbose": verbose,
    }
    if columns.suffixed:
        return eval_pipeline_single_fold_multi(
            pipeline, train, valid, metrics, target, minimize, **options
        )
    return eval_pipeline_single_fold(
        pipeline, train, valid, metrics[0], target, minimize[0], **options
    )


def _has_too_many_worse_folds(
    fold_results: list[dict],
    compare_df: pl.DataFrame,
    minimize: list[bool],
    columns: MetricColumns,
    max_worse_folds: int | None,
) -> bool:
    """True when the partial results are worse than the baseline on too many folds for any metric."""
    partial_details = format_details_rows(
        pl.DataFrame(fold_results), experiment_id=None, columns=columns
    )
    comparisons = compare_experiments(compare_df, partial_details, minimize, columns)
    return any(c["n_folds_lower_performance"] > max_worse_folds for c in comparisons)


def evaluate_cv(
    pipeline: Pipeline,
    lz: pl.LazyFrame,
    cv_indexes: list[tuple[np.ndarray, np.ndarray]],
    row_id: str,
    metrics: list[Metric],
    target: str,
    minimize: list[bool],
    columns: MetricColumns,
    eval_overfitting: bool = True,
    store_preds: bool = True,
    verbose: bool = True,
    compare_df: pl.DataFrame | None = None,
    max_worse_folds: int | None = None,
) -> pl.DataFrame:
    """
    Evaluate a pipeline fold by fold.

    When ``compare_df`` holds a baseline's per-fold details, evaluation stops
    early once any metric is worse than the baseline on more than
    ``max_worse_folds`` folds.
    """
    fold_results = []
    for fold, (train_idx, valid_idx) in enumerate(cv_indexes, 1):
        with log_step(f"Fold {fold}", verbose):
            train = lz.filter(pl.col(row_id).is_in(train_idx))
            valid = lz.filter(pl.col(row_id).is_in(valid_idx))
            fold_results.append(
                evaluate_fold(
                    pipeline,
                    train,
                    valid,
                    metrics,
                    target,
                    minimize,
                    columns,
                    eval_overfitting,
                    store_preds,
                    verbose,
                )
            )

            if (
                compare_df is not None
                and compare_df.shape[0] > 0
                and _has_too_many_worse_folds(
                    fold_results, compare_df, minimize, columns, max_worse_folds
                )
            ):
                break

    return pl.DataFrame(fold_results)


def eval_pipeline_cv(
    pipeline: Pipeline,
    lz: pl.LazyFrame,
    cv_indexes: list[tuple[np.ndarray, np.ndarray]],
    row_id: str,
    metric: Metric,
    target: str,
    minimize: bool,
    eval_overfitting: bool = True,
    store_preds: bool = True,
    verbose: bool = True,
    compare_df: pl.DataFrame = pl.DataFrame(),
    th_lower_performance_n_folds: int | None = None,
) -> pl.DataFrame:
    """
    Evalute pipeline performance in a cross-validation fashion, by using cv_indexes.
    """
    return evaluate_cv(
        pipeline,
        lz,
        cv_indexes,
        row_id,
        [metric],
        target,
        [minimize],
        MetricColumns.single(),
        eval_overfitting,
        store_preds,
        verbose,
        compare_df,
        th_lower_performance_n_folds,
    )


def eval_pipeline_cv_multi(
    pipeline: Pipeline,
    lz: pl.LazyFrame,
    cv_indexes: list[tuple[np.ndarray, np.ndarray]],
    row_id: str,
    metrics: list[Metric],
    target: str,
    minimize: list[bool],
    eval_overfitting: bool = True,
    store_preds: bool = True,
    verbose: bool = True,
    compare_df: pl.DataFrame = pl.DataFrame(),
    th_lower_performance_n_folds: int | None = None,
) -> pl.DataFrame:
    """
    Evaluate pipeline in CV fashion for multiple metrics.

    Early stopping arrests if ANY metric has too many
    underperforming folds.
    """
    return evaluate_cv(
        pipeline,
        lz,
        cv_indexes,
        row_id,
        metrics,
        target,
        minimize,
        MetricColumns.multi(len(metrics)),
        eval_overfitting,
        store_preds,
        verbose,
        compare_df,
        th_lower_performance_n_folds,
    )


def compare_results_stats_multi(
    results_a: pl.DataFrame,
    results_b: pl.DataFrame,
    minimize: list[bool],
    n_metrics: int,
) -> list[dict[str, float | pl.DataFrame]]:
    """
    Compare two experiments across multiple metrics.

    Returns a list of comparison dicts (one per metric),
    each with the same structure as compare_results_stats.
    """
    return compare_experiments(
        results_a, results_b, minimize, MetricColumns.multi(n_metrics)
    )
