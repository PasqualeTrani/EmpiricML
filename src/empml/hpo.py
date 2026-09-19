"""
Hyperparameter optimization helpers for the Lab.

Each parameter combination becomes one Lab experiment. After the experiments
run, the best completed one is selected from the results table.
"""

import logging

import pandas as pd
import polars as pl

from empml.base import BaseTransformer, SKlearnEstimator
from empml.pipeline import Pipeline
from empml.results import MetricColumns
from empml.wrappers import SKlearnWrapper


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


def validate_primary_metric_idx(primary_metric_idx: int | str, n_metrics: int) -> None:
    """Reject invalid multi-metric selection targets and warn about 'all'."""
    if primary_metric_idx == "all":
        logging.warning(
            "primary_metric_idx='all': HPO will only "
            "select experiments that improve on ALL "
            "metrics. This may yield no results."
        )
        return
    if not isinstance(primary_metric_idx, int):
        raise ValueError("primary_metric_idx must be int or 'all'")
    if primary_metric_idx < 0 or primary_metric_idx >= n_metrics:
        raise ValueError(
            f"primary_metric_idx={primary_metric_idx} out of range [0, {n_metrics - 1}]"
        )


def build_hpo_pipelines(
    estimator: SKlearnEstimator,
    params: list[dict],
    features: list[str],
    target: str,
    preprocessor: Pipeline | BaseTransformer,
) -> list[Pipeline]:
    """One pipeline per parameter combination, named after the estimator's repr."""
    pipelines = []
    for p in params:
        model = estimator(**p)
        pipelines.append(
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "estimator",
                        SKlearnWrapper(
                            estimator=model, features=features, target=target
                        ),
                    ),
                ],
                name=f"{repr(model)}",
                description=(
                    f"{repr(model)} with "
                    f"features={features} and "
                    f"preprocessor={repr(preprocessor)}"
                ),
            )
        )
    return pipelines


def _better_on_all_metrics(
    candidates: pl.DataFrame,
    results: pl.DataFrame,
    compare_against: int | None,
    minimize: list[bool],
    columns: MetricColumns,
) -> pl.DataFrame:
    """Keep candidates whose mean CV score beats the baseline on every metric."""
    if not compare_against:
        return candidates

    baseline = results.filter(pl.col("experiment_id") == compare_against)
    if baseline.height == 0:
        return candidates

    for i, minimize_metric in zip(columns.indices, minimize, strict=False):
        col = columns.name("cv_mean_score", i)
        baseline_val = baseline[col].item()
        if minimize_metric:
            candidates = candidates.filter(pl.col(col) < baseline_val)
        else:
            candidates = candidates.filter(pl.col(col) > baseline_val)
    return candidates


def select_hpo_best(
    results: pl.DataFrame,
    n_experiments: int,
    minimize: list[bool],
    columns: MetricColumns,
    primary_metric_idx: int | str,
    compare_against: int | None,
) -> dict | None:
    """
    Summary row of the best completed experiment among the last ``n_experiments``.

    Single-metric Labs ignore ``primary_metric_idx``. With ``'all'``, only
    candidates that beat the baseline on every metric qualify; the first metric
    then ranks them, and None is returned when no candidate qualifies.
    """
    candidates = results.tail(n_experiments).filter(pl.col("is_completed"))

    metric_idx = 0
    if columns.suffixed and primary_metric_idx == "all":
        candidates = _better_on_all_metrics(
            candidates, results, compare_against, minimize, columns
        )
        if candidates.height == 0:
            logging.warning(
                "No HPO experiment improves on all metrics. Returning None."
            )
            return None
    elif columns.suffixed:
        metric_idx = primary_metric_idx

    sort_col = columns.name("cv_mean_score", metric_idx + 1)
    return (
        candidates.sort(sort_col, descending=minimize[metric_idx])
        .tail(1)
        .row(0, named=True)
    )
