"""
Permutation-based feature importance and selection helpers for the Lab.
"""

import numpy as np
import polars as pl

from empml.base import Metric
from empml.comparison import relative_performance
from empml.pipeline import Pipeline
from empml.utils import log_step


def permutation_feature_importance(
    pipeline: Pipeline,
    data: pl.LazyFrame,
    cv_indexes: list[tuple[np.ndarray, np.ndarray]],
    row_id: str,
    features: list[str],
    metric: Metric,
    target: str,
    minimize: bool,
    n_iters: int,
    verbose: bool,
) -> pl.DataFrame:
    """
    Relative performance drop per feature when that feature is shuffled.

    Returns one row per fold, with one column per feature and a 1-based
    ``fold_number``.
    """
    pfi_dfs = []
    for fold, (train_idx, valid_idx) in enumerate(cv_indexes):
        with log_step(f"Fold {fold + 1}", verbose):
            train = data.filter(pl.col(row_id).is_in(train_idx))
            valid = data.filter(pl.col(row_id).is_in(valid_idx))

            pipeline.fit(train)
            valid = valid.with_columns(
                pl.Series(pipeline.predict(valid)).alias("base_preds")
            )

            pfi = {
                f: _importance(pipeline, valid, f, metric, target, minimize, n_iters)
                for f in features
            }
            pfi_dfs.append(
                pl.DataFrame(pfi).with_columns(pl.lit(fold + 1).alias("fold_number"))
            )

    return pl.concat(pfi_dfs)


def _importance(
    pipeline: Pipeline,
    valid: pl.LazyFrame,
    feature: str,
    metric: Metric,
    target: str,
    minimize: bool,
    n_iters: int,
) -> float | None:
    """Compare base predictions with the mean score over shuffled copies of one feature."""
    shadow = valid.with_columns(
        pl.Series(
            pipeline.predict(
                valid.with_columns(
                    pl.col(feature).sample(fraction=1, seed=j, shuffle=True)
                )
            )
        ).alias(f"shadow_{j}")
        for j in range(n_iters)
    )

    base_metric = metric.compute_metric(shadow, target=target, preds="base_preds")
    shadow_metric = np.array(
        [
            metric.compute_metric(shadow, target=target, preds=f"shadow_{j}")
            for j in range(n_iters)
        ]
    ).mean()

    # A worse score without the feature means the feature matters.
    return relative_performance(minimize=not minimize, x1=base_metric, x2=shadow_metric)


def features_without_importance(pfi: pl.DataFrame) -> list[str]:
    """Features whose mean importance across folds is zero or negative."""
    mean_pfi = (
        pfi.drop("fold_number")
        .mean()
        .transpose(include_header=True, header_name="features", column_names=["pfi"])
    )
    return mean_pfi.filter(pl.col("pfi") <= 0)["features"].to_list()
