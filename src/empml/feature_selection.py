"""
Permutation-based feature importance and selection helpers for the Lab.
"""

from collections.abc import Callable

import numpy as np
import polars as pl

from empml.base import BaseEstimator, BaseTransformer, Metric
from empml.comparison import relative_performance
from empml.pipeline import Pipeline
from empml.target import TransformedTargetRegressor
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
    Relative performance drop when a final, preprocessed feature is shuffled.

    Returns one row per fold, with one column per feature and a 1-based
    ``fold_number``. ``features`` may be a subset of the model's inputs.
    """
    pfi_dfs = []
    for fold, (train_idx, valid_idx) in enumerate(cv_indexes):
        with log_step(f"Fold {fold + 1}", verbose):
            train = data.filter(pl.col(row_id).is_in(train_idx))
            valid = data.filter(pl.col(row_id).is_in(valid_idx)).collect()

            pipeline.fit(train)
            transformed, predict, model_features = _prepare_prediction(
                pipeline, valid.lazy()
            )
            # Cache the learned preprocessing, including lazy expressions, once.
            # Pipeline transformers preserve row order, so scoring stays aligned
            # with the original target even if preprocessing overwrites columns.
            transformed = transformed.collect().lazy()
            available = transformed.collect_schema().names()
            missing = [f for f in features if f not in available]
            if missing:
                raise ValueError(f"Features not found after preprocessing: {missing}")
            if model_features is not None:
                unused = [f for f in features if f not in model_features]
                if unused:
                    raise ValueError(
                        f"Features not used by the final estimator: {unused}"
                    )

            base_metric = _score_predictions(
                metric, valid, target, predict(transformed)
            )

            pfi = {
                f: _importance(
                    predict,
                    transformed,
                    valid,
                    f,
                    metric,
                    target,
                    minimize,
                    n_iters,
                    base_metric,
                )
                for f in features
            }
            pfi_dfs.append(
                pl.DataFrame(pfi).with_columns(pl.lit(fold + 1).alias("fold_number"))
            )

    return pl.concat(pfi_dfs)


def _prepare_prediction(
    estimator: Pipeline | BaseEstimator | BaseTransformer,
    lf: pl.LazyFrame,
) -> tuple[pl.LazyFrame, Callable[[pl.LazyFrame], np.ndarray], list[str] | None]:
    """Separate fitted preprocessing from prediction, keeping target inverses."""
    if isinstance(estimator, Pipeline):
        if estimator._is_transformer_only:
            raise ValueError(
                "Permutation importance requires a pipeline with an estimator."
            )
        for _name, step in estimator.steps[:-1]:
            if not isinstance(step, (BaseTransformer, Pipeline)):
                raise ValueError(
                    "Preprocessing steps must be transformers or pipelines."
                )
            lf = step.transform(lf)
        return _prepare_prediction(estimator.steps[-1][1], lf)

    if isinstance(estimator, TransformedTargetRegressor):
        lf, predict, features = _prepare_prediction(estimator.estimator, lf)

        def predict_original(lf: pl.LazyFrame) -> np.ndarray:
            return estimator._invert(np.asarray(predict(lf), dtype=np.float64))

        return lf, predict_original, features

    if isinstance(estimator, BaseTransformer):
        raise ValueError("Permutation importance requires a final estimator.")
    return lf, estimator.predict, getattr(estimator, "features", None)


def _score_predictions(
    metric: Metric, valid: pl.DataFrame, target: str, predictions: np.ndarray
) -> float:
    """Keep score columns separate from model inputs and the original target."""
    name = "_empml_preds"
    while name in valid.columns:
        name += "_"
    scored = valid.lazy().with_columns(pl.Series(name, predictions))
    return metric.compute_metric(scored, target=target, preds=name)


def _importance(
    predict: Callable[[pl.LazyFrame], np.ndarray],
    transformed: pl.LazyFrame,
    valid: pl.DataFrame,
    feature: str,
    metric: Metric,
    target: str,
    minimize: bool,
    n_iters: int,
    base_metric: float,
) -> float | None:
    """Compare base predictions with the mean score over shuffled copies of one feature."""
    shadow_metric = np.array(
        [
            _score_predictions(
                metric,
                valid,
                target,
                predict(
                    transformed.with_columns(
                        pl.col(feature).sample(fraction=1, seed=j, shuffle=True)
                    )
                ),
            )
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
