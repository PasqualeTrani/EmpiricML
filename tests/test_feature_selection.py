"""Permutation importance operates on the exact columns seen by the model."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from empml.base import (
    BaseEstimator,
    BaseTransformer,
    CVGenerator,
    DataDownloader,
    Metric,
)
from empml.errors import TargetTransformError
from empml.lab import ComparisonCriteria, Lab
from empml.metrics import MSE, Accuracy
from empml.pipeline import Pipeline
from empml.target import (
    Log1pTarget,
    ReciprocalTarget,
    SqrtTarget,
    TransformedTargetRegressor,
)
from empml.transformers import (
    Identity,
    InteractionFeatures,
    MeanTargetEncoder,
    PowerFeatures,
    StandardScaler,
)
from empml.wrappers import SKlearnWrapper


class FrameDownloader(DataDownloader):
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data.lazy()


class TwoFolds(CVGenerator):
    def split(self, lf, row_id):
        ids = lf.select(row_id).collect().to_series().to_numpy()
        return [(ids[8:], ids[:8]), (ids[:8], ids[8:])]


@pytest.fixture
def data():
    x = np.linspace(0.1, 1.6, 16)
    z = np.array([3, 1, 2, 4] * 4, dtype=float)
    return pl.DataFrame(
        {
            "id": np.arange(16)[::-1] * 3,
            "x": x,
            "z": z,
            "y": 2 * x**2 + z + 0.3,
            "cat": ["b", "a", None, "c"] * 4,
        }
    )


def make_lab(data, metric=None, minimize=True):
    return Lab(
        train_downloader=FrameDownloader(data),
        cv_generator=TwoFolds(),
        row_id="id",
        target="y",
        metric=MSE() if metric is None else metric,
        minimize=minimize if not isinstance(metric, list) else [minimize] * len(metric),
        comparison_criteria=ComparisonCriteria(n_folds_threshold=1, pct_threshold=0),
        name="importance",
    )


class RecordingEstimator(BaseEstimator):
    def __init__(self, features):
        self.features = features
        self.target = "y"
        self.fits = []
        self.predictions = []

    def fit(self, lf, **kwargs):
        self.fits.append(lf.collect())
        return self

    def predict(self, lf):
        frame = lf.collect()
        self.predictions.append(frame)
        return frame.select(pl.sum_horizontal(self.features)).to_series().to_numpy()


def expected_importance(valid, transformed, features, inverse=lambda p: p):
    """Independent numerical reference, with the same public shuffle seeds."""

    def score(frame):
        preds = frame.select(pl.sum_horizontal(features)).to_series().to_numpy()
        return np.mean((valid["y"].to_numpy() - inverse(preds)) ** 2)

    base = score(transformed)
    return {
        f: round(
            (
                np.mean(
                    [
                        score(transformed.with_columns(pl.col(f).shuffle(seed=j)))
                        for j in range(3)
                    ]
                )
                - base
            )
            / base
            * 100,
            2,
        )
        for f in features
    }


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_final_columns_are_permuted_independently(data, nested, overwrite):
    squared = "x" if overwrite else "x_squared"
    preprocessor = Pipeline(
        [
            ("power", PowerFeatures(["x"], suffix="" if overwrite else "_squared")),
            ("interaction", InteractionFeatures([(squared, "z")])),
        ]
    )
    features = [squared, f"{squared}_x_z"]
    model = RecordingEstimator(features)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    if nested:
        pipeline = Pipeline([("identity", Identity()), ("inner", pipeline)])

    result = make_lab(data).permutation_feature_importance(
        pipeline, features, n_iters=3, verbose=False
    )

    assert result.columns == [*features, "fold_number"]
    assert result["fold_number"].to_list() == [1, 2]
    assert len(model.fits) == 2
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        transformed = preprocessor.transform(valid.lazy()).collect()
        expected = expected_importance(valid, transformed, features)
        assert result.drop("fold_number").row(fold, named=True) == expected
        seen = model.predictions[fold * 7 : (fold + 1) * 7]
        assert_frame_equal(seen[0], transformed)
        for i, feature in enumerate(features):
            for j in range(3):
                assert_frame_equal(
                    seen[1 + i * 3 + j],
                    transformed.with_columns(pl.col(feature).shuffle(seed=j)),
                )


class CountingPreprocessor(BaseTransformer):
    def __init__(self):
        self.fits = 0
        self.transforms = 0
        self.executions = 0

    def fit(self, lf):
        self.fits += 1
        return self

    def transform(self, lf):
        self.transforms += 1

        def execute(frame):
            self.executions += 1
            return frame

        return lf.with_columns(x_squared=pl.col("x") ** 2).map_batches(execute)


def test_preprocessing_materialized_once_and_fit_once_per_fold(data):
    preprocessor = CountingPreprocessor()
    model = RecordingEstimator(["x", "x_squared"])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    make_lab(data).permutation_feature_importance(
        pipeline, model.features, n_iters=3, verbose=False
    )
    assert preprocessor.fits == 2
    assert preprocessor.transforms == 4  # Once for training and validation per fold.
    assert preprocessor.executions == 4
    assert len(model.fits) == 2
    assert len(model.predictions) == 14


@pytest.mark.parametrize("placement", ["model", "pipeline", "nested_targets"])
def test_target_inverse_is_preserved_through_nested_pipelines(data, placement):
    preprocessor = PowerFeatures(["x"], suffix="_squared")
    model = RecordingEstimator(["x_squared"])
    if placement == "model":
        pipeline = Pipeline(
            [
                ("power", preprocessor),
                ("model", TransformedTargetRegressor(model, Log1pTarget())),
            ]
        )
    else:
        inner = Pipeline([("power", preprocessor), ("model", model)])
        if placement == "nested_targets":
            inner = TransformedTargetRegressor(inner, SqrtTarget(), target="y")
        pipeline = Pipeline(
            [("target", TransformedTargetRegressor(inner, Log1pTarget(), target="y"))]
        )
    inverse = np.expm1 if placement != "nested_targets" else lambda p: np.expm1(p**2)
    result = make_lab(data).permutation_feature_importance(
        pipeline, ["x_squared"], n_iters=3, verbose=False
    )
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        transformed = preprocessor.transform(valid.lazy()).collect()
        assert (
            result["x_squared"][fold]
            == expected_importance(valid, transformed, ["x_squared"], inverse)[
                "x_squared"
            ]
        )
        train = data.slice((1 - fold) * 8, 8)
        expected_target = np.log1p(train["y"].to_numpy())
        if placement == "nested_targets":
            expected_target = np.sqrt(expected_target)
        np.testing.assert_allclose(model.fits[fold]["y"].to_numpy(), expected_target)


def test_learned_preprocessing_and_encoder_preserve_original_row_alignment(data):
    preprocessor = Pipeline(
        [
            ("scale", StandardScaler(["x"])),
            ("encode", MeanTargetEncoder(["cat"], encoder_col="y")),
        ]
    )
    model = RecordingEstimator(["x", "mean_cat_encoded"])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    result = make_lab(data).permutation_feature_importance(
        pipeline, model.features, n_iters=3, verbose=False
    )
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        transformed = model.predictions[fold * 7]
        assert transformed["id"].to_list() == valid["id"].to_list()
        train = data.slice((1 - fold) * 8, 8)
        np.testing.assert_allclose(
            transformed["x"].to_numpy(),
            (valid["x"].to_numpy() - train["x"].mean()) / train["x"].std(),
        )
        assert result.drop("fold_number").row(fold, named=True) == expected_importance(
            valid, transformed, model.features
        )


@pytest.mark.parametrize(
    "feature, message", [("missing", "after preprocessing"), ("z", "not used")]
)
def test_missing_or_unused_final_features_raise_clear_error(data, feature, message):
    pipeline = Pipeline([("model", RecordingEstimator(["x"]))])
    with pytest.raises(ValueError, match=message):
        make_lab(data).permutation_feature_importance(
            pipeline, [feature], verbose=False
        )


def test_feature_subset_keeps_all_model_inputs(data):
    model = RecordingEstimator(["x", "z"])
    pipeline = Pipeline([("model", model)])
    result = make_lab(data).permutation_feature_importance(
        pipeline, ["x"], n_iters=3, verbose=False
    )
    assert result.columns == ["x", "fold_number"]
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        assert (
            result["x"][fold] == expected_importance(valid, valid, model.features)["x"]
        )


class CustomEstimator(BaseEstimator):
    """A supported estimator without a public feature list."""

    def fit(self, lf):
        return self

    def predict(self, lf):
        return lf.select("x_squared").collect().to_series().to_numpy()


def test_custom_estimator_without_feature_metadata(data):
    pipeline = Pipeline(
        [
            ("power", PowerFeatures(["x"], suffix="_squared")),
            ("model", CustomEstimator()),
        ]
    )
    result = make_lab(data).permutation_feature_importance(
        pipeline, ["x_squared"], n_iters=3, verbose=False
    )
    assert result.height == 2
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        transformed = valid.with_columns(x_squared=pl.col("x") ** 2)
        assert (
            result["x_squared"][fold]
            == expected_importance(valid, transformed, ["x_squared"])["x_squared"]
        )


class UnusedMetric(Metric):
    def compute_metric(self, lf, target, preds):
        raise AssertionError("Importance must only use the first metric")


def test_scoring_columns_never_overwrite_model_inputs_or_original_target(data):
    data = data.with_columns(
        base_preds=pl.col("x"), shadow_0=pl.col("z"), _empml_preds=pl.col("z")
    )
    model = RecordingEstimator(["base_preds", "shadow_0", "_empml_preds"])
    # Even if preprocessing changes y, evaluation must retain the original target.
    pipeline = Pipeline([("target_column", PowerFeatures(["y"])), ("model", model)])
    result = make_lab(data, [MSE(), UnusedMetric()]).permutation_feature_importance(
        pipeline, model.features, n_iters=3, verbose=False
    )
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        assert result.drop("fold_number").row(fold, named=True) == expected_importance(
            valid, valid, model.features
        )
    assert all(frame.columns == data.columns for frame in model.predictions)


def test_recursive_selection_keeps_sources_for_retained_derived_features(data):
    # Symmetry makes x uninformative once x_squared is in the model, while
    # symmetric noise keeps the baseline error nonzero in both folds.
    data = data.with_columns(
        x=pl.Series([-4.0, -3, -2, -1, 1, 2, 3, 4] * 2),
        z=pl.Series([3.0, 1, 2, 4, 4, 2, 1, 3] * 2),
    )
    data = data.with_columns(y=2 * pl.col("x") ** 2 + 0.1 * pl.col("z"))
    features = ["x", "x_squared"]
    selected = make_lab(data).recursive_permutation_feature_selection(
        estimator=LinearRegression(),
        features=features,
        preprocessor=PowerFeatures(["x"], suffix="_squared"),
        n_iters=3,
        verbose=False,
    )
    assert selected == ["x_squared"]
    assert features == ["x", "x_squared"]


@pytest.mark.parametrize("features", [[], ["x", "z"]])
def test_recursive_selection_returns_empty_when_no_features_remain(data, features):
    result = make_lab(data).recursive_permutation_feature_selection(
        estimator=DummyRegressor(), features=features, n_iters=3, verbose=False
    )
    assert result == []


def test_sklearn_wrapper_accepts_generated_feature_names(data):
    pipeline = Pipeline(
        [
            ("power", PowerFeatures(["x"], suffix="_squared")),
            ("model", SKlearnWrapper(LinearRegression(), ["x_squared", "z"], "y")),
        ]
    )
    result = make_lab(data).permutation_feature_importance(
        pipeline, ["x_squared"], n_iters=3, verbose=False
    )
    assert result["x_squared"].min() > 0


def test_target_inverse_keeps_projection_warnings(data):
    data = data.with_columns(x=pl.col("x") - 1)
    model = RecordingEstimator(["x"])
    pipeline = Pipeline([("model", TransformedTargetRegressor(model, SqrtTarget()))])
    with pytest.warns(UserWarning, match="moved onto the boundary"):
        result = make_lab(data).permutation_feature_importance(
            pipeline, ["x"], n_iters=3, verbose=False
        )
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        assert (
            result["x"][fold]
            == expected_importance(
                valid, valid, ["x"], lambda p: np.maximum(p, 0) ** 2
            )["x"]
        )


def test_target_inverse_keeps_reciprocal_failure(data):
    data = data.with_columns(x=pl.lit(0.0))
    pipeline = Pipeline(
        [
            (
                "model",
                TransformedTargetRegressor(
                    RecordingEstimator(["x"]), ReciprocalTarget()
                ),
            )
        ]
    )
    with (
        pytest.warns(UserWarning, match="moved onto the boundary"),
        pytest.raises(TargetTransformError, match="exactly 0"),
    ):
        make_lab(data).permutation_feature_importance(
            pipeline, ["x"], n_iters=3, verbose=False
        )


def test_transformer_only_pipeline_is_rejected(data):
    with pytest.raises(ValueError, match="pipeline with an estimator"):
        make_lab(data).permutation_feature_importance(
            Pipeline([("identity", Identity())]), ["x"], verbose=False
        )


def test_zero_baseline_keeps_null_importance(data):
    data = data.with_columns(y=pl.col("x"))
    pipeline = Pipeline([("model", RecordingEstimator(["x"]))])
    result = make_lab(data).permutation_feature_importance(
        pipeline, ["x"], n_iters=3, verbose=False
    )
    assert result["x"].to_list() == [None, None]


def test_maximized_metric_keeps_relative_drop_formula(data):
    data = data.with_columns(x=pl.Series([0.0, 1.0] * 8)).with_columns(y=pl.col("x"))
    pipeline = Pipeline([("model", RecordingEstimator(["x"]))])
    result = make_lab(data, Accuracy(), minimize=False).permutation_feature_importance(
        pipeline, ["x"], n_iters=3, verbose=False
    )
    for fold in range(2):
        valid = data.slice(fold * 8, 8)
        shuffled_scores = [
            np.mean(valid["x"].shuffle(seed=j).to_numpy() == valid["y"].to_numpy())
            for j in range(3)
        ]
        assert result["x"][fold] == round((1 - np.mean(shuffled_scores)) * 100, 2)
