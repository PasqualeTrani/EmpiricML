"""Tests for target engineering.

Covers the five target transformations and TransformedTargetRegressor.

The load-bearing property is the round trip: inverting a forward-transformed
target must return the original values, because that is what keeps the scores
recorded by the Lab in the original target units.
"""

import pickle
import warnings

import numpy as np
import polars as pl
import pytest

from empml.base import BaseEstimator
from empml.errors import TargetTransformError
from empml.metrics import RMSE
from empml.pipeline import Pipeline, compute_score
from empml.target import (
    CubeTarget,
    Log1pTarget,
    ReciprocalTarget,
    SqrtTarget,
    SquareTarget,
    TransformedTargetRegressor,
)
from empml.transformers import StandardScaler
from empml.wrappers import SKlearnWrapper

# All five transformations paired with the numpy reference for their forward map.
TRANSFORMS_WITH_REFERENCE = [
    (Log1pTarget(), np.log1p),
    (SquareTarget(), np.square),
    (SqrtTarget(), np.sqrt),
    (CubeTarget(), lambda y: y**3),
    (ReciprocalTarget(), lambda y: 1.0 / y),
]

ALL_TRANSFORMS = [transform for transform, _ in TRANSFORMS_WITH_REFERENCE]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class _ConstantRegressor:
    """Minimal sklearn-like regressor that returns the mean of the training target."""

    def fit(self, X, y, **kwargs):
        self.value_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class RecordingEstimator(BaseEstimator):
    """Captures the frame it was fitted on and replays a fixed prediction."""

    def __init__(self, target: str = "y", preds: np.ndarray | None = None):
        self.target = target
        self.preds = preds
        self.fitted_on = None
        self.fit_kwargs = None

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        self.fitted_on = lf.collect()
        self.fit_kwargs = fit_kwargs
        return self

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        if self.preds is not None:
            return self.preds
        # Echo the stored target so that predict() inverts exactly what fit() saw.
        return self.fitted_on[self.target].to_numpy()


@pytest.fixture
def positive_lf() -> pl.LazyFrame:
    """Strictly positive target, valid for every transformation."""
    return pl.LazyFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "y": [1.0, 2.5, 10.0, 0.5, 7.25],
        }
    )


@pytest.fixture
def signed_lf() -> pl.LazyFrame:
    """Target containing negative values."""
    return pl.LazyFrame({"f1": [1.0, 2.0, 3.0], "y": [-2.0, 1.0, 4.0]})


def transformed_target(transform, values: list[float]) -> np.ndarray:
    """Apply a transform's forward expression to a plain list of values."""
    return (
        pl.LazyFrame({"y": values})
        .select(transform.forward(pl.col("y").cast(pl.Float64)).alias("y"))
        .collect()["y"]
        .to_numpy()
    )


# ---------------------------------------------------------------------------
# TestForwardTransformations
# ---------------------------------------------------------------------------


class TestForwardTransformations:
    @pytest.mark.parametrize(
        ("transform", "reference"),
        TRANSFORMS_WITH_REFERENCE,
        ids=[repr(t) for t, _ in TRANSFORMS_WITH_REFERENCE],
    )
    def test_forward_matches_numpy(self, transform, reference):
        """Each forward expression matches its numpy equivalent."""
        values = [1.0, 2.5, 10.0, 0.5, 7.25]
        result = transformed_target(transform, values)
        assert result == pytest.approx(reference(np.array(values)))

    @pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=repr)
    def test_round_trip_returns_original_target(self, transform):
        """inverse(forward(y)) recovers y. The core correctness property."""
        values = [1.0, 2.5, 10.0, 0.5, 7.25]
        recovered = transform.inverse(transformed_target(transform, values))
        assert recovered == pytest.approx(values)

    def test_cube_round_trip_covers_negative_values(self):
        """CubeTarget is bijective over the reals, negatives included."""
        values = [-8.0, -1.5, 0.0, 1.5, 8.0]
        recovered = CubeTarget().inverse(transformed_target(CubeTarget(), values))
        assert recovered == pytest.approx(values)

    def test_log1p_inverse_is_expm1_not_expm1_features(self):
        """Log1pTarget inverts log(1+x) with exp(x)-1, not the exp(x-1) of Expm1Features."""
        assert Log1pTarget().inverse(np.array([0.0, 1.0])) == pytest.approx(
            [0.0, np.e - 1]
        )

    def test_cube_inverse_uses_cbrt_not_fractional_power(self):
        """A fractional power would return NaN for negative predictions."""
        assert CubeTarget().inverse(np.array([-8.0, 27.0])) == pytest.approx(
            [-2.0, 3.0]
        )

    @pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=repr)
    def test_repr_is_the_class_name(self, transform):
        """Transforms are stateless, so repr stays stable for HPO descriptions."""
        assert repr(transform) == f"{type(transform).__name__}()"


# ---------------------------------------------------------------------------
# TestFitTimeDomainValidation
# ---------------------------------------------------------------------------


class TestFitTimeDomainValidation:
    @pytest.mark.parametrize(
        ("transform", "values"),
        [
            (Log1pTarget(), [1.0, -1.0]),
            (Log1pTarget(), [1.0, -2.0]),
            (SquareTarget(), [1.0, -0.5]),
            (SqrtTarget(), [1.0, -0.5]),
            (ReciprocalTarget(), [1.0, 0.0]),
        ],
        ids=[
            "log1p_at_minus_one",
            "log1p_below",
            "square_neg",
            "sqrt_neg",
            "recip_zero",
        ],
    )
    def test_invalid_target_fails_at_fit(self, transform, values):
        """An out-of-domain target raises before the estimator is ever fitted."""
        lf = pl.LazyFrame({"f1": [1.0, 2.0], "y": values})
        model = TransformedTargetRegressor(RecordingEstimator(), transform)

        with pytest.raises(TargetTransformError) as excinfo:
            model.fit(lf)

        message = str(excinfo.value)
        assert type(transform).__name__ in message
        assert "'y'" in message
        assert "1 of 2 rows" in message
        assert model.estimator.fitted_on is None

    def test_error_reports_every_invalid_row(self):
        """The count covers all offending rows, not just the first."""
        lf = pl.LazyFrame({"y": [-1.0, -2.0, -3.0, 4.0]})
        model = TransformedTargetRegressor(RecordingEstimator(), SqrtTarget())

        with pytest.raises(TargetTransformError, match="3 of 4 rows"):
            model.fit(lf)

    def test_cube_accepts_negative_zero_and_positive(self, signed_lf):
        """CubeTarget has no domain restriction."""
        model = TransformedTargetRegressor(RecordingEstimator(), CubeTarget())
        model.fit(signed_lf.with_columns(pl.lit(0.0).alias("y")))
        assert model.estimator.fitted_on is not None

    @pytest.mark.parametrize(
        ("transform", "boundary"),
        [
            (SquareTarget(), 0.0),
            (SqrtTarget(), 0.0),
            (Log1pTarget(), -0.999),
            (ReciprocalTarget(), 1e-300),
        ],
        ids=["square_zero", "sqrt_zero", "log1p_above_minus_one", "recip_tiny"],
    )
    def test_boundary_values_are_accepted(self, transform, boundary):
        """The valid side of each boundary fits without complaint."""
        lf = pl.LazyFrame({"y": [boundary, 1.0]})
        TransformedTargetRegressor(RecordingEstimator(), transform).fit(lf)

    def test_missing_target_column_is_reported(self):
        """A target absent from the frame raises a targeted message."""
        model = TransformedTargetRegressor(
            RecordingEstimator(target="price"), Log1pTarget()
        )
        with pytest.raises(TargetTransformError, match="'price' is not present"):
            model.fit(pl.LazyFrame({"f1": [1.0]}))


# ---------------------------------------------------------------------------
# TestNullAndNaNHandling
# ---------------------------------------------------------------------------


class TestNullAndNaNHandling:
    def test_nulls_and_nans_are_not_domain_violations(self):
        """Only genuinely invalid values are counted, so the total stays meaningful."""
        lf = pl.LazyFrame({"y": [1.0, None, float("nan"), -5.0]})
        model = TransformedTargetRegressor(RecordingEstimator(), SqrtTarget())

        with pytest.raises(TargetTransformError, match="1 of 4 rows"):
            model.fit(lf)

    def test_null_and_nan_targets_pass_through_the_transform(self):
        """A frame of nulls and NaNs fits, leaving the estimator to object."""
        lf = pl.LazyFrame({"y": [1.0, None, float("nan")]})
        model = TransformedTargetRegressor(RecordingEstimator(), Log1pTarget())
        model.fit(lf)

        stored = model.estimator.fitted_on["y"]
        assert stored[0] == pytest.approx(np.log1p(1.0))
        assert stored[1] is None
        assert np.isnan(stored[2])

    def test_nan_predictions_are_not_flagged_as_unreachable(self):
        """A NaN prediction propagates rather than triggering the boundary warning."""
        model = TransformedTargetRegressor(
            RecordingEstimator(preds=np.array([float("nan"), 4.0])), SquareTarget()
        )
        model.fit(pl.LazyFrame({"y": [1.0, 2.0]}))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = model.predict(pl.LazyFrame({"y": [1.0, 2.0]}))

        assert np.isnan(result[0])
        assert result[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestPredictTimeProjection
# ---------------------------------------------------------------------------


class TestPredictTimeProjection:
    def test_square_clips_negative_prediction_and_warns_once(self):
        """A negative prediction is pulled to zero, with one warning naming the count."""
        model = TransformedTargetRegressor(
            RecordingEstimator(preds=np.array([-1.0, 4.0])), SquareTarget()
        )
        model.fit(pl.LazyFrame({"y": [1.0, 2.0]}))

        with pytest.warns(UserWarning, match="1 of 2 predictions") as record:
            result = model.predict(pl.LazyFrame({"y": [1.0, 2.0]}))

        assert len(record) == 1
        assert result == pytest.approx([0.0, 2.0])

    def test_sqrt_clips_rather_than_squaring_a_negative(self):
        """Squaring -5 would yield 25, the largest prediction from the lowest output."""
        model = TransformedTargetRegressor(
            RecordingEstimator(preds=np.array([-5.0, 3.0])), SqrtTarget()
        )
        model.fit(pl.LazyFrame({"y": [1.0, 4.0]}))

        with pytest.warns(UserWarning):
            result = model.predict(pl.LazyFrame({"y": [1.0, 4.0]}))

        assert result == pytest.approx([0.0, 9.0])

    def test_reciprocal_raises_on_a_zero_prediction(self):
        """Zero is the excluded boundary, so there is nowhere to project to."""
        model = TransformedTargetRegressor(
            RecordingEstimator(preds=np.array([0.0, 2.0])), ReciprocalTarget()
        )
        model.fit(pl.LazyFrame({"y": [1.0, 2.0]}))

        with (
            pytest.warns(UserWarning),
            pytest.raises(TargetTransformError, match="no nearest valid value"),
        ):
            model.predict(pl.LazyFrame({"y": [1.0, 2.0]}))

    def test_in_domain_predictions_emit_no_warning(self):
        """The common case stays quiet."""
        model = TransformedTargetRegressor(
            RecordingEstimator(preds=np.array([1.0, 4.0])), SquareTarget()
        )
        model.fit(pl.LazyFrame({"y": [1.0, 2.0]}))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert model.predict(pl.LazyFrame({"y": [1.0, 2.0]})) == pytest.approx(
                [1.0, 2.0]
            )


# ---------------------------------------------------------------------------
# TestTargetResolution
# ---------------------------------------------------------------------------


class TestTargetResolution:
    def test_target_is_inferred_from_the_wrapped_estimator(self):
        """The common case needs no explicit target."""
        inner = SKlearnWrapper(_ConstantRegressor(), features=["f1"], target="price")
        assert TransformedTargetRegressor(inner, Log1pTarget()).target == "price"

    def test_explicit_target_overrides_the_estimator(self):
        inner = SKlearnWrapper(_ConstantRegressor(), features=["f1"], target="price")
        model = TransformedTargetRegressor(inner, Log1pTarget(), target="revenue")
        assert model.target == "revenue"

    def test_wrapping_a_pipeline_requires_an_explicit_target(self):
        """A Pipeline has no target of its own, so construction fails early."""
        inner = Pipeline([("scale", StandardScaler(features=["f1"]))])
        with pytest.raises(TargetTransformError, match="Pass target="):
            TransformedTargetRegressor(inner, Log1pTarget())

    def test_wrapping_a_pipeline_with_an_explicit_target_works(self, positive_lf):
        inner = Pipeline(
            [
                ("scale", StandardScaler(features=["f1"])),
                ("model", SKlearnWrapper(_ConstantRegressor(), ["f1"], "y")),
            ]
        )
        model = TransformedTargetRegressor(inner, Log1pTarget(), target="y")
        model.fit(positive_lf)
        assert model.predict(positive_lf).shape == (5,)


# ---------------------------------------------------------------------------
# TestFitDelegation
# ---------------------------------------------------------------------------


class TestFitDelegation:
    def test_estimator_receives_the_transformed_target(self, positive_lf):
        """The wrapped estimator sees log1p(y) in place of y."""
        model = TransformedTargetRegressor(RecordingEstimator(), Log1pTarget())
        model.fit(positive_lf)

        expected = np.log1p(positive_lf.collect()["y"].to_numpy())
        assert model.estimator.fitted_on["y"].to_numpy() == pytest.approx(expected)

    def test_other_columns_and_row_order_are_untouched(self, positive_lf):
        """Only the target column changes; positional alignment must survive."""
        model = TransformedTargetRegressor(RecordingEstimator(), Log1pTarget())
        model.fit(positive_lf)

        original = positive_lf.collect()
        stored = model.estimator.fitted_on
        assert stored.columns == original.columns
        assert stored["f1"].to_list() == original["f1"].to_list()
        assert stored["f2"].to_list() == original["f2"].to_list()

    def test_fit_kwargs_are_forwarded(self, positive_lf):
        model = TransformedTargetRegressor(RecordingEstimator(), Log1pTarget())
        model.fit(positive_lf, sample_weight=[1, 1, 1, 1, 1])
        assert "sample_weight" in model.estimator.fit_kwargs

    def test_predict_inverts_what_fit_transformed(self, positive_lf):
        """An estimator echoing its training target returns the original values."""
        model = TransformedTargetRegressor(RecordingEstimator(), Log1pTarget())
        model.fit(positive_lf)

        original = positive_lf.collect()["y"].to_numpy()
        assert model.predict(positive_lf) == pytest.approx(original)

    def test_integer_target_is_cast_before_transforming(self):
        """An Int64 target must not use integer semantics for pow()."""
        lf = pl.LazyFrame({"y": [1, 2, 3]})
        model = TransformedTargetRegressor(RecordingEstimator(), SqrtTarget())
        model.fit(lf)
        assert model.estimator.fitted_on["y"].to_numpy() == pytest.approx(
            np.sqrt([1.0, 2.0, 3.0])
        )


# ---------------------------------------------------------------------------
# TestMisuse
# ---------------------------------------------------------------------------


class TestMisuse:
    def test_predict_proba_raises(self, positive_lf):
        model = TransformedTargetRegressor(RecordingEstimator(), Log1pTarget())
        with pytest.raises(TargetTransformError, match="regression only"):
            model.predict_proba(positive_lf)

    def test_repr_names_the_transform_and_target(self):
        inner = SKlearnWrapper(_ConstantRegressor(), features=["f1"], target="price")
        text = repr(TransformedTargetRegressor(inner, Log1pTarget()))
        assert "TransformedTargetRegressor(" in text
        assert "transform=Log1pTarget()" in text
        assert "target='price'" in text


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def _pipeline(self, transform):
        return Pipeline(
            [
                ("scale", StandardScaler(features=["f1"], suffix="_s")),
                (
                    "model",
                    TransformedTargetRegressor(
                        SKlearnWrapper(_ConstantRegressor(), ["f1"], "y"), transform
                    ),
                ),
            ],
            name="target_transform",
        )

    def test_validates_as_a_final_estimator_step(self, positive_lf):
        pipe = self._pipeline(Log1pTarget())
        assert pipe._is_transformer_only is False
        pipe.fit(positive_lf)
        assert pipe.predict(positive_lf).shape == (5,)

    def test_predictions_are_returned_in_original_target_units(self, positive_lf):
        """The model learns the mean of log1p(y); predictions come back in y units."""
        pipe = self._pipeline(Log1pTarget())
        pipe.fit(positive_lf)
        preds = pipe.predict(positive_lf)

        original = positive_lf.collect()["y"].to_numpy()
        assert preds == pytest.approx(np.expm1(np.mean(np.log1p(original))))

    def test_scores_are_computed_in_original_target_units(self, positive_lf):
        """The decisive end-to-end check: compute_score needs no changes.

        The same pipeline is scored against the original target and against a
        log-space target; the two differ, and only the first matches an RMSE
        computed by hand in original units.
        """
        pipe = self._pipeline(Log1pTarget())
        pipe.fit(positive_lf)
        preds = pipe.predict(positive_lf)
        original = positive_lf.collect()["y"].to_numpy()

        score = compute_score(positive_lf, preds, RMSE(), "y")
        assert score == pytest.approx(float(np.sqrt(np.mean((original - preds) ** 2))))

        # Scoring the same predictions against a log-space target would give a
        # different, meaningless number. This is what the wrapper prevents.
        log_space = positive_lf.with_columns(pl.col("y").log1p())
        assert compute_score(log_space, preds, RMSE(), "y") != pytest.approx(score)

    def test_transformed_predictions_differ_from_untransformed(self, positive_lf):
        """Proves the transformation actually changes what the model learns."""
        plain = Pipeline([("model", SKlearnWrapper(_ConstantRegressor(), ["f1"], "y"))])
        plain.fit(positive_lf)

        logged = self._pipeline(Log1pTarget())
        logged.fit(positive_lf)

        # The mean of log1p(y), inverted, is the geometric-style mean: strictly
        # below the arithmetic mean for a spread target.
        assert logged.predict(positive_lf)[0] < plain.predict(positive_lf)[0]

    def test_fitted_pipeline_survives_a_pickle_round_trip(self, positive_lf):
        """Lab checkpoints and retrieve_pipeline use plain pickle."""
        pipe = self._pipeline(Log1pTarget())
        pipe.fit(positive_lf)
        expected = pipe.predict(positive_lf)

        restored = pickle.loads(pickle.dumps(pipe))
        assert restored.predict(positive_lf) == pytest.approx(expected)
