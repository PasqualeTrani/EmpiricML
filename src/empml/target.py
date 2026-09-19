"""
Target engineering: fit a model on a transformed target and return predictions
in the original units.

A target transformation cannot be a :class:`~empml.base.BaseTransformer` pipeline
step, because pipeline steps also run at prediction time, when the target column
is absent, and transformers have no inverse. It belongs inside the estimator
instead: :class:`TransformedTargetRegressor` transforms the target when fitting
and inverts the model's output when predicting, so the scores recorded by the Lab
stay in the original target units and remain comparable with untransformed runs.
"""

# standard import libraries
import warnings
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

# wranglers
import polars as pl

# internal imports
from empml.base import BaseEstimator
from empml.errors import TargetTransformError

# ------------------------------------------------------------------------------------------
# Target Transformations
# ------------------------------------------------------------------------------------------


class TargetTransform(ABC):
    """
    A reversible transformation of the target column.

    Implementations are stateless and parameterless, so a single instance is safe
    to share across folds and to pickle inside a fitted pipeline.

    The four methods split by when they run: `forward` and `out_of_domain` build
    Polars expressions applied to the target column at fit time, while `inverse`,
    `unreachable` and `project` operate on the model's numpy output at prediction
    time.
    """

    #: Requirement on the target, quoted verbatim in the fit-time error message.
    domain: ClassVar[str] = "any real value"

    @abstractmethod
    def forward(self, column: pl.Expr) -> pl.Expr:
        """Map the target into the space the model is fitted in."""
        pass

    @abstractmethod
    def inverse(self, preds: np.ndarray) -> np.ndarray:
        """Map the model's output back into the original target units."""
        pass

    def out_of_domain(self, column: pl.Expr) -> pl.Expr:
        """Boolean expression, true for target values this transform cannot accept."""
        return pl.lit(False)

    def unreachable(self, preds: np.ndarray) -> np.ndarray:
        """Boolean mask, true for predictions outside the image of `forward`."""
        return np.zeros(preds.shape, dtype=bool)

    def project(self, preds: np.ndarray) -> np.ndarray:
        """Move unreachable predictions onto the nearest value `inverse` accepts."""
        return preds

    def __repr__(self) -> str:
        """Transforms carry no state, so the class name fully describes them."""
        return f"{type(self).__name__}()"


class Log1pTarget(TargetTransform):
    """Fit on log(1 + y) and predict back with exp(p) - 1."""

    domain = "y > -1"

    def forward(self, column: pl.Expr) -> pl.Expr:
        return column.log1p()

    def inverse(self, preds: np.ndarray) -> np.ndarray:
        # expm1, not exp(p) - 1, for precision near zero. Note that
        # transformers.Expm1Features computes exp(x - 1) and is NOT this inverse.
        return np.expm1(preds)

    def out_of_domain(self, column: pl.Expr) -> pl.Expr:
        # log1p(-1) is -inf and log1p(y < -1) is NaN; neither can be fitted.
        return column <= -1


class SquareTarget(TargetTransform):
    """Fit on y ** 2 and predict back with sqrt(p)."""

    # Squaring is not injective over the reals and sqrt never returns a negative,
    # so a negative target would come back positive with its sign silently lost.
    domain = "y >= 0, because the inverse (sqrt) never returns a negative value"

    def forward(self, column: pl.Expr) -> pl.Expr:
        return column.pow(2)

    def inverse(self, preds: np.ndarray) -> np.ndarray:
        return np.sqrt(preds)

    def out_of_domain(self, column: pl.Expr) -> pl.Expr:
        return column < 0

    def unreachable(self, preds: np.ndarray) -> np.ndarray:
        return preds < 0

    def project(self, preds: np.ndarray) -> np.ndarray:
        return np.maximum(preds, 0.0)


class SqrtTarget(TargetTransform):
    """Fit on sqrt(y) and predict back with p ** 2."""

    domain = "y >= 0"

    def forward(self, column: pl.Expr) -> pl.Expr:
        return column.sqrt()

    def inverse(self, preds: np.ndarray) -> np.ndarray:
        return np.square(preds)

    def out_of_domain(self, column: pl.Expr) -> pl.Expr:
        return column < 0

    def unreachable(self, preds: np.ndarray) -> np.ndarray:
        # Squaring a negative prediction is numerically valid but meaningless: it
        # would turn the most negative model output into the largest prediction.
        # The image of sqrt is [0, inf), so anything below zero is unreachable.
        return preds < 0

    def project(self, preds: np.ndarray) -> np.ndarray:
        return np.maximum(preds, 0.0)


class CubeTarget(TargetTransform):
    """Fit on y ** 3 and predict back with cbrt(p). Bijective over the reals."""

    def forward(self, column: pl.Expr) -> pl.Expr:
        return column.pow(3)

    def inverse(self, preds: np.ndarray) -> np.ndarray:
        # np.cbrt, not preds ** (1 / 3), which is NaN for every negative input.
        return np.cbrt(preds)


class ReciprocalTarget(TargetTransform):
    """Fit on 1 / y and predict back with 1 / p."""

    domain = "y != 0"

    def forward(self, column: pl.Expr) -> pl.Expr:
        return pl.lit(1.0) / column

    def inverse(self, preds: np.ndarray) -> np.ndarray:
        return 1.0 / preds

    def out_of_domain(self, column: pl.Expr) -> pl.Expr:
        return column == 0

    def unreachable(self, preds: np.ndarray) -> np.ndarray:
        return preds == 0

    def project(self, preds: np.ndarray) -> np.ndarray:
        """
        Unlike the other transforms, there is nowhere to project to: zero is
        itself the excluded boundary and both one-sided limits of 1 / p are
        infinite. Substituting the smallest representable float would produce
        predictions around 1e308, which corrupt a metric far more thoroughly
        than an explicit failure does.
        """
        raise TargetTransformError(
            f"{self!r} cannot invert a prediction of exactly 0: the inverse 1 / p "
            "is unbounded there, so there is no nearest valid value to fall back "
            "on. The model is a poor fit for a reciprocal target; consider "
            "Log1pTarget or fitting on the untransformed target."
        )


# ------------------------------------------------------------------------------------------
# Transformed Target Estimator
# ------------------------------------------------------------------------------------------


def _infer_target(estimator: BaseEstimator) -> str:
    """Read the target column name off the wrapped estimator."""
    target = getattr(estimator, "target", None)
    if isinstance(target, str) and target:
        return target
    raise TargetTransformError(
        f"{type(estimator).__name__} does not expose a 'target' attribute, so the "
        "target column cannot be inferred. Pass target='<column name>' explicitly."
    )


class TransformedTargetRegressor(BaseEstimator):
    """
    Fits the wrapped estimator on a transformed target and inverts its predictions.

    The target column is transformed in place before the wrapped estimator is
    fitted, and every prediction is mapped back into the original units. Metrics
    are therefore computed on the original scale, so experiments using a target
    transformation stay directly comparable with experiments that do not.

    Target transformations apply to regression only.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to fit on the transformed target. Usually a `SKlearnWrapper`
        or `TorchWrapper`, but a whole `Pipeline` also works.
    transform : TargetTransform
        The transformation to apply, for example `Log1pTarget()`.
    target : str, optional
        The target column name. Inferred from `estimator.target` when omitted,
        which is the usual case; required when wrapping a `Pipeline`, because a
        pipeline has no target of its own.

    Examples
    --------
    >>> from lightgbm import LGBMRegressor
    >>> from empml.wrappers import SKlearnWrapper
    >>>
    >>> model = TransformedTargetRegressor(
    ...     estimator=SKlearnWrapper(
    ...         estimator=LGBMRegressor(),
    ...         features=['sqft_living', 'bedrooms'],
    ...         target='price',
    ...     ),
    ...     transform=Log1pTarget(),
    ... )
    >>> model.fit(train_lf)          # LightGBM sees log1p(price)
    >>> predictions = model.predict(valid_lf)   # returned in price units
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        transform: TargetTransform,
        target: str | None = None,
    ):
        self.estimator = estimator
        self.transform = transform
        # Resolved eagerly so that a missing target fails when the pipeline is
        # built, rather than several folds into a cross-validation run.
        self.target = target if target is not None else _infer_target(estimator)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        """Fit the wrapped estimator on the transformed target."""
        column = pl.col(self.target).cast(pl.Float64)
        self._reject_out_of_domain(lf, column)

        # with_columns is elementwise, so row order is preserved and predictions
        # still line up positionally with the untransformed rows when scored.
        transformed = lf.with_columns(self.transform.forward(column).alias(self.target))
        self.estimator.fit(transformed, **fit_kwargs)
        return self

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        """Predict with the wrapped estimator and return the original target units."""
        preds = np.asarray(self.estimator.predict(lf), dtype=np.float64)
        return self._invert(preds)

    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        """Target transformations are regression-only, so this always raises."""
        raise TargetTransformError(
            "Target transformations apply to regression only, so "
            "TransformedTargetRegressor does not provide predict_proba. Use the "
            "wrapped estimator directly for classification."
        )

    def _reject_out_of_domain(self, lf: pl.LazyFrame, column: pl.Expr) -> None:
        """Fail before fitting if any target value cannot be transformed."""
        if self.target not in lf.collect_schema().names():
            raise TargetTransformError(
                f"Target column {self.target!r} is not present in the data passed "
                f"to {type(self).__name__}.fit()."
            )

        # Nulls and NaNs are left to the wrapped estimator to report, so the count
        # below stays a count of genuinely invalid values.
        counts = (
            lf.select(
                invalid=self.transform.out_of_domain(column).fill_null(False).sum(),
                total=pl.len(),
            )
            .collect()
            .row(0, named=True)
        )
        if counts["invalid"]:
            raise TargetTransformError(
                f"{self.transform!r} cannot be applied to target column "
                f"{self.target!r}: {counts['invalid']} of {counts['total']} rows "
                f"are outside its domain (requires {self.transform.domain}). "
                "Filter or shift the target before fitting, or choose another "
                "transformation."
            )

    def _invert(self, preds: np.ndarray) -> np.ndarray:
        """Map predictions back, projecting any the transform cannot invert."""
        unreachable = self.transform.unreachable(preds)
        n_unreachable = int(unreachable.sum())
        if n_unreachable:
            # A model artifact rather than a user error, so the run continues with
            # the minimum-distance correction and one warning carrying the count.
            warnings.warn(
                f"{n_unreachable} of {preds.size} predictions fall outside the "
                f"range {self.transform!r} can invert and were moved onto the "
                "boundary. The transformation may be a poor fit for this target.",
                stacklevel=3,
            )
            preds = self.transform.project(preds)
        return self.transform.inverse(preds)

    def __repr__(self) -> str:
        """Return a representation showing the estimator, transform and target."""
        return (
            f"TransformedTargetRegressor({self.estimator!r}, "
            f"transform={self.transform!r}, target={self.target!r})"
        )
