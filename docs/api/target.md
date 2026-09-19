# empml.target

Target engineering: fit a model on a transformed target and get predictions back in the original units.

| Object | Description |
| :--- | :--- |
| `TransformedTargetRegressor` | Fits the wrapped estimator on a transformed target and inverts its predictions. |
| `TargetTransform` | Abstract base class for reversible target transformations. |
| `Log1pTarget` | Fit on `log(1 + y)`, predict back with `exp(p) - 1`. |
| `SquareTarget` | Fit on `y ** 2`, predict back with `sqrt(p)`. |
| `SqrtTarget` | Fit on `sqrt(y)`, predict back with `p ** 2`. |
| `CubeTarget` | Fit on `y ** 3`, predict back with `cbrt(p)`. |
| `ReciprocalTarget` | Fit on `1 / y`, predict back with `1 / p`. |

## Why this is an estimator and not a transformer

A target transformation cannot be a pipeline step. Pipeline steps also run at prediction time, when the
target column is not in the data, and `BaseTransformer` has no inverse. The transformation therefore lives
inside the estimator: the target is transformed when fitting, and every prediction is mapped back before it
leaves `predict()`.

The practical consequence is that **scores stay in the original target units**. An experiment using
`Log1pTarget` and an experiment without it can be compared directly in `lab.results`, with no mental
conversion and no changes to metrics or the Lab.

## Quick start

Wrap the estimator you would normally pass as the pipeline's final step:

```python
from lightgbm import LGBMRegressor
from empml.pipeline import Pipeline
from empml.wrappers import SKlearnWrapper
from empml.target import TransformedTargetRegressor, Log1pTarget

features = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']

pipe = Pipeline(
    steps=[
        ('model', TransformedTargetRegressor(
            estimator=SKlearnWrapper(
                estimator=LGBMRegressor(verbose=-1),
                features=features,
                target='price',
            ),
            transform=Log1pTarget(),
        )),
    ],
    name='LGBM_log_target',
    description='LightGBM fitted on log1p(price).',
)

lab.run_experiment(pipeline=pipe)
```

The target column name is not repeated: it is read from the `SKlearnWrapper` you passed in. Setting it in
two places would allow the two to drift apart, and the resulting failure would be silent — the model would
train on the raw target while predictions were inverted anyway.

## Where to place the wrapper

Wrapping only the model and wrapping a whole nested `Pipeline` are **not** the same experiment when the
pipeline contains a target encoder.

```python
# The target encoder sees the ORIGINAL price; only the model is fitted in log space.
Pipeline([
    ('encode', MeanTargetEncoder(features=['zipcode'], encoder_col='price')),
    ('model', TransformedTargetRegressor(
        SKlearnWrapper(LGBMRegressor(), features, 'price'), Log1pTarget())),
])

# The target encoder sees log1p(price); the WHOLE pipeline is fitted in log space.
TransformedTargetRegressor(
    estimator=Pipeline([
        ('encode', MeanTargetEncoder(features=['zipcode'], encoder_col='price')),
        ('model', SKlearnWrapper(LGBMRegressor(), features, 'price')),
    ]),
    transform=Log1pTarget(),
    target='price',      # required: a Pipeline has no target of its own
)
```

The second form is usually what "fit everything on log(y)" means.

## Domains

Each transformation rejects targets it cannot invert, and clips predictions that fall outside the range it
can map back.

| Transformation | Forward | Inverse | Rejected at fit | Clipped at predict |
| :--- | :--- | :--- | :--- | :--- |
| `Log1pTarget` | `log(1 + y)` | `exp(p) - 1` | `y <= -1` | none |
| `SquareTarget` | `y ** 2` | `sqrt(p)` | `y < 0` | `p < 0` becomes `0` |
| `SqrtTarget` | `sqrt(y)` | `p ** 2` | `y < 0` | `p < 0` becomes `0` |
| `CubeTarget` | `y ** 3` | `cbrt(p)` | none | none |
| `ReciprocalTarget` | `1 / y` | `1 / p` | `y == 0` | `p == 0` raises |

**At fit time**, an out-of-domain target raises `TargetTransformError` naming the transformation, the target
column and the number of offending rows. This is a data or configuration problem you can fix before spending
any compute, so it fails immediately rather than recording a score built on an uninvertible transformation.

`SquareTarget` and `SqrtTarget` reject negative targets rather than quietly redefining themselves as
sign-preserving. Squaring is not injective over the reals and `sqrt` never returns a negative value, so a
target of `-3` would be fitted as `9` and come back as `+3`, with the sign silently destroyed.

**At predict time**, a model can emit a value the inverse cannot accept — a small negative number when the
inverse is `sqrt`, for example. This is a property of the fitted model, not a user error, and it appears
mid-run, so the prediction is moved onto the nearest valid value and a single warning reports how many rows
were affected. A large count is a signal that the transformation suits this target poorly.

`ReciprocalTarget` is the one exception: zero *is* its excluded boundary and both one-sided limits of `1 / p`
are infinite, so there is no nearest valid value and it raises instead. Substituting the smallest
representable float would produce predictions around `1e308`, corrupting the metric far more thoroughly.

Null and NaN targets are passed through untouched and never counted as domain violations, so the reported
count stays a count of genuinely invalid values.

## TransformedTargetRegressor

Fits the wrapped estimator on a transformed target and inverts its predictions. Regression only.

### Methods

```python
def __init__(
    self,
    estimator: BaseEstimator,
    transform: TargetTransform,
    target: str | None = None,
):
    """
    Parameters:
    -----------
    estimator : BaseEstimator
        The estimator to fit on the transformed target. Usually a SKlearnWrapper
        or TorchWrapper, but a whole Pipeline also works.
    transform : TargetTransform
        The transformation to apply, for example Log1pTarget().
    target : str, optional
        The target column name. Inferred from estimator.target when omitted;
        required when wrapping a Pipeline, which has no target of its own.
    """

def fit(self, lf: pl.LazyFrame, **fit_kwargs):
    """Fit the wrapped estimator on the transformed target.

    Raises TargetTransformError if the target column is missing or holds values
    outside the transformation's domain.
    """

def predict(self, lf: pl.LazyFrame) -> np.ndarray:
    """Predict with the wrapped estimator and return the original target units."""

def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
    """Always raises: target transformations apply to regression only."""
```

## TargetTransform

Abstract base class for reversible target transformations. Implementations are stateless and parameterless,
so one instance is safe to share across folds and to pickle inside a fitted pipeline.

Subclass it to add your own transformation; only `forward` and `inverse` are required.

### Methods

```python
def forward(self, column: pl.Expr) -> pl.Expr:
    """Map the target into the space the model is fitted in. Required."""

def inverse(self, preds: np.ndarray) -> np.ndarray:
    """Map the model's output back into the original target units. Required."""

def out_of_domain(self, column: pl.Expr) -> pl.Expr:
    """Boolean expression, true for target values this transform cannot accept.
    Defaults to accepting everything."""

def unreachable(self, preds: np.ndarray) -> np.ndarray:
    """Boolean mask, true for predictions outside the image of forward().
    Defaults to no mask."""

def project(self, preds: np.ndarray) -> np.ndarray:
    """Move unreachable predictions onto the nearest value inverse() accepts.
    Defaults to leaving them unchanged."""
```

`forward` receives an expression rather than a column name, so the cast to `Float64` is applied once by the
wrapper and implementations stay pure arithmetic.

## Log1pTarget

Fit on `log(1 + y)`, predict back with `exp(p) - 1`. The usual first choice for a right-skewed positive
target such as a price, a count or a demand figure.

Note that this is **not** the same as `transformers.Expm1Features`, which computes `exp(x - 1)`.

## SquareTarget

Fit on `y ** 2`, predict back with `sqrt(p)`. Requires a non-negative target.

## SqrtTarget

Fit on `sqrt(y)`, predict back with `p ** 2`. Requires a non-negative target. A gentler compression than
`Log1pTarget`.

## CubeTarget

Fit on `y ** 3`, predict back with `cbrt(p)`. The only transformation in the set with no restriction at all:
it is strictly increasing and bijective over the whole real line, so it accepts negative targets, zero and
positive targets alike.

## ReciprocalTarget

Fit on `1 / y`, predict back with `1 / p`. Requires a target that is never exactly zero. Note that very
small targets are accepted but produce enormous transformed values.
