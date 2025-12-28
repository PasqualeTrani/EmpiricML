# empml.base

| Object | Description |
| :--- | :--- |
| `DataDownloader` | Abstract class for downloading data into Polars LazyFrames. |
| `CVGenerator` | Abstract base class for cross-validation splitting strategies. |
| `Metric` | Abstract base class for performance metrics. |
| `BaseTransformer` | Abstract base class for transformers that work with Polars LazyFrames. |
| `BaseEstimator` | Abstract base class for estimators that work with Polars LazyFrames. |
| `SKlearnEstimator` | Protocol for sklearn-like estimators. |

## DataDownloader
Abstract class for downloading data into Polars LazyFrames.

### Abstract Methods

```python
@abstractmethod
def get_data(self) -> pl.LazyFrame:
    pass
```

## CVGenerator
Abstract base class for cross-validation splitting strategies.

### Abstract Methods

```python
@abstractmethod
def split(self, lf : pl.LazyFrame, row_id : str) -> List[Tuple[np.array]]:
    """Generate a list of tuple with two elements: the first one is an array containing the row indexes for the train dataset, while the second contains the row indexes for the validation dataset"""
    pass 
```

## Metric
Abstract base class for performance metrics.

### Abstract Methods

```python
@abstractmethod
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    """
    Computes the metric, strictly requiring a Polars LazyFrame as input.
    The final calculation executes the lazy plan to return a scalar float.
    """
    pass
```

## BaseTransformer
Abstract base class for transformers that work with Polars LazyFrames.

### Abstract Methods

```python
@abstractmethod
def fit(self, lf: pl.LazyFrame):
    """Fit the transformer on the data."""
    pass

@abstractmethod
def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Transform the data."""
    pass
```

## BaseEstimator
Abstract base class for estimators that work with Polars LazyFrames.

### Abstract Methods

```python
@abstractmethod
def fit(self, df : pl.LazyFrame):
    """Fit the estimator on the data."""
    pass

@abstractmethod
def predict(self, df : pl.LazyFrame):
    """Predict by using the fitted estimator."""
    pass
```

## SKlearnEstimator
Protocol for sklearn-like estimators.

### Protocol Methods

```python
def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
    """Fit the estimator."""
    ...

def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions."""
    ...
```
