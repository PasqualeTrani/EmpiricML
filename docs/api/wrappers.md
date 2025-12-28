# empml.wrappers

| Object | Description |
| :--- | :--- |
| `SKlearnWrapper` | Wraps sklearn-like estimators for Polars LazyFrames. |
| `TorchWrapper` | Wrapper for PyTorch modules compatible with Polars LazyFrames. |

## SKlearnWrapper
Wraps sklearn-like estimators for Polars LazyFrames.

### Methods

```python
def __init__(self, estimator: SKlearnEstimator, features: List[str], target: str):
    pass

def fit(self, lf: pl.LazyFrame, **fit_kwargs):
    """Fit the wrapped estimator using Polars LazyFrame."""
    X = lf.select(self.features).collect().to_numpy()
    y = lf.select(self.target).collect().to_series().to_numpy()
    
    self.estimator.fit(X, y, **fit_kwargs)
    return self

def predict(self, lf: pl.LazyFrame) -> np.ndarray:
    """Predict using the wrapped estimator with Polars LazyFrame."""
    X = lf.select(self.features).collect().to_numpy()
    return self.estimator.predict(X)

def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
    """
    Predict class probabilities using the wrapped estimator with Polars LazyFrame.
    
    Only available if the wrapped estimator has a predict_proba method.
    """
    X = lf.select(self.features).collect().to_numpy()
    return self.estimator.predict_proba(X)
```

## TorchWrapper
Wrapper for PyTorch modules compatible with Polars LazyFrames.

### Methods

```python
def __init__(
    self,
    module: type[nn.Module],
    features: List[str],
    target: str,
    task: str = 'regression',
    # Module architecture parameters
    input_dim: Optional[int] = None,
    hidden_layers: Optional[List[int]] = None,
    output_dim: Optional[int] = None,
    # Skorch training parameters
    max_epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 128,
    optimizer: Any = None,
    criterion: Any = None,
    # Skorch regularization & training
    train_split: Any = None,
    callbacks: Optional[List] = None,
    warm_start: bool = False,
    verbose: int = 0,
    # Skorch device & performance
    device: str = 'cpu',
    # Skorch iterator settings
    iterator_train: Any = None,
    iterator_train__shuffle: bool = True,
    iterator_train__num_workers: int = 0,
    iterator_valid: Any = None,
    iterator_valid__shuffle: bool = False,
    # Additional skorch parameters (passed as **kwargs to NeuralNet)
    **kwargs
):
    pass

def fit(self, lf: pl.LazyFrame, **fit_kwargs):
    """
    Fit the wrapped PyTorch model using Polars LazyFrame.
    
    Automatically converts data to float32 as required by PyTorch and creates
    the skorch estimator on first call.
    """
    pass

def predict(self, lf: pl.LazyFrame) -> np.ndarray:
    """
    Predict using the wrapped PyTorch model with Polars LazyFrame.
    
    Automatically converts input to float32 and flattens output for regressors.
    """
    pass

def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
    """
    Predict class probabilities using the wrapped PyTorch classifier.
    
    Only available for classification tasks. Automatically converts input to float32.
    """
    pass
```
