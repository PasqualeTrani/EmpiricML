import polars as pl # type:ignore
import numpy as np  # type:ignore
from typing import List
from empml.base import BaseEstimator
from sklearn.base import BaseEstimator as SklearnBaseEstimator # type:ignore


class RegressorWrapper(BaseEstimator):
    """
    Wraps any sklearn-like estimator to work with Polars LazyFrames.
    
    Parameters
    ----------
    estimator : object
        Any sklearn-like estimator with fit() and predict() methods
    features : List[str]
        List of feature column names to use for training/prediction
    target : str
        Target column name for training
    
    Example
    -------
    >>> from lightgbm import LGBMRegressor
    >>> model = RegressorWrapper(
    ...     LGBMRegressor(n_estimators=100),
    ...     features=['feature1', 'feature2'],
    ...     target='target'
    ... )
    >>> model.fit(train_lf)
    >>> predictions = model.predict(test_lf)
    """
    
    def __init__(self, estimator : SklearnBaseEstimator, features: List[str], target: str):
        self.estimator = estimator
        self.features = features
        self.target = target
    
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
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped estimator."""
        return getattr(self.estimator, name)



class ClassifierWrapper(BaseEstimator):
    """
    Wraps any sklearn-like classifier to work with Polars LazyFrames.
    
    Parameters
    ----------
    estimator : object
        Any sklearn-like classifier with fit() and predict() methods
    features : List[str]
        List of feature column names to use for training/prediction
    target : str
        Target column name for training
    
    Example
    -------
    >>> from lightgbm import LGBMClassifier
    >>> model = ClassifierWrapper(
    ...     LGBMClassifier(n_estimators=100),
    ...     features=['feature1', 'feature2'],
    ...     target='target'
    ... )
    >>> model.fit(train_lf)
    >>> predictions = model.predict(test_lf)
    """
    
    def __init__(self, estimator, features: List[str], target: str):
        self.estimator = estimator
        self.features = features
        self.target = target
    
    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        """Fit the wrapped classifier using Polars LazyFrame."""
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()
        
        self.estimator.fit(X, y, **fit_kwargs)
        return self
    
    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        """Predict using the wrapped classifier with Polars LazyFrame."""
        X = lf.select(self.features).collect().to_numpy()
        return self.estimator.predict(X)
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        """Predict class probabilities using the wrapped classifier with Polars LazyFrame."""
        X = lf.select(self.features).collect().to_numpy()
        return self.estimator.predict_proba(X)
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped estimator."""
        return getattr(self.estimator, name)