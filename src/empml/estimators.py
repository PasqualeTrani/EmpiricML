import polars as pl 
import numpy as np  
from typing import List
from empml.base import BaseEstimator, SKlearnEstimator

# streaming engine as the default for .collect()
pl.Config.set_engine_affinity(engine='streaming')

class EstimatorWrapper(BaseEstimator):
    """
    Wraps any sklearn-like estimator to work with Polars LazyFrames.
    
    Works with both regressors and classifiers. For classifiers with predict_proba,
    that method will be automatically available.
    
    Parameters
    ----------
    estimator : SklearnBaseEstimator
        Any sklearn-like estimator with fit() and predict() methods
    features : List[str]
        List of feature column names to use for training/prediction
    target : str
        Target column name for training
    
    Examples
    --------
    >>> from lightgbm import LGBMRegressor, LGBMClassifier
    >>> 
    >>> # Regressor
    >>> regressor = EstimatorWrapper(
    ...     LGBMRegressor(n_estimators=100),
    ...     features=['feature1', 'feature2'],
    ...     target='target'
    ... )
    >>> regressor.fit(train_lf)
    >>> predictions = regressor.predict(test_lf)
    >>> 
    >>> # Classifier
    >>> classifier = EstimatorWrapper(
    ...     LGBMClassifier(n_estimators=100),
    ...     features=['feature1', 'feature2'],
    ...     target='target'
    ... )
    >>> classifier.fit(train_lf)
    >>> predictions = classifier.predict(test_lf)
    >>> probabilities = classifier.predict_proba(test_lf)  # Available for classifiers
    """
    
    def __init__(self, estimator: SKlearnEstimator, features: List[str], target: str):
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        """
        Predict class probabilities using the wrapped estimator with Polars LazyFrame.
        
        Only available if the wrapped estimator has a predict_proba method.
        
        Raises
        ------
        AttributeError
            If the wrapped estimator doesn't have predict_proba method
        """
        X = lf.select(self.features).collect().to_numpy()
        return self.estimator.predict_proba(X)