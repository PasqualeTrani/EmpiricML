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


# Convenience factory functions
def lgbm_reg(features: List[str], target: str, **kwargs):
    from lightgbm import LGBMRegressor # type:ignore
    return RegressorWrapper(LGBMRegressor(**kwargs), features, target)

def xgb_reg(features: List[str], target: str, **kwargs):
    from xgboost import XGBRegressor # type:ignore
    return RegressorWrapper(XGBRegressor(**kwargs), features, target)

def ctb_reg(features: List[str], target: str, **kwargs):
    from catboost import CatBoostRegressor # type:ignore
    return RegressorWrapper(CatBoostRegressor(**kwargs), features, target)

def rf_reg(features: List[str], target: str, **kwargs):
    from sklearn.ensemble import RandomForestRegressor # type:ignore
    return RegressorWrapper(RandomForestRegressor(**kwargs), features, target)

def lr_reg(features: List[str], target: str, **kwargs):
    from sklearn.linear_model import LinearRegression # type:ignore
    return RegressorWrapper(LinearRegression(**kwargs), features, target)

def en_reg(features: List[str], target: str, **kwargs):
    from sklearn.linear_model import ElasticNet # type:ignore
    return RegressorWrapper(ElasticNet(**kwargs), features, target)

def knn_reg(features: List[str], target: str, **kwargs):
    from sklearn.neighbors import KNeighborsRegressor # type:ignore
    return RegressorWrapper(KNeighborsRegressor(**kwargs), features, target)

def dt_reg(features: List[str], target: str, **kwargs):
    from sklearn.tree import DecisionTreeRegressor # type:ignore
    return RegressorWrapper(DecisionTreeRegressor(**kwargs), features, target)

def et_reg(features: List[str], target: str, **kwargs):
    from sklearn.ensemble import ExtraTreesRegressor # type:ignore
    return RegressorWrapper(ExtraTreesRegressor(**kwargs), features, target)

def svr_reg(features: List[str], target: str, **kwargs):
    from sklearn.svm import SVR # type:ignore
    return RegressorWrapper(SVR(**kwargs), features, target)

def hgb_reg(features: List[str], target: str, **kwargs):
    from sklearn.ensemble import HistGradientBoostingRegressor # type:ignore
    return RegressorWrapper(HistGradientBoostingRegressor(**kwargs), features, target)