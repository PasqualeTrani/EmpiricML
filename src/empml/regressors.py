# base imports 
from abc import ABC, abstractmethod

# data wranglers 
import polars as pl
import numpy as np

# regressors
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR

# ------------------------------------------------------------------------------------------
# DEFINITION OF THE ABSTRACT CLASS 
# ------------------------------------------------------------------------------------------

class BaseRegressor(ABC):
    @abstractmethod
    def fit(self, df : pl.LazyFrame):
        pass
    
    @abstractmethod
    def predict(self, df : pl.LazyFrame):
        pass

# ------------------------------------------------------------------------------------------
# REGRESSOR IMPLEMENTATIONS 
# ------------------------------------------------------------------------------------------

class lgbm_reg(LGBMRegressor, BaseRegressor):
    """
    Extends LGBMRegressor to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **lgbm_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**lgbm_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)


class xgb_reg(XGBRegressor, BaseRegressor):
    """
    Extends XGBRegressor to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **xgb_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**xgb_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)
    

class ctb_reg(CatBoostRegressor, BaseRegressor):
    """
    Extends CatBoostRegressor to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **cat_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**cat_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)
    
class lr_reg(LinearRegression, BaseRegressor):
    """
    Extends LinearRegression to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)


class en_reg(ElasticNet, BaseRegressor):
    """
    Extends ElasticNet to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)
    

class knn_reg(KNeighborsRegressor, BaseRegressor):
    """
    Extends KNeighborsRegressor to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)

class dt_reg(DecisionTreeRegressor, BaseRegressor):
    """
    Extends DecisionTreeRegressor to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)


class rf_reg(RandomForestRegressor, BaseRegressor):
    """
    Extends RandomForestRegressor to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)


class et_reg(ExtraTreesRegressor, BaseRegressor):
    """
    Extends ExtraTreesRegressor to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)
    
    
class svr_reg(SVR, BaseRegressor):
    """
    Extends SVR to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)


class hgb_reg(HistGradientBoostingRegressor, BaseRegressor):
    """
    Extends HistGradientBoostingRegressor to accept feature and target names 
    on initialization and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(**sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)