# base imports 
from abc import ABC, abstractmethod

# data wranglers 
import polars as pl
import numpy as np

# classifiers
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ------------------------------------------------------------------------------------------
# DEFINITION OF THE ABSTRACT CLASS 
# ------------------------------------------------------------------------------------------

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, df : pl.LazyFrame):
        pass
    
    @abstractmethod
    def predict(self, df : pl.LazyFrame):
        pass

    @abstractmethod
    def predict_proba(self, df : pl.LazyFrame):
        pass

# ------------------------------------------------------------------------------------------
# CLASSIFIER IMPLEMENTATIONS 
# ------------------------------------------------------------------------------------------

class lgbm_clf(LGBMClassifier, BaseClassifier):
    """
    Extends LGBMClassifier to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


class xgb_clf(XGBClassifier, BaseClassifier):
    """
    Extends XGBClassifier to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


class ctb_clf(CatBoostClassifier, BaseClassifier):
    """
    Extends CatBoostClassifier to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


## Linear Models (Logistic Regression)

class lr_clf(LogisticRegression, BaseClassifier):
    """
    Extends LogisticRegression to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


class en_clf(LogisticRegression, BaseClassifier):
    """
    Extends LogisticRegression with L1/L2 penalties (equivalent of ElasticNet) to 
    accept feature and target names on initialization and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, penalty='elasticnet', solver='saga', **sk_kwargs):
        self.features = features
        self.target = target
        
        super().__init__(penalty=penalty, solver=solver, **sk_kwargs)

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        return super().fit(X, y, **fit_kwargs)

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict(X)
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)

    
## K-Nearest Neighbors

class knn_clf(KNeighborsClassifier, BaseClassifier):
    """
    Extends KNeighborsClassifier to accept feature and target names on initialization
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

    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


## Tree-Based Models

class dt_clf(DecisionTreeClassifier, BaseClassifier):
    """
    Extends DecisionTreeClassifier to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


class rf_clf(RandomForestClassifier, BaseClassifier):
    """
    Extends RandomForestClassifier to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


class et_clf(ExtraTreesClassifier, BaseClassifier):
    """
    Extends ExtraTreesClassifier to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)
    
    
class svc_clf(SVC, BaseClassifier):
    """
    Extends SVC to accept feature and target names on initialization
    and fit/predict using Polars LazyFrames.
    """
    def __init__(self, features: list[str], target: str, **sk_kwargs):
        # SVC needs probability=True for predict_proba
        if 'probability' not in sk_kwargs:
            sk_kwargs['probability'] = True
            
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


class hgb_clf(HistGradientBoostingClassifier, BaseClassifier):
    """
    Extends HistGradientBoostingClassifier to accept feature and target names 
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)


class gnb_clf(GaussianNB, BaseClassifier):
    """
    Extends GaussianNB to accept feature and target names on initialization
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
    
    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.features).collect().to_numpy()
        
        return super().predict_proba(X)