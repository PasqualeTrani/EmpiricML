"""
In this file there are all the base classes used in the other ones.
"""

# standard import libraries 
from dataclasses import dataclass 
from abc import ABC, abstractmethod
from pathlib import Path

# wranglers 
import polars as pl # type:ignore
import numpy as np # type:ignore

# ------------------------------------------------------------------------------------------
# Data Downloader 
# ------------------------------------------------------------------------------------------

class DataDownloader(ABC):
    """Abstract class for downloading data into Polars LazyFrames."""
    @abstractmethod
    def get_data(self) -> pl.LazyFrame:
        pass

# ------------------------------------------------------------------------------------------
# CV Index Generator
# ------------------------------------------------------------------------------------------

class CVGenerator(ABC):
    @abstractmethod
    def split(self, lf : pl.LazyFrame, row_id : str) -> list[tuple[np.array]]:
        """Generate a list of tuple with two elements: the first one is an array containing the row indexes for the train dataset, while the second contains the row indexes for the validation dataset"""
        pass 


# ------------------------------------------------------------------------------------------
# Performance Metric
# ------------------------------------------------------------------------------------------

class Metric(ABC):
    @abstractmethod
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        """
        Computes the metric, strictly requiring a Polars LazyFrame as input.
        The final calculation executes the lazy plan to return a scalar float.
        """
        pass


# ------------------------------------------------------------------------------------------
# Transformer in Pipelines
# ------------------------------------------------------------------------------------------

class BaseTransformer(ABC):
    """Abstract base class for transformers that work with Polars LazyFrames."""
    
    @abstractmethod
    def fit(self, lf: pl.LazyFrame):
        """Fit the transformer on the data."""
        pass
    
    @abstractmethod
    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Fit and transform in one step."""
        self.fit(lf)
        return self.transform(lf)
    

# ------------------------------------------------------------------------------------------
# Estimator
# ------------------------------------------------------------------------------------------

class BaseEstimator(ABC):
    """Abstract base class for estimators that work with Polars LazyFrames."""
    @abstractmethod
    def fit(self, df : pl.LazyFrame):
        """Fit the estimator on the data."""
        pass
    
    @abstractmethod
    def predict(self, df : pl.LazyFrame):
        """Predict by using the fitted estimator."""
        pass