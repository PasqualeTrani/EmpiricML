from abc import ABC, abstractmethod

import polars as pl 
import numpy as np


# ------------------------------------------------------------------------------------------
# DEFINITION OF THE ABSTRACT CLASS 
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
# IMPLEMENTATIONS 
# ------------------------------------------------------------------------------------------

class MSE(Metric):
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (pl.col(target) - pl.col(preds)).pow(2).mean()
        return lf.select(metric_expr).collect().item()
        
class RMSE(Metric):
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (pl.col(target) - pl.col(preds)).pow(2).mean().sqrt()
        return lf.select(metric_expr).collect().item()
        
class MAE(Metric):
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (pl.col(target) - pl.col(preds)).abs().mean()
        return lf.select(metric_expr).collect().item()