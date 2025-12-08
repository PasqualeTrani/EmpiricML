# wranglers 
import polars as pl 
import numpy as np

# internal imports 
from empml.base import Metric # base class 

# ------------------------------------------------------------------------------------------
# Implementations of the Metric base class
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