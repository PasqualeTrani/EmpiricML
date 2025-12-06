from abc import ABC, abstractmethod

import polars as pl 
import numpy as np


class Metric(ABC):
    @abstractmethod
    def compute_metric(self, df : pl.DataFrame, target : str, preds : str) -> float:
        pass

class mse(Metric):
    def compute_metric(self, df : pl.DataFrame, target : str, preds : str) -> float:
        return (df[target] - df[preds]).pow(2).mean() 
    
class rmse(Metric):
    def compute_metric(self, df : pl.DataFrame, target : str, preds : str) -> float:
        return np.sqrt((df[target] - df[preds]).pow(2).mean())
    
class mae(Metric):
    def compute_metric(self, df : pl.DataFrame, target : str, preds : str) -> float:
        return (df[target] - df[preds]).abs().mean() 