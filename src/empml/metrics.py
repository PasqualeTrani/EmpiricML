"""
Metric implementations for regression and classification tasks.

All metrics inherit from the base Metric class and operate on Polars LazyFrames
for efficient computation.
"""

# wranglers 
import polars as pl 
import numpy as np

# internal imports 
from empml.base import Metric # base class 

# ------------------------------------------------------------------------------------------
# REGRESSION Implementations of Metric Class
# ------------------------------------------------------------------------------------------

class MSE(Metric):
    """Mean Squared Error."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (pl.col(target) - pl.col(preds)).pow(2).mean()
        return lf.select(metric_expr).collect().item()
        
class RMSE(Metric):
    """Root Mean Squared Error."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (pl.col(target) - pl.col(preds)).pow(2).mean().sqrt()
        return lf.select(metric_expr).collect().item()
        
class MAE(Metric):
    """Mean Absolute Error."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (pl.col(target) - pl.col(preds)).abs().mean()
        return lf.select(metric_expr).collect().item()
    
class MSLE(Metric):
    """Mean Squared Logarithmic Error. Uses log1p for numerical stability."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (
            (pl.col(target).log1p() - pl.col(preds).log1p()).pow(2).mean()
        )
        return lf.select(metric_expr).collect().item()

class RMSLE(Metric):
    """Root Mean Squared Logarithmic Error. Uses log1p for numerical stability."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (
            (pl.col(target).log1p() - pl.col(preds).log1p()).pow(2).mean().sqrt()
        )
        return lf.select(metric_expr).collect().item()

class MAPE(Metric):
    """Mean Absolute Percentage Error. Returns percentage value (0-100)."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (
            ((pl.col(target) - pl.col(preds)).abs() / pl.col(target).abs())
            .mean() * 100
        )
        return lf.select(metric_expr).collect().item()
    
class WMAE(Metric):
    """Weighted Mean Absolute Error. Computed as sum(|errors|) / sum(target)."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (
            (pl.col(target) - pl.col(preds)).abs().sum() / pl.col(target).sum()
        )
        return lf.select(metric_expr).collect().item()
    

# ------------------------------------------------------------------------------------------
# CLASSIFICATION Implementations of Metric Class
# ------------------------------------------------------------------------------------------

class Accuracy(Metric):
    """Classification accuracy. Proportion of correct predictions."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (pl.col(target) == pl.col(preds)).mean()
        return lf.select(metric_expr).collect().item()

class Precision(Metric):
    """Precision for binary classification. TP / (TP + FP)."""
    
    def __init__(self, positive_class: int = 1):
        super().__init__()
        self.positive_class = positive_class
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (
            ((pl.col(preds) == self.positive_class) & (pl.col(target) == self.positive_class)).sum() /
            (pl.col(preds) == self.positive_class).sum()
        )
        return lf.select(metric_expr).collect().item()

class Recall(Metric):
    """Recall (Sensitivity) for binary classification. TP / (TP + FN)."""
    
    def __init__(self, positive_class: int = 1):
        super().__init__()
        self.positive_class = positive_class
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (
            ((pl.col(preds) == self.positive_class) & (pl.col(target) == self.positive_class)).sum() /
            (pl.col(target) == self.positive_class).sum()
        )
        return lf.select(metric_expr).collect().item()

class F1Score(Metric):
    """F1 Score for binary classification. Harmonic mean of precision and recall."""
    
    def __init__(self, positive_class: int = 1):
        super().__init__()
        self.positive_class = positive_class
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        tp = ((pl.col(preds) == self.positive_class) & (pl.col(target) == self.positive_class)).sum()
        pred_pos = (pl.col(preds) == self.positive_class).sum()
        actual_pos = (pl.col(target) == self.positive_class).sum()
        
        precision = tp / pred_pos
        recall = tp / actual_pos
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return lf.select(f1).collect().item()

class Specificity(Metric):
    """Specificity (True Negative Rate) for binary classification. TN / (TN + FP)."""
    
    def __init__(self, positive_class: int = 1):
        super().__init__()
        self.positive_class = positive_class
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        metric_expr = (
            ((pl.col(preds) != self.positive_class) & (pl.col(target) != self.positive_class)).sum() /
            (pl.col(target) != self.positive_class).sum()
        )
        return lf.select(metric_expr).collect().item()

class BalancedAccuracy(Metric):
    """Balanced Accuracy for binary classification. (Recall + Specificity) / 2."""
    
    def __init__(self, positive_class: int = 1):
        super().__init__()
        self.positive_class = positive_class
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        sensitivity = (
            ((pl.col(preds) == self.positive_class) & (pl.col(target) == self.positive_class)).sum() /
            (pl.col(target) == self.positive_class).sum()
        )
        specificity = (
            ((pl.col(preds) != self.positive_class) & (pl.col(target) != self.positive_class)).sum() /
            (pl.col(target) != self.positive_class).sum()
        )
        balanced_acc = (sensitivity + specificity) / 2
        
        return lf.select(balanced_acc).collect().item()
    
class ROCAUC(Metric):
    """Area Under the ROC Curve for binary classification. Requires probability scores."""
    
    def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
        # Collect data and convert to numpy for sklearn computation
        df = lf.select([pl.col(target), pl.col(preds)]).collect()
        y_true = df[target].to_numpy()
        y_scores = df[preds].to_numpy()
        
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_scores)