# empml.metrics

| Object | Description |
| :--- | :--- |
| `MSE` | Mean Squared Error. |
| `RMSE` | Root Mean Squared Error. |
| `MAE` | Mean Absolute Error. |
| `MSLE` | Mean Squared Logarithmic Error. |
| `RMSLE` | Root Mean Squared Logarithmic Error. |
| `MAPE` | Mean Absolute Percentage Error. |
| `WMAE` | Weighted Mean Absolute Error. |
| `Accuracy` | Classification accuracy. |
| `Precision` | Precision for binary classification. |
| `Recall` | Recall (Sensitivity) for binary classification. |
| `F1Score` | F1 Score for binary classification. |
| `Specificity` | Specificity (True Negative Rate) for binary classification. |
| `BalancedAccuracy` | Balanced Accuracy for binary classification. |
| `ROCAUC` | Area Under the ROC Curve for binary classification. |

## MSE
Mean Squared Error.

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (pl.col(target) - pl.col(preds)).pow(2).mean()
    return lf.select(metric_expr).collect().item()
```

## RMSE
Root Mean Squared Error.

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (pl.col(target) - pl.col(preds)).pow(2).mean().sqrt()
    return lf.select(metric_expr).collect().item()
```

## MAE
Mean Absolute Error.

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (pl.col(target) - pl.col(preds)).abs().mean()
    return lf.select(metric_expr).collect().item()
```

## MSLE
Mean Squared Logarithmic Error. Uses log1p for numerical stability.

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (
        (pl.col(target).log1p() - pl.col(preds).log1p()).pow(2).mean()
    )
    return lf.select(metric_expr).collect().item()
```

## RMSLE
Root Mean Squared Logarithmic Error. Uses log1p for numerical stability.

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (
        (pl.col(target).log1p() - pl.col(preds).log1p()).pow(2).mean().sqrt()
    )
    return lf.select(metric_expr).collect().item()
```

## MAPE
Mean Absolute Percentage Error. Returns percentage value (0-100).

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (
        ((pl.col(target) - pl.col(preds)).abs() / pl.col(target).abs())
        .mean() * 100
    )
    return lf.select(metric_expr).collect().item()
```

## WMAE
Weighted Mean Absolute Error. Computed as sum(|errors|) / sum(target).

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (
        (pl.col(target) - pl.col(preds)).abs().sum() / pl.col(target).sum()
    )
    return lf.select(metric_expr).collect().item()
```

## Accuracy
Classification accuracy. Proportion of correct predictions.

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (pl.col(target) == pl.col(preds)).mean()
    return lf.select(metric_expr).collect().item()
```

## Precision
Precision for binary classification. TP / (TP + FP).

### Methods

```python
def __init__(self, positive_class: int = 1):
    pass

def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (
        ((pl.col(preds) == self.positive_class) & (pl.col(target) == self.positive_class)).sum() /
        (pl.col(preds) == self.positive_class).sum()
    )
    return lf.select(metric_expr).collect().item()
```

## Recall
Recall (Sensitivity) for binary classification. TP / (TP + FN).

### Methods

```python
def __init__(self, positive_class: int = 1):
    pass

def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (
        ((pl.col(preds) == self.positive_class) & (pl.col(target) == self.positive_class)).sum() /
        (pl.col(target) == self.positive_class).sum()
    )
    return lf.select(metric_expr).collect().item()
```

## F1Score
F1 Score for binary classification. Harmonic mean of precision and recall.

### Methods

```python
def __init__(self, positive_class: int = 1):
    pass

def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    tp = ((pl.col(preds) == self.positive_class) & (pl.col(target) == self.positive_class)).sum()
    pred_pos = (pl.col(preds) == self.positive_class).sum()
    actual_pos = (pl.col(target) == self.positive_class).sum()
    
    precision = tp / pred_pos
    recall = tp / actual_pos
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return lf.select(f1).collect().item()
```

## Specificity
Specificity (True Negative Rate) for binary classification. TN / (TN + FP).

### Methods

```python
def __init__(self, positive_class: int = 1):
    pass

def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    metric_expr = (
        ((pl.col(preds) != self.positive_class) & (pl.col(target) != self.positive_class)).sum() /
        (pl.col(target) != self.positive_class).sum()
    )
    return lf.select(metric_expr).collect().item()
```

## BalancedAccuracy
Balanced Accuracy for binary classification. (Recall + Specificity) / 2.

### Methods

```python
def __init__(self, positive_class: int = 1):
    pass

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
```

## ROCAUC
Area Under the ROC Curve for binary classification. Requires probability scores.

### Methods

```python
def compute_metric(self, lf: pl.LazyFrame, target: str, preds: str) -> float:
    # Collect data and convert to numpy for sklearn computation
    df = lf.select([pl.col(target), pl.col(preds)]).collect()
    y_true = df[target].to_numpy()
    y_scores = df[preds].to_numpy()
    
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)
```
