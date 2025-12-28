# empml.cv

| Object | Description |
| :--- | :--- |
| `KFold` | Standard K-Fold cross-validation with random shuffling. |
| `StratifiedKFold` | Stratified K-Fold that preserves class distribution across folds. |
| `GroupKFold` | Group K-Fold that prevents data leakage between groups. |
| `LeaveOneGroupOut` | Leave-One-Group-Out cross-validation. |
| `TimeSeriesSplit` | Time series cross-validation generator that splits data based on date ranges. |
| `TrainTestSplit` | Single train-test split with random shuffling. |

## KFold
Standard K-Fold cross-validation with random shuffling.

### Methods

```python
def __init__(self, n_splits: int = 5, random_state: int = None):
    pass

def split(self, lf: pl.LazyFrame, row_id: str) -> List[Tuple[np.array]]:
    """
    Generate k-fold train/validation splits.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data to split.
    row_id : str
        Column name containing unique row identifiers.
    
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, validation_indices) tuples for each fold.
    """
    pass
```

## StratifiedKFold
Stratified K-Fold that preserves class distribution across folds.

### Methods

```python
def __init__(self, target_col: str, n_splits: int = 5, random_state: int = None):
    pass

def split(self, lf: pl.LazyFrame, row_id: str) -> List[Tuple[np.array]]:
    """
    Generate stratified k-fold train/validation splits.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data to split.
    row_id : str
        Column name containing unique row identifiers.
    
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, validation_indices) tuples for each fold.
    """
    pass
```

## GroupKFold
Group K-Fold that prevents data leakage between groups.

### Methods

```python
def __init__(self, group_col: str, n_splits: int = 5, random_state: int = None):
    pass

def split(self, lf: pl.LazyFrame, row_id: str) -> List[Tuple[np.array]]:
    """
    Generate group-aware k-fold train/validation splits.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data to split.
    row_id : str
        Column name containing unique row identifiers.
    
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, validation_indices) tuples for each fold.
    """
    pass
```

## LeaveOneGroupOut
Leave-One-Group-Out cross-validation.

### Methods

```python
def __init__(self, group_col: str):
    pass

def split(self, lf: pl.LazyFrame, row_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate leave-one-group-out train/validation splits.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data to split.
    row_id : str
        Column name containing unique row identifiers.
    
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, validation_indices) tuples, one per group.
    """
    pass
```

## TimeSeriesSplit
Time series cross-validation generator that splits data based on date ranges.

### Methods

```python
def __init__(self, windows: List[Tuple[str, str, str, str]], date_col: str):
    """
    Initialize the TimeSeriesSplit cross-validator.
    
    Parameters
    ----------
    windows : List[Tuple[str, str, str, str]]
        List of date range tuples defining train/validation splits for each fold.
        Each tuple contains (train_start, train_end, val_start, val_end).
    date_col : str
        Name of the date/timestamp column in the dataset.
    """
    pass

def split(self, lf: pl.LazyFrame, row_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate train/validation row indices for each fold based on date windows.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data containing the date column and row identifier.
    row_id : str
        Name of the column containing unique row identifiers.
    
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of tuples, one per fold, where each tuple contains:
        - train_indices: numpy array of row IDs for training
        - val_indices: numpy array of row IDs for validation
    
    Notes
    -----
    - If date_col is not already datetime type, it will be automatically converted
      from string format using polars' str.to_datetime() method.
    - Date filtering uses inclusive start (>=) and exclusive end (<) boundaries.
    """
    pass
```

## TrainTestSplit
Single train-test split with random shuffling.

### Methods

```python
def __init__(self, test_size: float = 0.2, random_state: int = None):
    pass

def split(self, lf: pl.LazyFrame, row_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate single train/test split.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        Input data to split.
    row_id : str
        Column name containing unique row identifiers.
    
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Single-element list containing (train_indices, test_indices) tuple.
    """
    pass
```
