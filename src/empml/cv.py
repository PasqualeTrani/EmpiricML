from typing import List, Tuple

# wranglers 
import polars as pl # type:ignore
import numpy as np # type:ignore
import pandas as pd

# internal imports 
from empml.base import CVGenerator # base class 

# ------------------------------------------------------------------------------------------
# Implementations of the CVGenerator base class
# ------------------------------------------------------------------------------------------

class KFold(CVGenerator):

    def __init__(self, n_splits : int = 5, random_state : int = None):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, lf : pl.LazyFrame, row_id : str) -> List[Tuple[np.array]]:

        shuffle_df : pl.DataFrame = lf.collect().sample(fraction=1, seed=self.random_state, shuffle=True)
        n_rows : int = shuffle_df.shape[0]
        slice_size = int(n_rows/self.n_splits)

        valid_row_id = [shuffle_df.slice(offset=slice_size * i, length = slice_size)[row_id].to_numpy() for i in range(self.n_splits)]

        result = [
            (
                np.concatenate([valid_row_id[j] for j in range(self.n_splits) if j!=i]), 
                row
            ) 
            for i, row in enumerate(valid_row_id)
        ]

        return result 


class StratifiedKFold(CVGenerator):
    """
    Stratified K-Fold split that preserves the percentage of samples for each class.
    """
    def __init__(self, target_col : str, n_splits : int = 5, random_state : int = None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_col = target_col # column name for stratification 

    def split(self, lf : pl.LazyFrame, row_id : str) -> List[Tuple[np.array]]:
    
        df : pl.DataFrame = lf.collect()
        
        # Get unique classes and their counts
        class_indices = {}
        for class_val in df[self.target_col].unique().sort():
            class_df = df.filter(pl.col(self.target_col) == class_val)
            shuffled = class_df.sample(fraction=1, seed=self.random_state, shuffle=True)
            class_indices[class_val] = shuffled[row_id].to_numpy()
        
        # Split each class into n_splits folds
        fold_indices = [[] for _ in range(self.n_splits)]
        for class_val, indices in class_indices.items():
            n_samples = len(indices)
            fold_size = n_samples // self.n_splits
            
            for i in range(self.n_splits):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < self.n_splits - 1 else n_samples
                fold_indices[i].extend(indices[start_idx:end_idx])
        
        # Convert to numpy arrays
        fold_indices = [np.array(fold) for fold in fold_indices]
        
        # Create train/validation splits
        result = [
            (
                np.concatenate([fold_indices[j] for j in range(self.n_splits) if j != i]),
                fold_indices[i]
            )
            for i in range(self.n_splits)
        ]
        
        return result


class GroupKFold(CVGenerator):
    """
    Group K-Fold split that ensures samples from the same group don't appear in both train and validation.
    """
    
    def __init__(self, group_col : str, n_splits : int = 5, random_state : int = None):

        self.n_splits = n_splits
        self.random_state = random_state
        self.group_col = group_col

    def split(self, lf : pl.LazyFrame, row_id : str) -> List[Tuple[np.array]]:
        
        df : pl.DataFrame = lf.collect()
        
        # Get unique groups
        unique_groups = df[self.group_col].unique().to_numpy()
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_groups)
        
        # Split groups into folds
        n_groups = len(unique_groups)
        fold_size = n_groups // self.n_splits
        
        group_folds = []
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_splits - 1 else n_groups
            group_folds.append(unique_groups[start_idx:end_idx])
        
        # Get row IDs for each fold based on group assignment
        fold_indices = []
        for fold_groups in group_folds:
            fold_df = df.filter(pl.col(self.group_col).is_in(fold_groups))
            fold_indices.append(fold_df[row_id].to_numpy())
        
        # Create train/validation splits
        result = [
            (
                np.concatenate([fold_indices[j] for j in range(self.n_splits) if j != i]),
                fold_indices[i]
            )
            for i in range(self.n_splits)
        ]
        
        return result
    
class TimeSeriesSplit(CVGenerator):

    def __init__(self, windows : List[Tuple[str]], date_col : str):

        self.windows = windows
        self.date_col = date_col

    def split(self, lf : pl.LazyFrame, row_id : str) -> List[Tuple[np.array]]:

        # check if date_col is datetime, otherwise cast it 
        dates_dtype = lf.select([self.date_col]).collect().to_series().dtype
        is_datetime = dates_dtype in [pl.Datetime, pl.Datetime('ms'), pl.Datetime('us'), pl.Datetime('ns')]

        if not is_datetime:
            lf = lf.with_columns(pl.col(self.date_col).str.to_datetime().alias(self.date_col))

        result = [
            (
                lf.filter(pl.col(self.date_col)>=pd.to_datetime(window[0])).filter(pl.col(self.date_col)<pd.to_datetime(window[1])).collect().select([row_id]).to_series().to_numpy(),  # train row ids 
                lf.filter(pl.col(self.date_col)>=pd.to_datetime(window[2])).filter(pl.col(self.date_col)<pd.to_datetime(window[3])).collect().select([row_id]).to_series().to_numpy(),  # valid row ids
            ) 
            for window in self.windows
        ]

        return result 