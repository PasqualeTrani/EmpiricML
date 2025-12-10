# wranglers 
import polars as pl
import numpy as np

# internal imports 
from empml.base import CVGenerator # base class 

# ------------------------------------------------------------------------------------------
# Implementation of the CVGenerator base class
# ------------------------------------------------------------------------------------------

class KFold(CVGenerator):

    def __init__(self, n_splits : int = 5, random_state : int = None):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, lf : pl.LazyFrame, row_id : str) -> list[tuple[np.array]]:

        shuffle_df : pl.DataFrame = lf.collect().sample(fraction=1, seed=0, shuffle=True)
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