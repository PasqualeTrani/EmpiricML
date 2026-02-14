import numpy as np
import pandas as pd

# wranglers
import polars as pl

# internal imports
from empml.base import CVGenerator  # base class

# streaming engine as the default for .collect()
pl.Config.set_engine_affinity(engine="streaming")

# ------------------------------------------------------------------------------------------
# Implementations of the CVGenerator base class
# ------------------------------------------------------------------------------------------


class KFold(CVGenerator):
    """
    Standard K-Fold cross-validation with random shuffling.

    Randomly shuffles data and splits it into k equal-sized folds. Each fold
    serves as validation set once while remaining folds form the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int, optional
        Random seed for reproducible shuffling. If None, shuffling is random.

    Examples
    --------
    >>> cv = KFold(n_splits=5, random_state=42)
    >>> splits = cv.split(data, row_id='id')
    """

    def __init__(self, n_splits: int = 5, random_state: int = None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, lf: pl.LazyFrame, row_id: str) -> list[tuple[np.array]]:
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
        shuffle_df: pl.DataFrame = lf.collect().sample(
            fraction=1, seed=self.random_state, shuffle=True
        )
        n_rows: int = shuffle_df.shape[0]
        slice_size = int(n_rows / self.n_splits)

        valid_row_id = [
            shuffle_df.slice(offset=slice_size * i, length=slice_size)[
                row_id
            ].to_numpy()
            for i in range(self.n_splits)
        ]

        result = [
            (
                np.concatenate(
                    [valid_row_id[j] for j in range(self.n_splits) if j != i]
                ),
                row,
            )
            for i, row in enumerate(valid_row_id)
        ]

        return result


class StratifiedKFold(CVGenerator):
    """
    Stratified K-Fold that preserves class distribution across folds.

    Ensures each fold maintains the same proportion of samples for each class
    as the original dataset. Useful for imbalanced classification problems.

    Parameters
    ----------
    target_col : str
        Column name containing class labels for stratification.
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int, optional
        Random seed for reproducible shuffling.

    Examples
    --------
    >>> cv = StratifiedKFold(target_col='label', n_splits=5, random_state=42)
    >>> splits = cv.split(data, row_id='id')
    """

    def __init__(self, target_col: str, n_splits: int = 5, random_state: int = None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_col = target_col

    def split(self, lf: pl.LazyFrame, row_id: str) -> list[tuple[np.array]]:
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
        df: pl.DataFrame = lf.collect()

        # Get unique classes and their counts
        class_indices = {}
        for class_val in df[self.target_col].unique().sort():
            class_df = df.filter(pl.col(self.target_col) == class_val)
            shuffled = class_df.sample(fraction=1, seed=self.random_state, shuffle=True)
            class_indices[class_val] = shuffled[row_id].to_numpy()

        # Split each class into n_splits folds
        fold_indices = [[] for _ in range(self.n_splits)]
        for _class_val, indices in class_indices.items():
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
                np.concatenate(
                    [fold_indices[j] for j in range(self.n_splits) if j != i]
                ),
                fold_indices[i],
            )
            for i in range(self.n_splits)
        ]

        return result


class GroupKFold(CVGenerator):
    """
    Group K-Fold that prevents data leakage between groups.

    Ensures samples from the same group appear only in training or validation,
    never both. Essential when data points are not independent (e.g., multiple
    observations per patient, store, or time period).

    Parameters
    ----------
    group_col : str
        Column name containing group identifiers.
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int, optional
        Random seed for reproducible group shuffling.

    Examples
    --------
    >>> cv = GroupKFold(group_col='customer_id', n_splits=5, random_state=42)
    >>> splits = cv.split(data, row_id='id')
    """

    def __init__(self, group_col: str, n_splits: int = 5, random_state: int = None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.group_col = group_col

    def split(self, lf: pl.LazyFrame, row_id: str) -> list[tuple[np.array]]:
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
        df: pl.DataFrame = lf.collect()

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
                np.concatenate(
                    [fold_indices[j] for j in range(self.n_splits) if j != i]
                ),
                fold_indices[i],
            )
            for i in range(self.n_splits)
        ]

        return result


class LeaveOneGroupOut(CVGenerator):
    """
    Leave-One-Group-Out cross-validation.

    Creates one fold for each unique group, where that group serves as the
    validation set and all other groups form the training set. Results in
    n_groups folds total. Useful for testing model generalization to entirely
    new groups (e.g., new customers, stores, or time periods).

    Parameters
    ----------
    group_col : str
        Column name containing group identifiers.

    Examples
    --------
    >>> cv = LeaveOneGroupOut(group_col='store_id')
    >>> splits = cv.split(data, row_id='id')
    >>> # If data has 10 stores, this creates 10 folds

    Notes
    -----
    - Number of folds equals the number of unique groups
    - Can result in many folds if there are many groups
    - Training set size varies based on validation group size
    """

    def __init__(self, group_col: str):
        self.group_col = group_col

    def split(
        self, lf: pl.LazyFrame, row_id: str
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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
        df: pl.DataFrame = lf.collect()

        # Get unique groups
        unique_groups = df[self.group_col].unique().sort().to_numpy()

        # Create one fold per group
        result = []
        for group in unique_groups:
            # Validation: current group
            val_df = df.filter(pl.col(self.group_col) == group)
            val_indices = val_df[row_id].to_numpy()

            # Training: all other groups
            train_df = df.filter(pl.col(self.group_col) != group)
            train_indices = train_df[row_id].to_numpy()

            result.append((train_indices, val_indices))

        return result


class TimeSeriesSplit(CVGenerator):
    """
    Time series cross-validation generator that splits data based on date ranges.

    This class implements a custom time series splitting strategy where train and
    validation windows are explicitly defined by date ranges. Unlike rolling window
    approaches, this allows for flexible, non-contiguous splits tailored to specific
    time periods (e.g., avoiding seasonal effects, testing specific periods).

    Attributes
    ----------
    windows : List[Tuple[str, str, str, str]]
        List of date range tuples, where each tuple contains:
        - windows[i][0]: Train start date (inclusive)
        - windows[i][1]: Train end date (exclusive)
        - windows[i][2]: Validation start date (inclusive)
        - windows[i][3]: Validation end date (exclusive)
        Dates should be in a format parseable by pandas.to_datetime (e.g., 'YYYY-MM-DD').
    date_col : str
        Name of the column in the DataFrame containing dates/timestamps.

    Examples
    --------
    >>> # Create a time series split with two folds
    >>> windows = [
    ...     ('2023-01-01', '2023-07-01', '2023-07-01', '2023-07-31'),  # Fold 1
    ...     ('2023-01-01', '2023-08-01', '2023-08-01', '2023-08-31'),  # Fold 2
    ... ]
    >>> cv = TimeSeriesSplit(windows=windows, date_col='transaction_date')
    >>> splits = cv.split(data, row_id='id')

    Notes
    -----
    - The split method automatically converts string dates to datetime if needed.
    - Train and validation windows can overlap or have gaps depending on requirements.
    - Each fold's train set can have different sizes, allowing for expanding window strategies.
    """

    def __init__(self, windows: list[tuple[str, str, str, str]], date_col: str):
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
        self.windows = windows
        self.date_col = date_col

    def split(
        self, lf: pl.LazyFrame, row_id: str
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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
        # check if date_col is datetime, otherwise cast it
        dates_dtype = lf.select([self.date_col]).collect().to_series().dtype
        is_datetime = dates_dtype in [
            pl.Datetime,
            pl.Datetime("ms"),
            pl.Datetime("us"),
            pl.Datetime("ns"),
        ]

        if not is_datetime:
            lf = lf.with_columns(
                pl.col(self.date_col).str.to_datetime().alias(self.date_col)
            )

        result = [
            (
                lf.filter(pl.col(self.date_col) >= pd.to_datetime(window[0]))
                .filter(pl.col(self.date_col) < pd.to_datetime(window[1]))
                .collect()
                .select([row_id])
                .to_series()
                .to_numpy(),  # train row ids
                lf.filter(pl.col(self.date_col) >= pd.to_datetime(window[2]))
                .filter(pl.col(self.date_col) < pd.to_datetime(window[3]))
                .collect()
                .select([row_id])
                .to_series()
                .to_numpy(),  # valid row ids
            )
            for window in self.windows
        ]

        return result


class TrainTestSplit(CVGenerator):
    """
    Single train-test split with random shuffling.

    Randomly shuffles data and splits it into training and test sets based on
    the specified test size. Returns a single split similar to sklearn's
    train_test_split function.

    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split. Should be
        between 0.0 and 1.0.
    random_state : int, optional
        Random seed for reproducible shuffling. If None, shuffling is random.

    Examples
    --------
    >>> cv = TrainTestSplit(test_size=0.2, random_state=42)
    >>> splits = cv.split(data, row_id='id')
    >>> train_ids, test_ids = splits[0]
    """

    def __init__(self, test_size: float = 0.2, random_state: int = None):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self, lf: pl.LazyFrame, row_id: str
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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
        shuffle_df: pl.DataFrame = lf.collect().sample(
            fraction=1, seed=self.random_state, shuffle=True
        )
        n_rows: int = shuffle_df.shape[0]
        test_slice_size = int(n_rows * self.test_size)

        # Split into test and train
        test_row_id = shuffle_df.slice(offset=0, length=test_slice_size)[
            row_id
        ].to_numpy()
        train_row_id = shuffle_df.slice(
            offset=test_slice_size, length=n_rows - test_slice_size
        )[row_id].to_numpy()

        result = [(train_row_id, test_row_id)]

        return result
