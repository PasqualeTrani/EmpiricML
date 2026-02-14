from typing import Any

import numpy as np
import polars as pl

from empml.base import BaseEstimator, SKlearnEstimator

# streaming engine as the default for .collect()
pl.Config.set_engine_affinity(engine="streaming")


def _check_torch_available():
    """Check if torch is available and raise informative error if not."""
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required to use neural network models in EmpiricML. "
            "Please install it first:\n\n"
            "  For CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  For CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121\n\n"
            "Visit https://pytorch.org/get-started/locally/ for more options."
        ) from e


def _check_skorch_available():
    """Check if skorch is available and raise informative error if not."""
    try:
        from skorch import NeuralNetClassifier, NeuralNetRegressor  # type: ignore

        return NeuralNetRegressor, NeuralNetClassifier
    except ImportError as e:
        raise ImportError(
            "skorch is required to use neural network models in EmpiricML. "
            "Please install it: pip install skorch"
        ) from e


class SKlearnWrapper(BaseEstimator):
    """
    Wraps any sklearn-like estimator to work with Polars LazyFrames.

    Works with both regressors and classifiers. For classifiers with predict_proba,
    that method will be automatically available.

    Parameters
    ----------
    estimator : SklearnBaseEstimator
        Any sklearn-like estimator with fit() and predict() methods
    features : List[str]
        List of feature column names to use for training/prediction
    target : str
        Target column name for training

    Examples
    --------
    >>> from lightgbm import LGBMRegressor, LGBMClassifier
    >>>
    >>> # Regressor
    >>> regressor = EstimatorWrapper(
    ...     LGBMRegressor(n_estimators=100),
    ...     features=['feature1', 'feature2'],
    ...     target='target'
    ... )
    >>> regressor.fit(train_lf)
    >>> predictions = regressor.predict(test_lf)
    >>>
    >>> # Classifier
    >>> classifier = EstimatorWrapper(
    ...     LGBMClassifier(n_estimators=100),
    ...     features=['feature1', 'feature2'],
    ...     target='target'
    ... )
    >>> classifier.fit(train_lf)
    >>> predictions = classifier.predict(test_lf)
    >>> probabilities = classifier.predict_proba(test_lf)  # Available for classifiers
    """

    def __init__(self, estimator: SKlearnEstimator, features: list[str], target: str):
        self.estimator = estimator
        self.features = features
        self.target = target

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        """Fit the wrapped estimator using Polars LazyFrame."""
        X = lf.select(self.features).collect().to_numpy()
        y = lf.select(self.target).collect().to_series().to_numpy()

        self.estimator.fit(X, y, **fit_kwargs)
        return self

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        """Predict using the wrapped estimator with Polars LazyFrame."""
        X = lf.select(self.features).collect().to_numpy()
        return self.estimator.predict(X)

    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        """
        Predict class probabilities using the wrapped estimator with Polars LazyFrame.

        Only available if the wrapped estimator has a predict_proba method.

        Raises
        ------
        AttributeError
            If the wrapped estimator doesn't have predict_proba method
        """
        X = lf.select(self.features).collect().to_numpy()
        return self.estimator.predict_proba(X)

    def __repr__(self):
        """Return a string representation showing the wrapped estimator and key parameters."""
        estimator_repr = repr(self.estimator)
        return f"EstimatorWrapper({estimator_repr}, features={self.features!r}, target={self.target!r})"


class TorchWrapper(BaseEstimator):
    """
    Universal wrapper for PyTorch modules that creates sklearn-like estimators
    compatible with Polars LazyFrames.

    This wrapper handles the complete integration between PyTorch/skorch and your
    framework by:
    - Automatically creating NeuralNetRegressor or NeuralNetClassifier from PyTorch modules
    - Converting data to float32 as required by PyTorch
    - Flattening regression predictions for framework compatibility
    - Exposing all skorch configuration options

    Parameters
    ----------
    module : torch.nn.Module class
        PyTorch module class (not an instance). Should accept input_dim, hidden_layers,
        output_dim as constructor parameters.
    features : List[str]
        List of feature column names to use for training/prediction
    target : str
        Target column name for training
    task : str, default='regression'
        Type of task: 'regression' or 'classification'
    input_dim : int, optional
        Number of input features (can be inferred from features list)
    hidden_layers : List[int], optional
        List of hidden layer sizes
    output_dim : int, optional
        Number of output units (can be inferred for regression)

    Skorch Training Parameters
    ---------------------------
    max_epochs : int, default=10
        Number of epochs to train
    lr : float, default=0.01
        Learning rate
    batch_size : int, default=128
        Batch size for training
    optimizer : torch.optim.Optimizer, default=torch.optim.SGD
        Optimizer class
    optimizer__momentum : float, default=0.9
        Momentum for SGD optimizer (if using SGD)
    optimizer__weight_decay : float, default=0.0
        L2 regularization parameter
    criterion : torch.nn loss, optional
        Loss function (auto-selected based on task if not provided)

    Skorch Regularization & Training
    ---------------------------------
    train_split : None, False, or skorch.dataset.CVSplit, default=None
        Validation split strategy. None = no validation, False = use all data for training
    callbacks : list of skorch callbacks, default=None
        List of callback instances for training customization
    warm_start : bool, default=False
        Whether to continue training from the current state
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress bar)

    Skorch Device & Performance
    ----------------------------
    device : str, default='cpu'
        Device to use ('cpu', 'cuda', 'cuda:0', etc.)

    Skorch Iterator Settings
    -------------------------
    iterator_train : torch.utils.data.DataLoader, optional
        Custom DataLoader for training
    iterator_train__shuffle : bool, default=True
        Whether to shuffle training data
    iterator_train__num_workers : int, default=0
        Number of workers for data loading
    iterator_valid : torch.utils.data.DataLoader, optional
        Custom DataLoader for validation
    iterator_valid__shuffle : bool, default=False
        Whether to shuffle validation data

    Module-specific Parameters
    ---------------------------
    module__* : any
        Any parameter prefixed with 'module__' will be passed to the PyTorch module
        constructor. For example: module__dropout=0.2, module__activation=nn.ReLU

    Examples
    --------
    Basic regression example:

    >>> from skorch import NeuralNetRegressor
    >>>
    >>> wrapper = TorchWrapper(
    ...     module=TorchMLP,
    ...     features=['age', 'income', 'credit_score'],
    ...     target='loan_amount',
    ...     task='regression',
    ...     hidden_layers=[64, 32],
    ...     max_epochs=50,
    ...     lr=0.001,
    ...     batch_size=64,
    ...     verbose=1
    ... )
    >>> wrapper.fit(train_lf)
    >>> predictions = wrapper.predict(test_lf)

    Classification example with custom settings:

    >>> wrapper = TorchWrapper(
    ...     module=TorchMLP,
    ...     features=['feature1', 'feature2', 'feature3'],
    ...     target='class',
    ...     task='classification',
    ...     hidden_layers=[128, 64, 32],
    ...     output_dim=3,  # 3 classes
    ...     max_epochs=100,
    ...     lr=0.001,
    ...     optimizer=torch.optim.Adam,
    ...     batch_size=32,
    ...     module__dropout=0.3,
    ...     device='cuda',
    ...     verbose=1
    ... )
    >>> wrapper.fit(train_lf)
    >>> predictions = wrapper.predict(test_lf)
    >>> probabilities = wrapper.predict_proba(test_lf)

    Advanced example with callbacks and custom training:

    >>> from skorch.callbacks import EarlyStopping, LRScheduler
    >>> from torch.optim.lr_scheduler import ReduceLROnPlateau
    >>>
    >>> wrapper = TorchWrapper(
    ...     module=TorchMLP,
    ...     features=feature_list,
    ...     target='target',
    ...     task='regression',
    ...     hidden_layers=[256, 128, 64],
    ...     max_epochs=200,
    ...     lr=0.01,
    ...     optimizer=torch.optim.Adam,
    ...     optimizer__weight_decay=1e-5,
    ...     callbacks=[
    ...         EarlyStopping(patience=10),
    ...         LRScheduler(policy=ReduceLROnPlateau, patience=5)
    ...     ],
    ...     train_split=None,  # No validation split
    ...     batch_size=128,
    ...     iterator_train__shuffle=True,
    ...     device='cuda',
    ...     verbose=1
    ... )
    """

    def __init__(
        self,
        module: Any,  # torch.nn.Module class
        features: list[str],
        target: str,
        task: str = "regression",
        # Module architecture parameters
        input_dim: int | None = None,
        hidden_layers: list[int] | None = None,
        output_dim: int | None = None,
        # Skorch training parameters
        max_epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 128,
        optimizer: Any = None,
        criterion: Any = None,
        # Skorch regularization & training
        train_split: Any = None,
        callbacks: list | None = None,
        warm_start: bool = False,
        verbose: int = 0,
        # Skorch device & performance
        device: str = "cpu",
        # Skorch iterator settings
        iterator_train: Any = None,
        iterator_train__shuffle: bool = True,
        iterator_train__num_workers: int = 0,
        iterator_valid: Any = None,
        iterator_valid__shuffle: bool = False,
        # Additional skorch parameters (passed as **kwargs to NeuralNet)
        **kwargs,
    ):

        self.module = module
        self.features = features
        self.target = target
        self.task = task.lower()

        # Module parameters
        self.input_dim = input_dim if input_dim is not None else len(features)
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.output_dim = output_dim

        # Skorch training parameters
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion

        # Skorch regularization & training
        self.train_split = train_split
        self.callbacks = callbacks if callbacks is not None else []
        self.warm_start = warm_start
        self.verbose = verbose

        # Skorch device & performance
        self.device = device

        # Skorch iterator settings
        self.iterator_train = iterator_train
        self.iterator_train__shuffle = iterator_train__shuffle
        self.iterator_train__num_workers = iterator_train__num_workers
        self.iterator_valid = iterator_valid
        self.iterator_valid__shuffle = iterator_valid__shuffle

        # Store additional kwargs for skorch
        self.kwargs = kwargs

        # Placeholder for the skorch estimator (created in fit)
        self.estimator_ = None

    def _create_estimator(self, y: np.ndarray):
        """
        Create the appropriate skorch estimator based on task type.

        Parameters
        ----------
        y : np.ndarray
            Target array used to infer output_dim for classification
        """
        # Determine output_dim if not specified
        output_dim = self.output_dim
        if output_dim is None:
            if self.task == "classification":
                output_dim = len(np.unique(y))
            else:  # regression
                output_dim = 1

        # Prepare module parameters
        module_params = {
            "module__input_dim": self.input_dim,
            "module__hidden_layers": self.hidden_layers,
            "module__output_dim": output_dim,
        }

        # Add any module-specific kwargs
        for key, value in self.kwargs.items():
            if key.startswith("module__"):
                module_params[key] = value

        # Common parameters for both regressor and classifier
        common_params = {
            "module": self.module,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "train_split": self.train_split,
            "callbacks": self.callbacks,
            "warm_start": self.warm_start,
            "verbose": self.verbose,
            "device": self.device,
            "iterator_train__shuffle": self.iterator_train__shuffle,
            "iterator_train__num_workers": self.iterator_train__num_workers,
            "iterator_valid__shuffle": self.iterator_valid__shuffle,
        }

        # Add optional parameters if specified
        if self.optimizer is not None:
            common_params["optimizer"] = self.optimizer
        if self.criterion is not None:
            common_params["criterion"] = self.criterion
        if self.iterator_train is not None:
            common_params["iterator_train"] = self.iterator_train
        if self.iterator_valid is not None:
            common_params["iterator_valid"] = self.iterator_valid

        # Add optimizer-specific kwargs (e.g., optimizer__weight_decay)
        for key, value in self.kwargs.items():
            if key.startswith("optimizer__") or key.startswith("criterion__"):
                common_params[key] = value

        # Merge module and common parameters
        all_params = {**common_params, **module_params}

        # Create the appropriate estimator
        _check_torch_available()
        NeuralNetRegressor, NeuralNetClassifier = _check_skorch_available()
        if self.task == "classification":
            self.estimator_ = NeuralNetClassifier(**all_params)
        elif self.task == "regression":
            self.estimator_ = NeuralNetRegressor(**all_params)
        else:
            raise ValueError(
                f"task must be 'regression' or 'classification', got '{self.task}'"
            )

    def fit(self, lf: pl.LazyFrame, **fit_kwargs):
        """
        Fit the wrapped PyTorch model using Polars LazyFrame.

        Automatically converts data to float32 as required by PyTorch and creates
        the skorch estimator on first call.

        Parameters
        ----------
        lf : pl.LazyFrame
            Training data as Polars LazyFrame
        **fit_kwargs
            Additional parameters passed to skorch's fit method

        Returns
        -------
        self
        """
        X = lf.select(self.features).collect().to_numpy().astype("float32")
        y = lf.select(self.target).collect().to_series().to_numpy()

        # Convert y to appropriate dtype
        if self.task == "regression":
            y = y.astype("float32")
        else:  # classification
            # Keep y as integers for classification
            y = y.astype("int64")

        # Create estimator on first fit
        if self.estimator_ is None:
            self._create_estimator(y)

        self.estimator_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, lf: pl.LazyFrame) -> np.ndarray:
        """
        Predict using the wrapped PyTorch model with Polars LazyFrame.

        Automatically converts input to float32 and flattens output for regressors.

        Parameters
        ----------
        lf : pl.LazyFrame
            Data to predict on as Polars LazyFrame

        Returns
        -------
        np.ndarray
            Predictions as 1D array
        """
        if self.estimator_ is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        X = lf.select(self.features).collect().to_numpy().astype("float32")
        preds = self.estimator_.predict(X)

        # Flatten predictions if they're 2D with single output column
        # This is common for regression models that output shape (n_samples, 1)
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.ravel()

        return preds

    def predict_proba(self, lf: pl.LazyFrame) -> np.ndarray:
        """
        Predict class probabilities using the wrapped PyTorch classifier.

        Only available for classification tasks. Automatically converts input to float32.

        Parameters
        ----------
        lf : pl.LazyFrame
            Data to predict on as Polars LazyFrame

        Returns
        -------
        np.ndarray
            Class probabilities with shape (n_samples, n_classes)

        Raises
        ------
        RuntimeError
            If model hasn't been fitted yet
        AttributeError
            If task is not 'classification'
        """
        if self.estimator_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba()")

        if self.task != "classification":
            raise AttributeError(
                "predict_proba is only available for classification tasks"
            )

        X = lf.select(self.features).collect().to_numpy().astype("float32")
        return self.estimator_.predict_proba(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Required for sklearn compatibility and hyperparameter tuning.
        """
        params = {
            "module": self.module,
            "features": self.features,
            "target": self.target,
            "task": self.task,
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "output_dim": self.output_dim,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "train_split": self.train_split,
            "callbacks": self.callbacks,
            "warm_start": self.warm_start,
            "verbose": self.verbose,
            "device": self.device,
            "iterator_train": self.iterator_train,
            "iterator_train__shuffle": self.iterator_train__shuffle,
            "iterator_train__num_workers": self.iterator_train__num_workers,
            "iterator_valid": self.iterator_valid,
            "iterator_valid__shuffle": self.iterator_valid__shuffle,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        """
        Set parameters for this estimator.

        Required for sklearn compatibility and hyperparameter tuning.
        """
        # Separate regular params from kwargs
        valid_params = set(self.get_params(deep=False).keys()) - {"kwargs"}

        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        # Reset estimator to force recreation with new parameters
        self.estimator_ = None
        return self

    def __repr__(self):
        """Return a string representation showing the module and key parameters."""
        module_name = (
            self.module.__name__
            if hasattr(self.module, "__name__")
            else str(self.module)
        )
        return (
            f"TorchWrapper(module={module_name}, task='{self.task}', "
            f"features={len(self.features)} features, "
            f"hidden_layers={self.hidden_layers}, max_epochs={self.max_epochs})"
        )
