# Examples

This section demonstrates the core functionalities of the `Lab` class for managing machine learning experiments.

## 1. Initializing Lab

The `Lab` class is the central orchestrator. It requires a data source, an evaluation metric, a cross-validation strategy, and comparison criteria.

```python
from empml.lab import Lab, ComparisonCriteria
from empml.data import ParquetDownloader
from empml.cv import KFold
from empml.metrics import RMSE

# Initialize the Lab environment
lab = Lab(
    train_downloader=ParquetDownloader('./data/train.parquet'),
    metric=RMSE(),
    cv_generator=KFold(n_splits=5, random_state=7),
    target='target_column',
    comparison_criteria=ComparisonCriteria(
        n_folds_threshold=2, # Experiment considered better if it improves >2 folds
        pct_threshold=0.01   # and improves metric by at least 1%
    ),
    minimize=True,  # True for errors (RMSE), False for scores (Accuracy)
    row_id='row_id' # Unique identifier for each row
)
```

## 2. Running a Pipeline Experiment

Create a pipeline typically involving transformers and a model wrapper, then run it.

```python
from lightgbm import LGBMRegressor
from empml.pipeline import Pipeline
from empml.wrappers import SKlearnWrapper
from empml.transformers import Log1pFeatures

# Define a pipeline with feature engineering and model
pipeline = Pipeline([
    ('log_features', Log1pFeatures(features=['feature1', 'feature2'])),
    ('model', SKlearnWrapper(
        estimator=LGBMRegressor(verbose=-1),
        features=['feature1', 'feature2', 'feature3'],
        target='target_column'
    ))
], name='LGBM_Experiment', description='LGBM with Log1p transformation')

# Run the experiment
lab.run_experiment(pipeline)
```

## 3. Running Base Experiments

Run a suite of default baseline models (Linear Regression, KNN, Random Forest, etc.) to establish performance benchmarks.

```python
# Run a suite of baseline models
lab.run_base_experiments(
    features=['feature1', 'feature2', 'feature3'],
    problem_type='regression' # or 'classification'
)
```

## 4. Setting the Best Experiment

Mark a specific experiment as the current "best" to compare future experiments against.

```python
# Manually set the best experiment ID (e.g., ID 1)
lab._set_best_experiment(experiment_id=1)
```

## 5. Comparing Against Best Experiment

Run a new experiment and automatically compare it against the set baseline.

```python
# Define another pipeline with different parameters
pipeline_v2 = Pipeline([
    ('model', SKlearnWrapper(
        estimator=LGBMRegressor(n_estimators=200, verbose=-1),
        features=['feature1', 'feature2', 'feature3'],
        target='target_column'
    ))
], name='LGBM_v2')

# Run and compare against the best experiment (ID 1)
lab.run_experiment(pipeline_v2, compare_against=1)

# Alternatively, use auto_mode to automatically compare against the current best experiment
# and update the best_experiment attribute if the new pipeline performs better according to the comparison criteria
lab._set_best_experiment(1)
lab.run_experiment(pipeline_v2, auto_mode=True)
```

## 6. Hyperparameter Optimization (HPO)

Perform grid or random search over a managed search space.

```python
# Define hyperparameter search space
params_list = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 64]
}

# Launch optimization
best_result = lab.hpo(
    features=['feature1', 'feature2', 'feature3'],
    params_list=params_list,
    estimator=LGBMRegressor,
    search_type='random',
    num_samples=10,
    store_preds=False
)
```

## 7. Permutation Feature Importance

Analyze which features contribute most to the model's performance.

```python
# Retrieve the best pipeline
pipeline = lab.retrieve_pipeline(experiment_id=1)

# Calculate feature importance
pfi : pl.DataFrame= lab.permutation_feature_importance(
    pipeline=pipeline,
    features=['feature1', 'feature2', 'feature3'],
    n_iters=5
)

pfi
```

## 8. Retrieving Predictions & Error Analysis

Load out-of-fold predictions to analyze where the model fails.

```python
import polars as pl

# Retrieve predictions from specific experiments
preds = lab.retrieve_predictions(
    experiment_ids=[1], # list of experiment IDs
    extra_features=['date'] # Optional: add extra columns from training data
)
```
