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

## 9. Multi-Metric Lab

Evaluate models on multiple metrics simultaneously. The Lab tracks all metrics and requires improvement on **all** of them for a model to be considered better.

```python
from empml.lab import Lab, ComparisonCriteria
from empml.data import ParquetDownloader
from empml.cv import KFold
from empml.metrics import RMSE, MAE

# Initialize Lab with multiple metrics
lab = Lab(
    train_downloader=ParquetDownloader('./data/train.parquet'),
    metric=[RMSE(), MAE()],
    cv_generator=KFold(n_splits=5, random_state=7),
    target='target_column',
    comparison_criteria=ComparisonCriteria(
        n_folds_threshold=2,
        pct_threshold=0.01
    ),
    minimize=[True, True],  # both RMSE and MAE should be minimized
    name='multi_metric_lab'
)

# Run experiments as usual - results will contain suffixed columns
# (cv_mean_score_1 for RMSE, cv_mean_score_2 for MAE)
lab.run_experiment(pipeline, auto_mode=True)

# View best score for each metric
lab.show_best_score(metric_idx=0)  # best by RMSE
lab.show_best_score(metric_idx=1)  # best by MAE

# HPO with multi-metric: optimize based on a specific metric or all
best_result = lab.hpo(
    features=['feature1', 'feature2', 'feature3'],
    params_list={'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    estimator=LGBMRegressor,
    primary_metric_idx='all'  # or 0 for RMSE only, 1 for MAE only
)
```

## 10. Loading Data from SQL Databases

EmpiricML provides built-in downloaders for SQL databases and cloud data warehouses.

```python
from empml.data import PostgreSQLDownloader, SQLiteDownloader, SQLDownloader

# PostgreSQL
pg_downloader = PostgreSQLDownloader(
    query='SELECT * FROM my_table',
    host='localhost',
    user='my_user',
    password='my_password',
    database='my_database',
    port=5432
)

# SQLite
sqlite_downloader = SQLiteDownloader(
    query='SELECT * FROM my_table',
    path='/path/to/database.db'
)

# Generic SQL with connection URI
sql_downloader = SQLDownloader(
    query='SELECT * FROM my_table',
    connection_uri='postgresql://user:password@host:5432/database'
)

# Use any downloader as train or test data source
lab = Lab(
    train_downloader=pg_downloader,
    metric=RMSE(),
    cv_generator=KFold(n_splits=5),
    target='target_column',
    comparison_criteria=ComparisonCriteria(n_folds_threshold=1, pct_threshold=0.01),
    minimize=True
)
```

## 11. Recursive Feature Selection

Automatically identify and remove features that hurt model performance using permutation-based importance.

```python
from lightgbm import LGBMRegressor

# Recursively remove features with negative importance
selected_features = lab.recursive_permutation_feature_selection(
    estimator=LGBMRegressor(verbose=-1),
    features=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
    n_iters=5,
    verbose=True
)

# selected_features contains only the features that contribute positively
print(f"Selected features: {selected_features}")
```

## 12. Using Transformers in Pipelines

EmpiricML provides a rich set of transformers for feature engineering.

### Feature Interactions

```python
from empml.transformers import InteractionFeatures

# Create pairwise multiplication features
interactions = InteractionFeatures(
    feature_pairs=[('feature1', 'feature2'), ('feature2', 'feature3')],
    separator='_x_'
)
# Creates columns: feature1_x_feature2, feature2_x_feature3
```

### Frequency Encoding

```python
from empml.transformers import FrequencyEncoder

# Encode categories by their frequency
freq_encoder = FrequencyEncoder(
    features=['category_col'],
    normalize=True,         # proportion instead of raw count
    replace_original=False  # keep original column
)
```

### Robust Scaling

```python
from empml.transformers import RobustScaler

# Scale features using median/IQR (outlier-resistant)
scaler = RobustScaler(features=['feature1', 'feature2'])
```

### Quantile Binning

```python
from empml.transformers import QuantileBinning

# Discretize into quantile-based bins
binner = QuantileBinning(
    features=['continuous_feature'],
    num_bins=5,
    labels=['very_low', 'low', 'medium', 'high', 'very_high']
)
```

### Clustering and Dimensionality Reduction

```python
from empml.transformers import KMeansCluster, PCATransformer

# Add cluster labels as a new feature
kmeans = KMeansCluster(
    features=['feature1', 'feature2', 'feature3'],
    num_clusters=5,
    new_feature='cluster_label'
)

# Reduce dimensions with PCA
pca = PCATransformer(
    features=['feature1', 'feature2', 'feature3', 'feature4'],
    n_components=2,
    prefix='pc_'
)
```

### Combining Transformers in a Pipeline

```python
from empml.pipeline import Pipeline
from empml.transformers import StandardScaler, InteractionFeatures, PCATransformer
from empml.wrappers import SKlearnWrapper
from lightgbm import LGBMRegressor

# Build a feature engineering + model pipeline
pipeline = Pipeline([
    ('interactions', InteractionFeatures(
        feature_pairs=[('f1', 'f2'), ('f2', 'f3')]
    )),
    ('scaler', StandardScaler(features=['f1', 'f2', 'f3'])),
    ('pca', PCATransformer(features=['f1', 'f2', 'f3'], n_components=2)),
    ('model', SKlearnWrapper(
        estimator=LGBMRegressor(verbose=-1),
        features=['f1', 'f2', 'f3', 'f1_x_f2', 'f2_x_f3', 'pc_0', 'pc_1'],
        target='target'
    ))
], name='Advanced Pipeline')

lab.run_experiment(pipeline)
```
