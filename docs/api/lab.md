# empml.lab

| Object | Description |
| :--- | :--- |
| `ComparisonCriteria` | Statistical criteria for comparing experiment performance. |
| `Lab` | Experimentation framework for ML model development and evaluation. |

## ComparisonCriteria
Statistical criteria for comparing experiment performance.

### Attributes

* `n_folds_threshold`: Integer. Number of folds where the new model is allowed to perform worse than the baseline.
* `pct_threshold`: Float (optional). Minimum percentage improvement required to consider a new model better.
* `alpha`: Float (optional). Significance level for statistical testing.
* `n_iters`: Integer (optional). Number of iterations for permutation testing.

## Lab
Experimentation framework for ML model development and evaluation.
Manages experiment lifecycle: data loading, CV splitting, pipeline execution, results tracking, and statistical comparison. Supports HPO and feature selection.

### Initialization

```python
def __init__(
    self,
    train_downloader: DataDownloader,
    metric: Metric,
    cv_generator: CVGenerator,
    target: str,
    comparison_criteria : ComparisonCriteria,
    minimize: bool = True,
    row_id: str | None = None,
    test_downloader: DataDownloader | None = None,
    name: str | None = None
)
```

**Parameters**

*   `train_downloader`: Source for training data.
*   `metric`: Performance metric for evaluation.
*   `cv_generator`: Cross-validation splitting strategy.
*   `target`: Name of target column.
*   `comparison_criteria`: Statistical criteria for experiment comparison.
*   `minimize`: Whether to minimize metric (default True).
*   `row_id`: Column name for row identifier.
*   `test_downloader`: Optional test data source.
*   `name`: Lab identifier (auto-generated if None).

### Methods

#### `run_experiment`
Execute pipeline evaluation with CV and track results.

```python
def run_experiment(
    self,
    pipeline: Pipeline,
    eval_overfitting : bool = True,      # if True, overfitting will be evaluated
    store_preds : bool = True,           # if True, predictions will be stored
    verbose : bool = True,               # if True, progress will be printed
    compare_against: int | None = None,  # id of the experiment to compare against
    auto_mode : bool = False             # if True, the best experiment will be automatically updated if all criteria are met
)
```

#### `multi_run_experiment`
Execute multiple experiments sequentially.

```python
def multi_run_experiment(
    self,
    pipelines: List[Pipeline],
    eval_overfitting : bool = True,      # if True, overfitting will be evaluated
    store_preds : bool = True,           # if True, predictions will be stored
    verbose : bool = True,               # if True, progress will be printed
    compare_against: int | None = None,  # id of the experiment to compare against
    auto_mode : bool = False             # if True, the best experiment will be automatically updated if all criteria are met
)
```

#### `run_base_experiments`
Run suite of baseline models for quick benchmarking.

```python
def run_base_experiments(
    self, 
    features: str, 
    preprocess_pipe : Pipeline | None = None,  # transformer-only pipeline to apply to features
    eval_overfitting: bool = True,             # if True, overfitting will be evaluated
    store_preds: bool = True,                   # if True, predictions will be stored
    verbose: bool = True,                       # if True, progress will be printed
    compare_against: int | None = None,         # id of the experiment to compare against
    problem_type: str = 'regression'            # 'regression' or 'classification'
)
```

#### `hpo`
Hyperparameter optimization via grid or random search.

```python
def hpo(
    self, 
    features : List[str], 
    params_list : Dict[str, List[float | int | str]], 
    estimator : SKlearnEstimator, 
    preprocessor : Pipeline | BaseTransformer = Identity(),  # transformer-only pipeline to apply to features
    eval_overfitting : bool = True,             # if True, overfitting will be evaluated
    store_preds : bool = True,                   # if True, predictions will be stored
    verbose : bool = True,                       # if True, progress will be printed
    compare_against: int | None = None,         # id of the experiment to compare against
    search_type : str = 'grid',                  # 'grid' or 'random'
    num_samples : int = 64,                      # number of samples for random search
    random_state : int = 0
)
```

#### `retrieve_predictions`
Load predictions from specified experiments.

```python
def retrieve_predictions(self, experiment_ids = List[int], extra_features : List[str] = []) -> pl.LazyFrame
```

#### `compute_pvalue`
Compute permutation test p-value comparing two experiments.

```python
def compute_pvalue(self, experiment_ids : Tuple[int, int], n_iters : int = 200, extra_features: List[str] = []) -> float
```

#### `permutation_feature_importance`
Compute permutation feature importance for each feature.

```python
def permutation_feature_importance(
    self, 
    pipeline : Pipeline, 
    features : List[str],
    n_iters : int = 5, 
    verbose : bool = True      
) -> pl.DataFrame
```

#### `recursive_permutation_feature_selection`
Recursively eliminate features with negative importance.

```python
def recursive_permutation_feature_selection(
    self, 
    estimator : SKlearnEstimator, 
    features : List[str], 
    preprocessor : Pipeline | BaseTransformer = Identity(), 
    n_iters : int = 5, 
    verbose : bool = True
) -> List[str]
```

#### `run_experiment_on_test`
Compute performance metrics of a pipeline associated with an experiment on the test set.

```python
def run_experiment_on_test(
    self, 
    experiment_id : int, 
    eval_overfitting : bool = True, 
    store_preds : bool = True, 
    verbose : bool = True
) -> Dict[str, Union[float, List[float]]]
```

#### `retrieve_pipeline`
Retrieve a pipeline related to an experiment.

```python
def retrieve_pipeline(self, experiment_id : int) -> Pipeline
```

#### `show_best_score`
Show the stats related to the experiment with the best cv_mean_score.

```python
def show_best_score(self) -> pl.DataFrame
```

#### `save_check_point`
Serialize current lab state to disk.

```python
def save_check_point(self, check_point_name : str | None = None) -> None
```

### Example Usage

```python
from empml.lab import Lab, ComparisonCriteria
# Assuming necessary components (downloader, metric, cv) are imported/defined

# 1. Define Criteria
criteria = ComparisonCriteria(
    n_folds_threshold=1,
    pct_threshold=0.05
)

# 2. Initialize Lab
lab = Lab(
    train_downloader=my_train_downloader_object,
    metric=my_metric_object,
    cv_generator=my_cv_object,
    target='target_variable',
    comparison_criteria=criteria,
    minimize=True,
    name='experiment_lab_01'
)

# 3. Run Base Experiments
lab.run_base_experiments(
    features=['feature_A', 'feature_B'],
    problem_type='classification'
)

# 4. View Best Results
lab.show_best_score()
```

## Helper Functions

### `restore_check_point`
Load saved lab state from checkpoint.

```python
def restore_check_point(lab_name : str, check_point_name : str) -> Lab
```
