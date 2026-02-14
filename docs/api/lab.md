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

**Note:** You must provide either `pct_threshold` OR both `alpha` and `n_iters`. Providing both approaches raises a `ValueError`.

## Lab
Experimentation framework for ML model development and evaluation.
Manages experiment lifecycle: data loading, CV splitting, pipeline execution, results tracking, and statistical comparison. Supports HPO, feature selection, and multi-metric evaluation.

### Initialization

```python
def __init__(
    self,
    train_downloader: DataDownloader,
    metric: Metric | List[Metric],
    cv_generator: CVGenerator,
    target: str,
    comparison_criteria : ComparisonCriteria,
    minimize: bool | List[bool] = True,
    row_id: str | None = None,
    test_downloader: DataDownloader | None = None,
    name: str | None = None
)
```

**Parameters**

*   `train_downloader`: Source for training data.
*   `metric`: Single performance metric or list of metrics for evaluation. When a list is provided, the Lab operates in **multi-metric mode** (see below).
*   `cv_generator`: Cross-validation splitting strategy.
*   `target`: Name of target column.
*   `comparison_criteria`: Statistical criteria for experiment comparison.
*   `minimize`: Whether to minimize metric(s) (default True). When `metric` is a list, this can be a list of bools matching the length of `metric`.
*   `row_id`: Column name for row identifier.
*   `test_downloader`: Optional test data source.
*   `name`: Lab identifier (auto-generated if None).

### Multi-Metric Support

When a list of metrics is passed to `metric`, the Lab enters multi-metric mode:

* **Suffixed columns**: Results columns are suffixed with `_1`, `_2`, etc. (e.g., `cv_mean_score_1`, `cv_mean_score_2`).
* **Comparison logic**: When comparing experiments, the new model is considered better **only if ALL metrics** pass the comparison criteria.
* **Primary metric**: The first metric in the list is used as the primary metric by default for operations like `show_best_score`.
* **`minimize` list**: Each metric can independently be set to minimize or maximize (e.g., `minimize=[True, False]` for RMSE + Accuracy).

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
    random_state : int = 0,
    primary_metric_idx : int | str = 'all'       # for multi-metric: which metric to optimize (0-indexed or 'all')
)
```

**Multi-Metric HPO**: The `primary_metric_idx` parameter controls best-model selection:

* `'all'` (default): The best model must improve on **all** metrics simultaneously.
* `0`, `1`, etc.: Select the best model based on a single metric (0-indexed).

#### `retrieve_predictions`
Load predictions from specified experiments.

```python
def retrieve_predictions(self, experiment_ids = List[int], extra_features : List[str] = []) -> pl.LazyFrame
```

#### `compute_pvalue`
Compute permutation test p-value(s) comparing two experiments.

```python
def compute_pvalue(
    self,
    experiment_ids : Tuple[int, int],
    n_iters : int = 200,
    extra_features: List[str] = []
) -> Union[float, List[float]]
```

Returns a single `float` for single-metric Labs, or a `List[float]` (one p-value per metric) for multi-metric Labs.

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

**Note:** For multi-metric Labs, this method uses the first metric only.

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
def show_best_score(self, metric_idx: int | None = None) -> pl.DataFrame
```

**Parameters**

* `metric_idx`: For multi-metric Labs, which metric to sort by (0-indexed). Defaults to the first metric.

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

# 2. Initialize Lab (single metric)
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

### Multi-Metric Example

```python
from empml.lab import Lab, ComparisonCriteria
from empml.metrics import RMSE, MAE

# Initialize Lab with multiple metrics
lab = Lab(
    train_downloader=my_train_downloader_object,
    metric=[RMSE(), MAE()],
    cv_generator=my_cv_object,
    target='target_variable',
    comparison_criteria=ComparisonCriteria(n_folds_threshold=1, pct_threshold=0.05),
    minimize=[True, True],  # both metrics should be minimized
    name='multi_metric_lab'
)

# Run experiments - comparison checks ALL metrics
lab.run_experiment(pipeline, auto_mode=True)

# View best score for a specific metric
lab.show_best_score(metric_idx=0)  # best by RMSE
lab.show_best_score(metric_idx=1)  # best by MAE
```

## Helper Functions

### `restore_check_point`
Load saved lab state from checkpoint.

```python
def restore_check_point(lab_name : str, check_point_name : str) -> Lab
```
