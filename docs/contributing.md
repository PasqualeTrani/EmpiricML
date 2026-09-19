# Contributing to EmpiricML

I appreciate your interest in contributing to EmpiricML! This guide will help you get started with reporting issues and extending the framework.

## 1. Reporting Issues

If you encounter any bugs, have feature requests, or want to suggest improvements, please use the GitHub Issues tracker.

### Bug Reports

When reporting a bug, please include:

- A clear and descriptive title.
- A detailed description of the issue.
- Steps to reproduce the bug (including code snippets if possible).
- The expected behavior vs. the actual behavior.
- Your environment details (OS, Python version, EmpiricML version).

### Feature Requests

For feature requests, please provide:

- A clear description of the proposed feature.
- The motivation behind the feature (why is it needed?).
- Potential implementation details (if you have ideas).

---

## 2. Extending Classes

EmpiricML is built to be modular and extensible. You can easily plug in new components by inheriting from the base classes defined in `empml.base`.

### Data Downloaders

If you need to load data from sources not currently supported (e.g., **S3**, **REST APIs**, **SharePoint**, etc.), you can create a custom downloader by extending the `DataDownloader` class.

EmpiricML already includes built-in downloaders for files (CSV, Parquet, Excel), SQL databases (PostgreSQL, MySQL, MSSQL, SQLite, Oracle), and cloud warehouses (Redshift, BigQuery, Snowflake, Databricks). See [`empml.data`](api/data.md) for the full list.

You need to implement the `get_data` method, which must return a `polars.LazyFrame`.

#### Example: Creating a REST API Downloader

```python
import polars as pl
from empml.base import DataDownloader

class RestAPIDownloader(DataDownloader):
    """
    Custom downloader that fetches data from a REST API endpoint.
    """
    def __init__(self, url: str, headers: dict | None = None):
        self.url = url
        self.headers = headers or {}

    def get_data(self) -> pl.LazyFrame:
        """
        Fetches JSON data from a REST API and returns a Polars LazyFrame.
        """
        import requests

        response = requests.get(self.url, headers=self.headers)
        response.raise_for_status()
        data = response.json()

        return pl.DataFrame(data).lazy()
```

### Transformers

 To add new data transformation logic (e.g., **Clustering**, dimensionality reduction, or custom feature engineering), you should extend the `BaseTransformer` class.

You must implement two methods:

1. `fit(self, lf: pl.LazyFrame)`: Learns parameters from the data (if stateful).
2. `transform(self, lf: pl.LazyFrame) -> pl.LazyFrame`: Applies the transformation.

#### Example: Creating a Custom Transformer

```python
import polars as pl
from empml.base import BaseTransformer

class MyTransformer(BaseTransformer):
    """
    A generic transformer template.
    Replace this with your custom transformation logic.
    """
    def __init__(self, param1: str, param2: int = 10):
        """
        Initialize your transformer with any parameters needed.
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2
        """
        self.param1 = param1
        self.param2 = param2
        # Add any attributes that will be learned during fit
        self.fitted_value_ = None

    def fit(self, lf: pl.LazyFrame):
        """
        Learn parameters from the data if needed.
        For stateless transformers, this can simply return self.
        
        Args:
            lf: Input Polars LazyFrame
            
        Returns:
            self
        """
        # Example: compute and store some statistic from the data
        # self.fitted_value_ = lf.select(pl.col(self.param1).mean()).collect().item()
        
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply the transformation to the data.
        
        Args:
            lf: Input Polars LazyFrame
            
        Returns:
            Transformed Polars LazyFrame
        """
        # Example: add a new column based on your transformation logic
        # return lf.with_columns(
        #     (pl.col(self.param1) * self.param2).alias("new_feature")
        # )
        
        return lf
```

By following these patterns, you can integrate virtually any custom logic into the EmpiricML pipeline.

**If you create a custom transformer or a custom downloader not included in the framework, please consider submitting a pull request to add it. Thank you!**

---

## 3. Running Tests

Install the development dependencies and run the checks from the repository root:

```bash
pip install -e ".[dev]"
pytest --cov=empml --cov-report=term-missing
ruff format --check src tests
ruff check src tests
```

The test suite leaves the repository clean. Each test runs in its own pytest temporary directory, so `Lab` artifact folders never appear in the project. The pytest, coverage, ruff, and mypy caches are all written to the git-ignored `.cache/` folder.

The Lab tests in `tests/test_lab_characterization.py` compare observable results with golden snapshots in `tests/fixtures/`. If you change Lab behavior on purpose, regenerate them with `EMPML_UPDATE_GOLDEN=1 pytest tests/test_lab_characterization.py` and review the fixture diff. `tests/fixtures/legacy_lab_checkpoint.pkl` is never regenerated, because it proves that checkpoints saved by older releases still load.

---

## 4. Code Layout

`Lab` is a thin coordinator. Each responsibility lives in its own module:

| Module | Responsibility |
| :--- | :--- |
| `empml.lab` | `Lab`, `ComparisonCriteria`, `restore_check_point`: the experiment workflow |
| `empml.pipeline` | `Pipeline` and fold-by-fold evaluation with early stopping |
| `empml.results` | Results table schemas and row formatting; `MetricColumns` names per-metric columns |
| `empml.comparison` | Experiment comparison statistics, the improvement rule, permutation p-values, and printed reports |
| `empml.hpo` | Hyperparameter search space, HPO pipelines, and best-result selection |
| `empml.baselines` | The baseline model catalog used by `run_base_experiments` |
| `empml.feature_selection` | Permutation feature importance |
| `empml.target` | Target transformations and `TransformedTargetRegressor` |
| `empml.artifacts` | Where a Lab stores pipelines, predictions, and checkpoints on disk |
| `empml.lab_utils` | Row-ID and prediction-storage helpers, plus compatibility aliases |

Single-metric and multi-metric Labs share one code path. A single metric behaves as a list of one metric whose columns keep their unsuffixed names (`cv_mean_score` instead of `cv_mean_score_1`).

`Lab` instances are pickled into checkpoints, and transformers are pickled inside pipelines. Keep their instance attribute names stable, and derive helpers from existing attributes instead of storing new ones.
