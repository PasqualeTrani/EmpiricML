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

If you need to load data from sources not currently supported (e.g., **S3**, **PostgreSQL**, **Amazon Redshift**, **SharePoint**, etc.), you can create a custom downloader by extending the `DataDownloader` class.

You need to implement the `get_data` method, which must return a `polars.LazyFrame`.

#### Example: Creating a PostgreSQL Downloader

```python
import polars as pl
from empml.base import DataDownloader

class PostgresDownloader(DataDownloader):
    """
    Custom downloader for PostgreSQL databases.
    """
    def __init__(self, connection_uri: str, query: str):
        self.connection_uri = connection_uri
        self.query = query

    def get_data(self) -> pl.LazyFrame:
        """
        Reads data from PostgreSQL and returns a Polars LazyFrame.
        """
        return pl.read_database_uri(
            query=self.query,
            uri=self.connection_uri
        ).lazy()
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