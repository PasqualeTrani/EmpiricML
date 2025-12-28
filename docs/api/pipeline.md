# empml.pipeline

| Object | Description |
| :--- | :--- |
| `Pipeline` | Custom pipeline for chaining transformers and an optional final estimator. |

## Pipeline
Custom pipeline for chaining transformers and an optional final estimator.

### Methods

```python
def __init__(self, steps: list[tuple[str, Union[BaseTransformer, BaseEstimator, 'Pipeline']]], name : str = '', description : str = ''):
    """
    Parameters:
    -----------
    steps : list of tuples
        List of (name, transformer/estimator/pipeline) tuples in the order they should be applied.
        If the last step is an estimator, the pipeline will support predict().
        If all steps are transformers (or pipelines acting as transformers), the pipeline 
        will support transform().
    """
    pass

def fit(self, lf: pl.LazyFrame, **fit_params):
    """
    Fit all transformers and the final estimator (if present).
    
    Parameters:
    -----------
    lf : pl.LazyFrame
        Training data
    **fit_params : dict
        Parameters to pass to the final estimator's fit method
    """
    pass

def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply all transformers sequentially.
    Only available for transformer-only pipelines.
    
    Parameters:
    -----------
    lf : pl.LazyFrame
        Data to transform
        
    Returns:
    --------
    pl.LazyFrame
        Transformed data
    """
    if not self._is_transformer_only:
        raise ValueError(
            "transform() is only available for transformer-only pipelines. "
            "This pipeline has an estimator as the final step. Use predict() instead."
        )
    
    lf_transformed = lf
    for name, step in self.steps:
        if isinstance(step, Pipeline):
            lf_transformed = step.transform(lf_transformed)
        else:
            lf_transformed = step.transform(lf_transformed)
    
    return lf_transformed

def fit_transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Fit and transform in one step.
    Only available for transformer-only pipelines.
    """
    pass

def predict(self, lf: pl.LazyFrame) -> np.ndarray:
    """
    Apply all transformers and predict with the final estimator.
    Only available for pipelines with an estimator as the final step.
    
    Parameters:
    -----------
    lf : pl.LazyFrame
        Data to predict on
        
    Returns:
    --------
    np.ndarray
        Predictions
    """
    pass

def fit_predict(self, lf: pl.LazyFrame, **fit_params) -> np.ndarray:
    """
    Fit the pipeline and return predictions on the same data.
    Only available for pipelines with an estimator.
    """
    pass
```

### Example Usage (Transformer-only)

```python
from empml.pipeline import Pipeline
from empml.transformers import AvgFeatures, MaxFeatures
import polars as pl

# Create a sample LazyFrame
data = pl.DataFrame({
    "f1": [1, 2, 3],
    "f2": [4, 5, 6],
    "target": [0, 1, 0]
}).lazy()

# Define a pipeline with only transformers
preprocessing_pipeline = Pipeline(steps=[
    ('avg_f1_f2', AvgFeatures(features=['f1', 'f2'], new_feature='avg_f1_f2')),
    ('max_f1_f2', MaxFeatures(features=['f1', 'f2'], new_feature='max_f1_f2'))
], name='Preprocessing', description='Feature Engineering Pipeline')

# transform data
processed_data = preprocessing_pipeline.fit_transform(data)

# Collect results
processed_data.collect()
```
