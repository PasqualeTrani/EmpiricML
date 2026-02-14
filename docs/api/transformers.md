# empml.transformers

| Object | Description |
| :--- | :--- |
| `Identity` | Pass-through transformer. |
| `AvgFeatures` | Compute mean across multiple features row-wise. |
| `MaxFeatures` | Compute max across multiple features row-wise. |
| `MinFeatures` | Compute min across multiple features row-wise. |
| `StdFeatures` | Compute standard deviation across multiple features row-wise. |
| `MedianFeatures` | Compute median across multiple features row-wise. |
| `ModuleFeatures` | Compute Euclidean norm (module) of two features. |
| `InteractionFeatures` | Create pairwise multiplication features from feature pairs. |
| `MeanTargetEncoder` | Target encoding using mean of target variable. |
| `StdTargetEncoder` | Target encoding using standard deviation of target variable. |
| `MaxTargetEncoder` | Target encoding using max of target variable. |
| `MinTargetEncoder` | Target encoding using min of target variable. |
| `MedianTargetEncoder` | Target encoding using median of target variable. |
| `KurtTargetEncoder` | Target encoding using kurtosis of target variable. |
| `SkewTargetEncoder` | Target encoding using skewness of target variable. |
| `OrdinalEncoder` | Encode categorical features as ordinal integers. |
| `DummyEncoder` | One-hot encode categorical features. |
| `FrequencyEncoder` | Encode categorical features by their frequency or proportion. |
| `StandardScaler` | Standardize features by removing the mean and scaling to unit variance. |
| `MinMaxScaler` | Scale features to [0, 1] range using min-max normalization. |
| `RobustScaler` | Scale features using median and interquartile range (IQR). |
| `Log1pFeatures` | Apply log(1+x) transformation. |
| `Expm1Features` | Apply exp(x-1) transformation. |
| `PowerFeatures` | Apply power transformation. |
| `InverseFeatures` | Apply inverse transformation (1/x). |
| `QuantileBinning` | Discretize continuous features into quantile-based bins. |
| `RankFeatures` | Convert features to percentile rank based on training distribution. |
| `SimpleImputer` | Impute missing values using mean or median. |
| `FillNulls` | Fill null and NaN values with a constant. |
| `GenerateLags` | Generate lagged features for time series data. |
| `KMeansCluster` | Assign cluster labels using KMeans on selected features. |
| `PCATransformer` | Reduce dimensionality using Principal Component Analysis. |

## Identity
Pass-through transformer that returns data unchanged. No parameters required.

## AvgFeatures
Compute mean across multiple features row-wise.

### Methods

```python
def __init__(self, features: list[str], new_feature: str):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to average.
    new_feature : str
        Name of the output column.
    """
    pass
```

## MaxFeatures
Compute max across multiple features row-wise.

### Methods

```python
def __init__(self, features: list[str], new_feature: str):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to compute max over.
    new_feature : str
        Name of the output column.
    """
    pass
```

## MinFeatures
Compute min across multiple features row-wise.

### Methods

```python
def __init__(self, features: list[str], new_feature: str):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to compute min over.
    new_feature : str
        Name of the output column.
    """
    pass
```

## StdFeatures
Compute standard deviation across multiple features row-wise.

### Methods

```python
def __init__(self, features: list[str], new_feature: str):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to compute std over.
    new_feature : str
        Name of the output column.
    """
    pass
```

## MedianFeatures
Compute median across multiple features row-wise.

### Methods

```python
def __init__(self, features: list[str], new_feature: str):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to compute median over.
    new_feature : str
        Name of the output column.
    """
    pass
```

## ModuleFeatures
Compute Euclidean norm (module) of two features: `sqrt(f1^2 + f2^2)`.

### Methods

```python
def __init__(self, features: tuple[str, str], new_feature: str):
    """
    Parameters:
    -----------
    features : tuple[str, str]
        Tuple of two column names.
    new_feature : str
        Name of the output column.
    """
    pass
```

## InteractionFeatures
Create pairwise multiplication features from feature pairs. For each pair `(f1, f2)`, creates a column `f1 * f2`.

### Methods

```python
def __init__(self, feature_pairs: list[tuple[str, str]], separator: str = '_x_'):
    """
    Parameters:
    -----------
    feature_pairs : list[tuple[str, str]]
        List of (col1, col2) tuples to multiply.
    separator : str
        String between feature names in the output column name (default: '_x_').
        Output column name: '{col1}{separator}{col2}'.
    """
    pass
```

## MeanTargetEncoder
Encode categorical features with the mean of a target variable. Unseen categories during transform are filled with the global mean.

### Methods

```python
def __init__(
    self,
    features: list[str],
    encoder_col: str,
    prefix: str = 'mean_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    """
    Parameters:
    -----------
    features : list[str]
        Categorical columns to encode.
    encoder_col : str
        Target column to aggregate.
    prefix : str
        Prefix for encoded column names (default: 'mean_').
    suffix : str
        Suffix for encoded column names (default: '_encoded').
    replace_original : bool
        If True, drop original columns and use their names
        for encoded columns, ignoring prefix/suffix (default: False).
    """
    pass
```

## StdTargetEncoder
Encode categorical features with the standard deviation of a target variable. Same interface as `MeanTargetEncoder`.

### Methods

```python
def __init__(
    self,
    features: list[str],
    encoder_col: str,
    prefix: str = 'std_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    pass
```

## MaxTargetEncoder
Encode categorical features with the max of a target variable. Same interface as `MeanTargetEncoder`.

### Methods

```python
def __init__(
    self,
    features: list[str],
    encoder_col: str,
    prefix: str = 'max_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    pass
```

## MinTargetEncoder
Encode categorical features with the min of a target variable. Same interface as `MeanTargetEncoder`.

### Methods

```python
def __init__(
    self,
    features: list[str],
    encoder_col: str,
    prefix: str = 'min_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    pass
```

## MedianTargetEncoder
Encode categorical features with the median of a target variable. Same interface as `MeanTargetEncoder`.

### Methods

```python
def __init__(
    self,
    features: list[str],
    encoder_col: str,
    prefix: str = 'median_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    pass
```

## KurtTargetEncoder
Encode categorical features with the kurtosis of a target variable. Same interface as `MeanTargetEncoder`.

### Methods

```python
def __init__(
    self,
    features: list[str],
    encoder_col: str,
    prefix: str = 'kurt_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    pass
```

## SkewTargetEncoder
Encode categorical features with the skewness of a target variable. Same interface as `MeanTargetEncoder`.

### Methods

```python
def __init__(
    self,
    features: list[str],
    encoder_col: str,
    prefix: str = 'skew_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    pass
```

## OrdinalEncoder
Encode categorical features with ordinal integers based on sorted order. Null values are encoded as `-99`, unknown categories (not seen during fit) as `-9999`.

### Methods

```python
def __init__(
    self,
    features: list[str],
    suffix: str = '_ordinal_encoded',
    replace_original: bool = False
):
    """
    Parameters:
    -----------
    features : list[str]
        Categorical columns to encode.
    suffix : str
        Suffix for encoded column names (default: '_ordinal_encoded').
    replace_original : bool
        If True, drop original columns and use their names
        for encoded columns, ignoring suffix (default: False).
    """
    pass
```

## DummyEncoder
One-hot encode categorical features. Creates binary columns for each category, plus dedicated columns for null and unknown values.

### Methods

```python
def __init__(self, features: list[str]):
    """
    Parameters:
    -----------
    features : list[str]
        Categorical columns to encode.
    """
    pass
```

## FrequencyEncoder
Encode categorical features by their frequency (count) or proportion. Unseen categories during transform are filled with 0.

### Methods

```python
def __init__(
    self,
    features: list[str],
    normalize: bool = True,
    prefix: str = 'freq_',
    suffix: str = '_encoded',
    replace_original: bool = False
):
    """
    Parameters:
    -----------
    features : list[str]
        Categorical columns to encode.
    normalize : bool
        If True, encode as proportion (0-1); if False, as raw count (default: True).
    prefix : str
        Prefix for encoded column names (default: 'freq_').
    suffix : str
        Suffix for encoded column names (default: '_encoded').
    replace_original : bool
        If True, drop original columns and use their names
        for encoded columns, ignoring prefix/suffix (default: False).
    """
    pass
```

## StandardScaler
Standardize features by removing the mean and scaling to unit variance (z-score normalization). Handles zero standard deviation by returning 0.

### Methods

```python
def __init__(self, features: list[str], suffix: str = ''):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to standardize.
    suffix : str
        Suffix for scaled column names (default: '' overwrites original).
    """
    pass
```

## MinMaxScaler
Scale features to [0, 1] range using min-max normalization. Handles zero range by returning 0.

### Methods

```python
def __init__(self, features: list[str], suffix: str = ''):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to scale.
    suffix : str
        Suffix for scaled column names (default: '' overwrites original).
    """
    pass
```

## RobustScaler
Scale features using median and interquartile range (IQR): `(x - median) / (Q75 - Q25)`. Less sensitive to outliers than `StandardScaler`. Handles zero IQR by returning 0.

### Methods

```python
def __init__(self, features: list[str], suffix: str = ''):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to scale.
    suffix : str
        Suffix for scaled column names (default: '' overwrites original).
    """
    pass
```

## Log1pFeatures
Apply `log(1+x)` transformation to features.

### Methods

```python
def __init__(self, features: list[str], suffix: str = ''):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to transform.
    suffix : str
        Suffix for output column names (default: '' overwrites original).
    """
    pass
```

## Expm1Features
Apply `exp(x-1)` transformation to features.

### Methods

```python
def __init__(self, features: list[str], suffix: str = ''):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to transform.
    suffix : str
        Suffix for output column names (default: '' overwrites original).
    """
    pass
```

## PowerFeatures
Raise features to a specified power.

### Methods

```python
def __init__(self, features: list[str], suffix: str = '', power: float = 2):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to transform.
    suffix : str
        Suffix for output column names (default: '' overwrites original).
    power : float
        Exponent for power transformation (default: 2).
    """
    pass
```

## InverseFeatures
Apply inverse (`1/x`) transformation to features.

### Methods

```python
def __init__(self, features: list[str], suffix: str = ''):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to transform.
    suffix : str
        Suffix for output column names (default: '' overwrites original).
    """
    pass
```

## QuantileBinning
Discretize continuous features into quantile-based bins. During `fit`, computes quantile bin edges. During `transform`, assigns bin indices (0 to num_bins-1).

### Methods

```python
def __init__(
    self,
    features: list[str],
    num_bins: int = 10,
    suffix: str = '_qbin',
    labels: list[str] | None = None
):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to discretize.
    num_bins : int
        Number of quantile bins (default: 10).
    suffix : str
        Suffix for binned column names (default: '_qbin').
    labels : list[str] | None
        Optional string labels for bins; length must equal num_bins.

    Raises:
    -------
    ValueError
        If labels length does not match num_bins.
    """
    pass
```

## RankFeatures
Convert features to percentile rank [0, 1] based on the training distribution.

### Methods

```python
def __init__(
    self,
    features: list[str],
    suffix: str = '_rank',
    method: Literal['average', 'min', 'max', 'dense'] = 'average'
):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to rank.
    suffix : str
        Suffix for ranked column names (default: '_rank').
    method : str
        Ranking method (default: 'average'):
        - 'average': average of min and max rank positions
        - 'min': lowest rank position
        - 'max': highest rank position
        - 'dense': like 'min' but ranks always increase by 1
    """
    pass
```

## SimpleImputer
Impute missing values (null and NaN) using mean or median strategy.

### Methods

```python
def __init__(self, features: list[str], strategy: str = 'mean'):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to impute.
    strategy : str
        Imputation strategy: 'mean' or 'median' (default: 'mean').

    Raises:
    -------
    ValueError
        If strategy is not 'mean' or 'median'.
    """
    pass
```

## FillNulls
Fill null and NaN values with a constant.

### Methods

```python
def __init__(self, features: list[str], value: float = -9999):
    """
    Parameters:
    -----------
    features : list[str]
        Columns to fill.
    value : float
        Constant to use for filling (default: -9999).
    """
    pass
```

## GenerateLags
Generate lagged features for time series data by joining shifted dates.

### Methods

```python
def __init__(
    self,
    ts_index: str,
    date_col: str,
    lag_col: str,
    lag_frequency: str = 'days',
    lag_min: int = 1,
    lag_max: int = 1,
    lag_step: int = 1
):
    """
    Parameters:
    -----------
    ts_index : str
        Column for time series identifier (e.g., entity ID).
    date_col : str
        Date/time column (must be Polars Date or Datetime type).
    lag_col : str
        Column to lag.
    lag_frequency : str
        Time unit for lags (default: 'days').
        Options: 'weeks', 'days', 'hours', 'minutes', 'seconds',
        'milliseconds', 'microseconds', 'nanoseconds'.
    lag_min : int
        Minimum lag period (default: 1).
    lag_max : int
        Maximum lag period (default: 1).
    lag_step : int
        Step size between lags (default: 1).

    Raises:
    -------
    ValueError
        If lag_frequency is not a valid time unit.
    """
    pass
```

## KMeansCluster
Assign cluster labels using KMeans on selected features. Uses scikit-learn's KMeans internally. Missing values are imputed with column means before clustering.

### Methods

```python
def __init__(
    self,
    features: list[str],
    num_clusters: int = 8,
    new_feature: str = 'kmeans_cluster',
    random_state: int = 42
):
    """
    Parameters:
    -----------
    features : list[str]
        Numeric columns to use for clustering.
    num_clusters : int
        Number of clusters (default: 8).
    new_feature : str
        Name of the output cluster column (default: 'kmeans_cluster').
    random_state : int
        Random seed for reproducibility (default: 42).
    """
    pass
```

## PCATransformer
Reduce dimensionality using Principal Component Analysis. Uses scikit-learn's PCA internally. Missing values are imputed with column means before PCA.

### Methods

```python
def __init__(
    self,
    features: list[str],
    n_components: int = 2,
    prefix: str = 'pc_'
):
    """
    Parameters:
    -----------
    features : list[str]
        Numeric columns to use for PCA.
    n_components : int
        Number of principal components to keep (default: 2).
    prefix : str
        Prefix for output column names (default: 'pc_').
        Columns are named '{prefix}0', '{prefix}1', etc.
    """
    pass
```
