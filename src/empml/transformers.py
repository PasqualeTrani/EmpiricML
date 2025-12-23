"""
Feature engineering transformers for machine learning pipelines.

Provides transformers for feature operations, encoding, scaling, imputation,
and time series lag generation, all compatible with Polars LazyFrame.
"""

# base imports 
from typing import Union, Literal, List, Dict, Tuple

# data wranglers 
import polars as pl 
import numpy as np

# internal imports 
from empml.base import BaseTransformer

# streaming engine as the default for .collect()
pl.Config.set_engine_affinity(engine='streaming')

# ------------------------------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------------------------------

class Identity(BaseTransformer):
    """Pass-through transformer that returns data unchanged."""
    
    def fit(self, X: pl.LazyFrame):
        """No-op fit method."""
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Return input unchanged."""
        return X

# ------------------------------------------------------------------------------------------
# Algebric operation between features
# ------------------------------------------------------------------------------------------

class AvgFeatures(BaseTransformer):
    """Compute mean across multiple features row-wise."""
    
    def __init__(self, features: List[str], new_feature: str):
        """
        Args:
            features: Columns to average
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        return X.with_columns(pl.mean_horizontal(self.features).alias(self.new_feature))
    
    
class MaxFeatures(BaseTransformer):
    """Compute max across multiple features row-wise."""
    
    def __init__(self, features: List[str], new_feature: str):
        """
        Args:
            features: Columns to compute max over
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        return X.with_columns(pl.max_horizontal(self.features).alias(self.new_feature))
    

class MinFeatures(BaseTransformer):
    """Compute min across multiple features row-wise."""
    
    def __init__(self, features: List[str], new_feature: str):
        """
        Args:
            features: Columns to compute min over
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        return X.with_columns(pl.min_horizontal(self.features).alias(self.new_feature))
    

class StdFeatures(BaseTransformer):
    """Compute standard deviation across multiple features row-wise."""
    
    def __init__(self, features: List[str], new_feature: str):
        """
        Args:
            features: Columns to compute std over
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        # Collect and convert to pandas for row-wise std calculation
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().std(1)).alias(self.new_feature))
    

class MedianFeatures(BaseTransformer):
    """Compute median across multiple features row-wise."""
    
    def __init__(self, features: List[str], new_feature: str):
        """
        Args:
            features: Columns to compute median over
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        # Collect and convert to pandas for row-wise median calculation
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().median(1)).alias(self.new_feature))
    

class KurtFeatures(BaseTransformer):
    """Compute kurtosis across multiple features row-wise."""
    
    def __init__(self, features: List[str], new_feature: str):
        """
        Args:
            features: Columns to compute kurtosis over
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        # Collect and convert to pandas for row-wise kurtosis calculation
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().kurt(1)).alias(self.new_feature))
    

class SkewFeatures(BaseTransformer):
    """Compute skewness across multiple features row-wise."""
    
    def __init__(self, features: List[str], new_feature: str):
        """
        Args:
            features: Columns to compute skewness over
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        # Collect and convert to pandas for row-wise skewness calculation
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().skew(1)).alias(self.new_feature))
    

class ModuleFeatures(BaseTransformer):
    """Compute Euclidean norm (module) of two features."""
    
    def __init__(self, features: Tuple[str, str], new_feature: str):
        """
        Args:
            features: Tuple of two column names
            new_feature: Name of output column
        """
        self.features = features
        self.new_feature = new_feature

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        f1, f2 = self.features  # Unpack the two features
        # Compute sqrt(f1^2 + f2^2)
        return X.with_columns(((pl.col(f1)**2) + (pl.col(f2)**2)).sqrt().alias(self.new_feature))
    

# ------------------------------------------------------------------------------------------
# Categorical Encoding 
# ------------------------------------------------------------------------------------------


# -------------------- TARGET ENCODING -------------------------------- #
class MeanTargetEncoder(BaseTransformer):
    """Encode categorical features with mean of target variable."""
    
    def __init__(self, features: List[str], encoder_col: str):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
        """
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X: pl.LazyFrame):
        """Compute mean target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {
            f: X.group_by(f).agg(pl.col(self.encoder_col).mean().alias(f'mean_{f}_target_encoded')) 
            for f in self.features
        }
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        return transf_X


class StdTargetEncoder(BaseTransformer):
    """Encode categorical features with std of target variable."""
    
    def __init__(self, features: List[str], encoder_col: str):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
        """
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X: pl.LazyFrame):
        """Compute std of target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {
            f: X.group_by(f).agg(pl.col(self.encoder_col).std().alias(f'std_{f}_target_encoded')) 
            for f in self.features
        }
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        return transf_X


class MaxTargetEncoder(BaseTransformer):
    """Encode categorical features with max of target variable."""
    
    def __init__(self, features: List[str], encoder_col: str):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
        """
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X: pl.LazyFrame):
        """Compute max target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {
            f: X.group_by(f).agg(pl.col(self.encoder_col).max().alias(f'max_{f}_target_encoded')) 
            for f in self.features
        }
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        return transf_X


class MinTargetEncoder(BaseTransformer):
    """Encode categorical features with min of target variable."""
    
    def __init__(self, features: List[str], encoder_col: str):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
        """
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X: pl.LazyFrame):
        """Compute min target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {
            f: X.group_by(f).agg(pl.col(self.encoder_col).min().alias(f'min_{f}_target_encoded')) 
            for f in self.features
        }
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        return transf_X


class MedianTargetEncoder(BaseTransformer):
    """Encode categorical features with median of target variable."""
    
    def __init__(self, features: List[str], encoder_col: str):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
        """
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X: pl.LazyFrame):
        """Compute median target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {
            f: X.group_by(f).agg(pl.col(self.encoder_col).median().alias(f'median_{f}_target_encoded')) 
            for f in self.features
        }
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        return transf_X


class KurtTargetEncoder(BaseTransformer):
    """Encode categorical features with kurtosis of target variable."""
    
    def __init__(self, features: List[str], encoder_col: str):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
        """
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X: pl.LazyFrame):
        """Compute kurtosis of target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {
            f: X.group_by(f).agg(pl.col(self.encoder_col).kurtosis().alias(f'kurt_{f}_target_encoded')) 
            for f in self.features
        }
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        return transf_X


class SkewTargetEncoder(BaseTransformer):
    """Encode categorical features with skewness of target variable."""
    
    def __init__(self, features: List[str], encoder_col: str):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
        """
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X: pl.LazyFrame):
        """Compute skewness of target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {
            f: X.group_by(f).agg(pl.col(self.encoder_col).skew().alias(f'skew_{f}_target_encoded')) 
            for f in self.features
        }
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        return transf_X


# -------------------- ORDINAL ENCODING -------------------------------- #

class OrdinalEncoder(BaseTransformer):
    """Encode categorical features with ordinal integers based on sorted order."""
    
    def __init__(self, features: List[str]):
        """
        Args:
            features: Categorical columns to encode
        """
        self.features = features

    def fit(self, X: pl.LazyFrame):
        """Learn ordinal mapping from sorted unique values."""
        self.encoding_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            # Get sorted unique values and assign sequential integers
            unique_values = (
                X.select(f)
                .filter(pl.col(f).is_not_null())
                .unique()
                .sort(f)
                .with_row_index(name=f'{f}_ordinal_encoded')
            )
            self.encoding_dict[f] = unique_values
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply ordinal encoding with special values for null (-99) and unknown (-9999)."""
        transf_X = X.clone()
        
        for f in self.features:
            # Join encoding dictionary
            transf_X = transf_X.join(self.encoding_dict[f], how='left', on=f)
            
            # Handle nulls and unknown categories
            transf_X = transf_X.with_columns(
                pl.when(pl.col(f).is_null())
                .then(pl.lit(-99))  # Null values
                .when(pl.col(f'{f}_ordinal_encoded').is_null())
                .then(pl.lit(-9999))  # Unknown categories
                .otherwise(pl.col(f'{f}_ordinal_encoded'))
                .alias(f'{f}_ordinal_encoded')
            )
        
        return transf_X


# -------------------- DUMMY ENCODING -------------------------------- #

class DummyEncoder(BaseTransformer):
    """One-hot encode categorical features with separate columns for null and unknown."""
    
    def __init__(self, features: List[str]):
        """
        Args:
            features: Categorical columns to encode
        """
        self.features = features
    
    def fit(self, X: pl.LazyFrame):
        """Learn unique categories for each feature."""
        self.encoding_dict: Dict[str, List[str]] = {}
        
        for f in self.features:
            # Get sorted unique non-null values
            unique_values = (
                X.select(f)
                .filter(pl.col(f).is_not_null())
                .unique()
                .collect()
                .get_column(f)
                .to_list()
            )
            self.encoding_dict[f] = sorted(unique_values)
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Create binary columns for each category, plus null and unknown."""
        transf_X = X.clone()
        
        for f in self.features:
            known_categories = self.encoding_dict[f]
            
            # Create binary column for each known category
            for category in known_categories:
                transf_X = transf_X.with_columns(
                    (pl.col(f) == category).cast(pl.Int8).alias(f'{f}_dummy_{category}')
                )
            
            # Binary column for null values
            transf_X = transf_X.with_columns(
                pl.col(f).is_null().cast(pl.Int8).alias(f'{f}_dummy_null')
            )
            
            # Binary column for unknown categories (not null and not in known)
            is_known = pl.lit(False)
            for category in known_categories:
                is_known = is_known | (pl.col(f) == category)
            
            transf_X = transf_X.with_columns(
                (pl.col(f).is_not_null() & ~is_known).cast(pl.Int8).alias(f'{f}_dummy_unknown')
            )
        
        return transf_X
    

# ------------------------------------------------------------------------------------------
# Scalers
# ------------------------------------------------------------------------------------------

class StandardScaler(BaseTransformer):
    """Standardize features by removing mean and scaling to unit variance."""
    
    def __init__(self, features: List[str]):
        """
        Args:
            features: Columns to standardize
        """
        self.features = features

    def fit(self, X: pl.LazyFrame):
        """Compute mean and std for each feature."""
        stats = X.select([
            pl.col(f).mean().alias(f'{f}_mean') for f in self.features
        ] + [
            pl.col(f).std().alias(f'{f}_std') for f in self.features
        ])
        
        self.stats = stats
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply z-score normalization: (x - mean) / std."""
        transf_X = X.clone()
        
        # Broadcast stats to all rows via cross join
        transf_X = transf_X.join(self.stats, how='cross')
        
        # Standardize each feature
        for f in self.features:
            transf_X = transf_X.with_columns(
                ((pl.col(f) - pl.col(f'{f}_mean')) / pl.col(f'{f}_std'))
                .alias(f'{f}_standard_scaled')
            )
        
        # Remove temporary stats columns
        cols_to_drop = [f'{f}_mean' for f in self.features] + [f'{f}_std' for f in self.features]
        transf_X = transf_X.drop(cols_to_drop)
        
        return transf_X


class MinMaxScaler(BaseTransformer):
    """Scale features to [0, 1] range using min-max normalization."""
    
    def __init__(self, features: List[str]):
        """
        Args:
            features: Columns to scale
        """
        self.features = features

    def fit(self, X: pl.LazyFrame):
        """Compute min and max for each feature."""
        stats = X.select([
            pl.col(f).min().alias(f'{f}_min') for f in self.features
        ] + [
            pl.col(f).max().alias(f'{f}_max') for f in self.features
        ])
        
        self.stats = stats
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply min-max scaling: (x - min) / (max - min)."""
        transf_X = X.clone()
        
        # Broadcast stats to all rows via cross join
        transf_X = transf_X.join(self.stats, how='cross')
        
        # Scale each feature
        for f in self.features:
            transf_X = transf_X.with_columns(
                ((pl.col(f) - pl.col(f'{f}_min')) / (pl.col(f'{f}_max') - pl.col(f'{f}_min')))
                .alias(f'{f}_minmax_scaled')
            )
        
        # Remove temporary stats columns
        cols_to_drop = [f'{f}_min' for f in self.features] + [f'{f}_max' for f in self.features]
        transf_X = transf_X.drop(cols_to_drop)
        
        return transf_X
    


# ------------------------------------------------------------------------------------------
# Transformation on Features
# ------------------------------------------------------------------------------------------   

class Log1pFeatures(BaseTransformer):
    """Apply log(1+x) transformation to features."""
    
    def __init__(self, features: List[str], new_features_suffix: str):
        """
        Args:
            features: Columns to transform
            new_features_suffix: Suffix for output column names
        """
        self.features = features
        self.new_feature_suffix = new_features_suffix

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply log(x+1) transformation."""
        return X.with_columns((pl.col(f)+1).log().alias(f'{f}_{self.new_feature_suffix}') for f in self.features)
    

class Expm1Features(BaseTransformer):
    """Apply exp(x-1) transformation to features."""
    
    def __init__(self, features: List[str], new_features_suffix: str):
        """
        Args:
            features: Columns to transform
            new_features_suffix: Suffix for output column names
        """
        self.features = features
        self.new_feature_suffix = new_features_suffix

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply exp(x-1) transformation."""
        return X.with_columns((pl.col(f)-1).exp().alias(f'{f}_{self.new_feature_suffix}') for f in self.features)
    

class PowerFeatures(BaseTransformer):
    """Apply power transformation to features."""
    
    def __init__(self, features: List[str], new_features_suffix: str, power: float = 2):
        """
        Args:
            features: Columns to transform
            new_features_suffix: Suffix for output column names
            power: Exponent for power transformation
        """
        self.features = features
        self.new_feature_suffix = new_features_suffix
        self.power = power

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Raise features to specified power."""
        return X.with_columns((pl.col(f)).pow(self.power).alias(f'{f}_{self.new_feature_suffix}') for f in self.features)
    

class InverseFeatures(BaseTransformer):
    """Apply inverse (1/x) transformation to features."""
    
    def __init__(self, features: List[str], new_features_suffix: str):
        """
        Args:
            features: Columns to transform
            new_features_suffix: Suffix for output column names
        """
        self.features = features
        self.new_feature_suffix = new_features_suffix

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Compute 1/x for each feature."""
        return X.with_columns((1/pl.col(f)).alias(f'{f}_{self.new_feature_suffix}') for f in self.features)



# ------------------------------------------------------------------------------------------
# Imputers
# ------------------------------------------------------------------------------------------

class SimpleImputer(BaseTransformer):
    """Impute missing values using mean or median strategy."""
    
    def __init__(self, features: List[str], strategy: str = 'mean'):
        """
        Args:
            features: Columns to impute
            strategy: 'mean' or 'median'
        
        Raises:
            ValueError: If strategy is not 'mean' or 'median'
        """
        self.features = features

        if strategy not in ['mean', 'median']:
            raise ValueError('SimpleImputer strategy params could be only "mean" or "median".')
        else:
            self.strategy = strategy

    def fit(self, X: pl.LazyFrame):
        """Compute imputation values based on strategy."""
        if self.strategy == 'median':
            self.values = X.select(self.features).median()
        else:
            self.values = X.select(self.features).mean()
        
        return self

    def transform(self, X: pl.LazyFrame):
        """Fill null and NaN values with computed statistics."""
        transf_X = (
            X
            .with_columns(pl.col(col).fill_null(self.values.select([col]).collect().item()) for col in self.features)
            .with_columns(pl.col(col).fill_nan(self.values.select([col]).collect().item()) for col in self.features)
        )

        return transf_X

class FillNulls(BaseTransformer):
    """Fill null and NaN values with a constant."""
    
    def __init__(self, features: List[str], value: float = -9999):
        """
        Args:
            features: Columns to fill
            value: Constant to use for filling
        """
        self.value = value
        self.features = features

    def fit(self, X: pl.LazyFrame):        
        return self

    def transform(self, X: pl.LazyFrame):
        """Replace null and NaN with constant value."""
        return X.with_columns(pl.col(col).fill_null(self.value).fill_nan(self.value) for col in self.features)


# ------------------------------------------------------------------------------------------
# Lags 
# ------------------------------------------------------------------------------------------

class GenerateLags(BaseTransformer):
    """Generate lagged features for time series data."""
    
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
        Args:
            ts_index: Column for time series identifier (e.g., entity ID)
            date_col: Date/time column
            lag_col: Column to lag
            lag_frequency: Time unit ('weeks', 'days', 'hours', etc.)
            lag_min: Minimum lag period
            lag_max: Maximum lag period
            lag_step: Step size between lags
        
        Raises:
            ValueError: If lag_frequency is not a valid time unit
        """
        time_arguments = [
            "weeks", "days", "hours", "minutes", "seconds",
            "milliseconds", "microseconds", "nanoseconds"
        ]
        self.ts_index = ts_index
        self.date_col = date_col
        self.lag_col = lag_col
        self.lag_min = lag_min
        self.lag_max = lag_max
        self.lag_step = lag_step

        if lag_frequency in time_arguments:
            self.lag_frequency = lag_frequency
        else:
            raise ValueError(f'lag_frequency should be in the following list: {time_arguments}')

    def fit(self, X: pl.LazyFrame):
        """Store base data for lag computation."""
        self.base_lag = X.select([self.ts_index, self.date_col, self.lag_col])
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Generate lag features by joining shifted dates."""
        # Combine fit and transform data, remove duplicates
        base = pl.concat([
            self.base_lag, 
            X.select([self.ts_index, self.date_col, self.lag_col])
        ]).unique()

        # Get time unit from date column dtype
        time_unit = X.select(self.date_col).collect().to_series().dtype.time_unit

        # Create lag feature for each time delta
        for delta in range(self.lag_min, self.lag_max + 1, self.lag_step):
            duration_dct = {self.lag_frequency: delta, 'time_unit': time_unit}
            X = (
                X.join(
                    base.with_columns(
                        pl.col(self.date_col) + pl.duration(**duration_dct)
                    ).rename({self.lag_col: f'{self.lag_col}_lag{delta}{self.lag_frequency}'}), 
                    how='left', 
                    on=[self.ts_index, self.date_col]
                )
            )

        return X