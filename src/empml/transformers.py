"""
Feature engineering transformers for machine learning pipelines.

Provides transformers for feature operations, encoding, scaling, imputation,
and time series lag generation, all compatible with Polars LazyFrame.
"""

# base imports 
import warnings
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
    
    def __init__(
        self, 
        features: List[str], 
        encoder_col: str,
        prefix: str = 'mean_',
        suffix: str = '_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
            prefix: Prefix for encoded column names (default: 'mean_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.encoder_col = encoder_col
        self.prefix = prefix
        self.suffix = suffix
        self.replace_original = replace_original
        
        # Handle empty prefix and suffix configurations
        if not self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "prefix='' and suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning
            )
            self.replace_original = True
        
        if self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "replace_original=True with prefix='' and suffix='' would cause errors. "
                "Setting prefix='mean_' and suffix='_encoded' for internal processing.",
                UserWarning
            )
            self.prefix = 'mean_'
            self.suffix = '_encoded'
        
        # Warn if prefix/suffix are set but will be ignored
        if self.replace_original and (self.prefix != 'mean_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning
            )

    def fit(self, X: pl.LazyFrame):
        """Compute mean target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            # Always use prefix/suffix during fit to avoid duplicate column names
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).mean().alias(temp_col_name)
            )
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        
        # Join all encoded columns
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        
        # If replacing originals, drop them and rename encoded columns
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            # Rename encoded columns to original names
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
        return transf_X


class StdTargetEncoder(BaseTransformer):
    """Encode categorical features with std of target variable."""
    
    def __init__(
        self, 
        features: List[str], 
        encoder_col: str,
        prefix: str = 'std_',
        suffix: str = '_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
            prefix: Prefix for encoded column names (default: 'std_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.encoder_col = encoder_col
        self.prefix = prefix
        self.suffix = suffix
        self.replace_original = replace_original
        
        # Handle empty prefix and suffix configurations
        if not self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "prefix='' and suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning,
                stacklevel=2
            )
            self.replace_original = True
        
        if self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "replace_original=True with prefix='' and suffix='' would cause errors. "
                "Setting prefix='std_' and suffix='_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.prefix = 'std_'
            self.suffix = '_encoded'
        
        # Warn if prefix/suffix are set but will be ignored
        if self.replace_original and (self.prefix != 'std_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Compute std of target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).std().alias(temp_col_name)
            )
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
        return transf_X


class MaxTargetEncoder(BaseTransformer):
    """Encode categorical features with max of target variable."""
    
    def __init__(
        self, 
        features: List[str], 
        encoder_col: str,
        prefix: str = 'max_',
        suffix: str = '_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
            prefix: Prefix for encoded column names (default: 'max_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.encoder_col = encoder_col
        self.prefix = prefix
        self.suffix = suffix
        self.replace_original = replace_original
        
        if not self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "prefix='' and suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning,
                stacklevel=2
            )
            self.replace_original = True
        
        if self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "replace_original=True with prefix='' and suffix='' would cause errors. "
                "Setting prefix='max_' and suffix='_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.prefix = 'max_'
            self.suffix = '_encoded'
        
        if self.replace_original and (self.prefix != 'max_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Compute max target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).max().alias(temp_col_name)
            )
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
        return transf_X


class MinTargetEncoder(BaseTransformer):
    """Encode categorical features with min of target variable."""
    
    def __init__(
        self, 
        features: List[str], 
        encoder_col: str,
        prefix: str = 'min_',
        suffix: str = '_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
            prefix: Prefix for encoded column names (default: 'min_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.encoder_col = encoder_col
        self.prefix = prefix
        self.suffix = suffix
        self.replace_original = replace_original
        
        if not self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "prefix='' and suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning,
                stacklevel=2
            )
            self.replace_original = True
        
        if self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "replace_original=True with prefix='' and suffix='' would cause errors. "
                "Setting prefix='min_' and suffix='_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.prefix = 'min_'
            self.suffix = '_encoded'
        
        if self.replace_original and (self.prefix != 'min_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Compute min target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).min().alias(temp_col_name)
            )
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
        return transf_X


class MedianTargetEncoder(BaseTransformer):
    """Encode categorical features with median of target variable."""
    
    def __init__(
        self, 
        features: List[str], 
        encoder_col: str,
        prefix: str = 'median_',
        suffix: str = '_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
            prefix: Prefix for encoded column names (default: 'median_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.encoder_col = encoder_col
        self.prefix = prefix
        self.suffix = suffix
        self.replace_original = replace_original
        
        if not self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "prefix='' and suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning,
                stacklevel=2
            )
            self.replace_original = True
        
        if self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "replace_original=True with prefix='' and suffix='' would cause errors. "
                "Setting prefix='median_' and suffix='_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.prefix = 'median_'
            self.suffix = '_encoded'
        
        if self.replace_original and (self.prefix != 'median_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Compute median target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).median().alias(temp_col_name)
            )
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
        return transf_X


class KurtTargetEncoder(BaseTransformer):
    """Encode categorical features with kurtosis of target variable."""
    
    def __init__(
        self, 
        features: List[str], 
        encoder_col: str,
        prefix: str = 'kurt_',
        suffix: str = '_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
            prefix: Prefix for encoded column names (default: 'kurt_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.encoder_col = encoder_col
        self.prefix = prefix
        self.suffix = suffix
        self.replace_original = replace_original
        
        if not self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "prefix='' and suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning,
                stacklevel=2
            )
            self.replace_original = True
        
        if self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "replace_original=True with prefix='' and suffix='' would cause errors. "
                "Setting prefix='kurt_' and suffix='_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.prefix = 'kurt_'
            self.suffix = '_encoded'
        
        if self.replace_original and (self.prefix != 'kurt_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Compute kurtosis of target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).kurtosis().alias(temp_col_name)
            )
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
        return transf_X


class SkewTargetEncoder(BaseTransformer):
    """Encode categorical features with skewness of target variable."""
    
    def __init__(
        self, 
        features: List[str], 
        encoder_col: str,
        prefix: str = 'skew_',
        suffix: str = '_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            encoder_col: Target column to aggregate
            prefix: Prefix for encoded column names (default: 'skew_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.encoder_col = encoder_col
        self.prefix = prefix
        self.suffix = suffix
        self.replace_original = replace_original
        
        if not self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "prefix='' and suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning,
                stacklevel=2
            )
            self.replace_original = True
        
        if self.replace_original and self.prefix == '' and self.suffix == '':
            warnings.warn(
                "replace_original=True with prefix='' and suffix='' would cause errors. "
                "Setting prefix='skew_' and suffix='_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.prefix = 'skew_'
            self.suffix = '_encoded'
        
        if self.replace_original and (self.prefix != 'skew_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Compute skewness of target value per category."""
        self.target_encoder_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).skew().alias(temp_col_name)
            )
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data."""
        transf_X = X.clone()
        
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how='left', on=f)
        
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
        return transf_X


# -------------------- ORDINAL ENCODING -------------------------------- #

class OrdinalEncoder(BaseTransformer):
    """Encode categorical features with ordinal integers based on sorted order."""
    
    def __init__(
        self, 
        features: List[str],
        suffix: str = '_ordinal_encoded',
        replace_original: bool = False
    ):
        """
        Args:
            features: Categorical columns to encode
            suffix: Suffix for encoded column names (default: '_ordinal_encoded')
            replace_original: If True, drop original columns and use their names 
                            for encoded columns, ignoring suffix (default: False)
        """
        self.features = features
        self.suffix = suffix
        self.replace_original = replace_original
        
        # Handle empty suffix configuration
        if not self.replace_original and self.suffix == '':
            warnings.warn(
                "suffix='' with replace_original=False would create duplicate "
                "column names. Setting replace_original=True automatically.",
                UserWarning,
                stacklevel=2
            )
            self.replace_original = True
        
        if self.replace_original and self.suffix == '':
            warnings.warn(
                "replace_original=True with suffix='' would cause errors. "
                "Setting suffix='_ordinal_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.suffix = '_ordinal_encoded'
        
        # Warn if suffix is set but will be ignored
        if self.replace_original and self.suffix != '_ordinal_encoded':
            warnings.warn(
                "replace_original=True: suffix argument is ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Learn ordinal mapping from sorted unique values."""
        self.encoding_dict: Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{f}{self.suffix}'
            # Get sorted unique values and assign sequential integers
            unique_values = (
                X.select(f)
                .filter(pl.col(f).is_not_null())
                .unique()
                .sort(f)
                .with_row_index(name=temp_col_name)
            )
            self.encoding_dict[f] = unique_values
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply ordinal encoding with special values for null (-99) and unknown (-9999)."""
        transf_X = X.clone()
        
        for f in self.features:
            temp_col_name = f'{f}{self.suffix}'
            # Join encoding dictionary
            transf_X = transf_X.join(self.encoding_dict[f], how='left', on=f)
            
            # Handle nulls and unknown categories
            transf_X = transf_X.with_columns(
                pl.when(pl.col(f).is_null())
                .then(pl.lit(-99))  # Null values
                .when(pl.col(temp_col_name).is_null())
                .then(pl.lit(-9999))  # Unknown categories
                .otherwise(pl.col(temp_col_name))
                .alias(temp_col_name)
            )
        
        # If replacing originals, drop them and rename encoded columns
        if self.replace_original:
            transf_X = transf_X.drop(self.features)
            rename_mapping = {
                f'{f}{self.suffix}': f 
                for f in self.features
            }
            transf_X = transf_X.rename(rename_mapping)
        
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
    
    def __init__(self, features: List[str], suffix: str = ''):
        """
        Args:
            features: Columns to standardize
            suffix: Suffix for scaled column names (default: '')
        """
        self.features = features
        self.suffix = suffix

    def fit(self, X: pl.LazyFrame):
        """Compute mean and std for each feature."""
        stats = X.select([
            pl.col(f).mean().alias(f'{f}_mean') for f in self.features
        ] + [
            pl.col(f).std().alias(f'{f}_std') for f in self.features
        ])
        
        self.stats : pl.DataFrame = stats.collect()
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply z-score normalization: (x - mean) / std."""
        
        # Standardize each feature
        for f in self.features:
            X = X.with_columns(
                ((pl.col(f) - self.stats[f'{f}_mean'].item()) / self.stats[f'{f}_std'].item())
                .alias(f'{f}{self.suffix}')
            )
        
        return X


class MinMaxScaler(BaseTransformer):
    """Scale features to [0, 1] range using min-max normalization."""
    
    def __init__(self, features: List[str], suffix: str = ''):
        """
        Args:
            features: Columns to scale
            suffix: Suffix for scaled column names (default: '')
        """
        self.features = features
        self.suffix = suffix 

    def fit(self, X: pl.LazyFrame):
        """Compute min and max for each feature."""
        stats = X.select([
            pl.col(f).min().alias(f'{f}_min') for f in self.features
        ] + [
            pl.col(f).max().alias(f'{f}_max') for f in self.features
        ])
        
        self.stats : pl.DataFrame = stats.collect()
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply min-max scaling: (x - min) / (max - min)."""
        
        # Scale each feature
        for f in self.features:
            X = X.with_columns(
                ((pl.col(f) - self.stats[f'{f}_min'].item()) / (self.stats[f'{f}_max'].item()) - (self.stats[f'{f}_min'].item()))
                .alias(f'{f}{self.suffix}')
            )
        
        return X
    


# ------------------------------------------------------------------------------------------
# Transformation on Features
# ------------------------------------------------------------------------------------------   

class Log1pFeatures(BaseTransformer):
    """Apply log(1+x) transformation to features."""
    
    def __init__(self, features: List[str], suffix: str = ''):
        """
        Args:
            features: Columns to transform
            suffix: Suffix for output column names
        """
        self.features = features
        self.suffix = suffix

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply log(x+1) transformation."""
        return X.with_columns((pl.col(f)+1).log().alias(f'{f}{self.suffix}') for f in self.features)
    

class Expm1Features(BaseTransformer):
    """Apply exp(x-1) transformation to features."""
    
    def __init__(self, features: List[str], suffix: str = ''):
        """
        Args:
            features: Columns to transform
            suffix: Suffix for output column names
        """
        self.features = features
        self.suffix = suffix

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply exp(x-1) transformation."""
        return X.with_columns((pl.col(f)-1).exp().alias(f'{f}{self.suffix}') for f in self.features)
    

class PowerFeatures(BaseTransformer):
    """Apply power transformation to features."""
    
    def __init__(self, features: List[str], suffix: str = '', power: float = 2):
        """
        Args:
            features: Columns to transform
            suffix: Suffix for output column names
            power: Exponent for power transformation
        """
        self.features = features
        self.suffix = suffix
        self.power = power

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Raise features to specified power."""
        return X.with_columns((pl.col(f)).pow(self.power).alias(f'{f}{self.suffix}') for f in self.features)
    

class InverseFeatures(BaseTransformer):
    """Apply inverse (1/x) transformation to features."""
    
    def __init__(self, features: List[str], suffix: str = ''):
        """
        Args:
            features: Columns to transform
            suffix: Suffix for output column names
        """
        self.features = features
        self.suffix = suffix

    def fit(self, X: pl.LazyFrame):
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Compute 1/x for each feature."""
        return X.with_columns((1/pl.col(f)).alias(f'{f}{self.suffix}') for f in self.features)



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