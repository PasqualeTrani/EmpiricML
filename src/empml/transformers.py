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
        return X.with_columns(
            pl.concat_list(self.features).list.std().alias(self.new_feature)
        )
    

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
        return X.with_columns(
            pl.concat_list(self.features).list.median().alias(self.new_feature)
        )
    
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


class InteractionFeatures(BaseTransformer):
    """Create pairwise multiplication features from feature pairs."""

    def __init__(self, feature_pairs: List[Tuple[str, str]], separator: str = '_x_'):
        """
        Args:
            feature_pairs: List of (col1, col2) tuples to multiply
            separator: String between feature names in output column (default: '_x_')
        """
        self.feature_pairs = feature_pairs
        self.separator = separator

    def fit(self, X: pl.LazyFrame):
        return self

    def transform(self, X: pl.LazyFrame):
        """Create f1 * f2 columns for each pair in a single with_columns call."""
        return X.with_columns([
            (pl.col(f1) * pl.col(f2)).alias(f'{f1}{self.separator}{f2}')
            for f1, f2 in self.feature_pairs
        ])


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
        """Compute mean target value per category and materialize mapping."""
        self.target_encoder_dict: Dict[str, pl.DataFrame] = {}
        
        # Calculate global mean for null-safety against unseen categories
        self.global_encoded_val = X.select(pl.col(self.encoder_col).mean()).collect().item()
        if self.global_encoded_val is None:
            self.global_encoded_val = 0.0 # Default if everything is null
        
        for f in self.features:
            # Always use prefix/suffix during fit to avoid duplicate column names
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            
            # Materialize lookup table to avoid plan explosion during transform
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).mean().alias(temp_col_name)
            ).collect()
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data and fill nulls for unseen categories."""
        
        # Join all encoded columns
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self.target_encoder_dict[f].lazy(), how='left', on=f)
            # Fill unseen categories with global mean
            X = X.with_columns(pl.col(temp_col_name).fill_null(self.global_encoded_val))
        
        # If replacing originals, drop them and rename encoded columns
        if self.replace_original:
            X = X.drop(self.features)
            # Rename encoded columns to original names
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        """Compute std of target value per category and materialize mapping."""
        self.target_encoder_dict: Dict[str, pl.DataFrame] = {}
        
        # Calculate global std for null-safety against unseen categories
        self.global_encoded_val = X.select(pl.col(self.encoder_col).std()).collect().item()
        if self.global_encoded_val is None:
            self.global_encoded_val = 0.0
            
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).std().alias(temp_col_name)
            ).collect()
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data and fill nulls."""
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self.target_encoder_dict[f].lazy(), how='left', on=f)
            X = X.with_columns(pl.col(temp_col_name).fill_null(self.global_encoded_val))
        
        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        """Compute max target value per category and materialize mapping."""
        self.target_encoder_dict: Dict[str, pl.DataFrame] = {}
        
        self.global_encoded_val = X.select(pl.col(self.encoder_col).max()).collect().item()
        if self.global_encoded_val is None:
            self.global_encoded_val = 0.0
            
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).max().alias(temp_col_name)
            ).collect()
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data and fill nulls."""
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self.target_encoder_dict[f].lazy(), how='left', on=f)
            X = X.with_columns(pl.col(temp_col_name).fill_null(self.global_encoded_val))
        
        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        """Compute min target value per category and materialize mapping."""
        self.target_encoder_dict: Dict[str, pl.DataFrame] = {}
        
        self.global_encoded_val = X.select(pl.col(self.encoder_col).min()).collect().item()
        if self.global_encoded_val is None:
            self.global_encoded_val = 0.0
            
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).min().alias(temp_col_name)
            ).collect()
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data and fill nulls."""
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self.target_encoder_dict[f].lazy(), how='left', on=f)
            X = X.with_columns(pl.col(temp_col_name).fill_null(self.global_encoded_val))
        
        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        """Compute median target value per category and materialize mapping."""
        self.target_encoder_dict: Dict[str, pl.DataFrame] = {}
        
        self.global_encoded_val = X.select(pl.col(self.encoder_col).median()).collect().item()
        if self.global_encoded_val is None:
            self.global_encoded_val = 0.0
            
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).median().alias(temp_col_name)
            ).collect()
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data and fill nulls."""
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self.target_encoder_dict[f].lazy(), how='left', on=f)
            X = X.with_columns(pl.col(temp_col_name).fill_null(self.global_encoded_val))
        
        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        """Compute kurtosis of target value per category and materialize mapping."""
        self.target_encoder_dict: Dict[str, pl.DataFrame] = {}
        
        self.global_encoded_val = X.select(pl.col(self.encoder_col).kurtosis()).collect().item()
        if self.global_encoded_val is None:
            self.global_encoded_val = 0.0
            
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).kurtosis().alias(temp_col_name)
            ).collect()
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data and fill nulls."""
        
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self.target_encoder_dict[f].lazy(), how='left', on=f)
            X = X.with_columns(pl.col(temp_col_name).fill_null(self.global_encoded_val))
        
        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        """Compute skewness of target value per category and materialize mapping."""
        self.target_encoder_dict: Dict[str, pl.DataFrame] = {}
        
        self.global_encoded_val = X.select(pl.col(self.encoder_col).skew()).collect().item()
        if self.global_encoded_val is None:
            self.global_encoded_val = 0.0
            
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            self.target_encoder_dict[f] = X.group_by(f).agg(
                pl.col(self.encoder_col).skew().alias(temp_col_name)
            ).collect()
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Join encoded values to input data and fill nulls."""
        for f in self.features:
            temp_col_name = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self.target_encoder_dict[f].lazy(), how='left', on=f)
            X = X.with_columns(pl.col(temp_col_name).fill_null(self.global_encoded_val))
        
        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        """Learn ordinal mapping and materialize lookup table."""
        self.encoding_dict: Dict[str, pl.DataFrame] = {}
        
        for f in self.features:
            temp_col_name = f'{f}{self.suffix}'
            # Get sorted unique values and assign sequential integers
            # Materialize lookup table to avoid plan explosion during transform
            unique_values = (
                X.select(f)
                .filter(pl.col(f).is_not_null())
                .unique()
                .sort(f)
                .with_row_index(name=temp_col_name)
                .collect()
            )
            self.encoding_dict[f] = unique_values
        
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Apply ordinal encoding with special values for null (-99) and unknown (-9999)."""
        
        for f in self.features:
            temp_col_name = f'{f}{self.suffix}'
            # Join encoding dictionary
            X = X.join(self.encoding_dict[f].lazy(), how='left', on=f)
            
            # Handle nulls and unknown categories
            X = X.with_columns(
                pl.when(pl.col(f).is_null())
                .then(pl.lit(-99))  # Null values
                .when(pl.col(temp_col_name).is_null())
                .then(pl.lit(-9999))  # Unknown categories
                .otherwise(pl.col(temp_col_name))
                .alias(temp_col_name)
            )
        
        # If replacing originals, drop them and rename encoded columns
        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{f}{self.suffix}': f 
                for f in self.features
            }
            X = X.rename(rename_mapping)
        
        return X


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
        
        for f in self.features:
            known_categories = self.encoding_dict[f]
            
            # Create all binary columns in a single with_columns call to prevent plan explosion
            X = X.with_columns([
                (pl.col(f) == category).cast(pl.Int8).alias(f'{f}_dummy_{category}')
                for category in known_categories
            ] + [
                pl.col(f).is_null().cast(pl.Int8).alias(f'{f}_dummy_null'),
                (pl.col(f).is_not_null() & ~pl.col(f).is_in(known_categories))
                .cast(pl.Int8)
                .alias(f'{f}_dummy_unknown')
            ])
        
        return X


# -------------------- FREQUENCY ENCODING -------------------------------- #

class FrequencyEncoder(BaseTransformer):
    """Encode categorical features by their frequency or proportion."""

    def __init__(
        self,
        features: List[str],
        normalize: bool = True,
        prefix: str = 'freq_',
        suffix: str = '_encoded',
        replace_original: bool = False,
    ):
        """
        Args:
            features: Categorical columns to encode
            normalize: If True, encode as proportion; if False, as raw count (default: True)
            prefix: Prefix for encoded column names (default: 'freq_')
            suffix: Suffix for encoded column names (default: '_encoded')
            replace_original: If True, drop original columns and use their names
                            for encoded columns, ignoring prefix/suffix (default: False)
        """
        self.features = features
        self.normalize = normalize
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
                "Setting prefix='freq_' and suffix='_encoded' for internal processing.",
                UserWarning,
                stacklevel=2
            )
            self.prefix = 'freq_'
            self.suffix = '_encoded'

        if self.replace_original and (self.prefix != 'freq_' or self.suffix != '_encoded'):
            warnings.warn(
                "replace_original=True: prefix and suffix arguments are ignored. "
                "Encoded columns will use original column names.",
                UserWarning,
                stacklevel=2
            )

    def fit(self, X: pl.LazyFrame):
        """Compute frequency (or proportion) per category and materialize mapping."""
        self._freq_dict: Dict[str, pl.DataFrame] = {}

        if self.normalize:
            self._total_count = X.select(pl.len().alias('_count')).collect().item()

        for f in self.features:
            temp_col = f'{self.prefix}{f}{self.suffix}'
            freq_df = (
                X.group_by(f)
                .agg(pl.len().alias(temp_col))
                .collect()
            )
            if self.normalize:
                freq_df = freq_df.with_columns(
                    (pl.col(temp_col).cast(pl.Float64) / self._total_count).alias(temp_col)
                )
            self._freq_dict[f] = freq_df

        return self

    def transform(self, X: pl.LazyFrame):
        """Join frequency values and fill unseen categories with 0."""
        for f in self.features:
            temp_col = f'{self.prefix}{f}{self.suffix}'
            X = X.join(self._freq_dict[f].lazy(), how='left', on=f)
            X = X.with_columns(
                pl.col(temp_col).fill_null(0.0 if self.normalize else 0)
            )

        if self.replace_original:
            X = X.drop(self.features)
            rename_mapping = {
                f'{self.prefix}{f}{self.suffix}': f
                for f in self.features
            }
            X = X.rename(rename_mapping)

        return X


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
        """Apply z-score normalization: (x - mean) / std. Handles std=0."""
        
        # Standardize each feature
        for f in self.features:
            mean_val = self.stats[f'{f}_mean'].item()
            std_val = self.stats[f'{f}_std'].item()
            
            # Handle null or zero std to avoid division by zero/inf
            if std_val is None or std_val == 0:
                X = X.with_columns(pl.lit(0.0).alias(f'{f}{self.suffix}'))
            else:
                X = X.with_columns(
                    ((pl.col(f) - mean_val) / std_val)
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
            min_val = self.stats[f'{f}_min'].item()
            max_val = self.stats[f'{f}_max'].item()
            
            # Prevent division by zero if all values are identical
            denominator = max_val - min_val
            if denominator == 0:
                X = X.with_columns(pl.lit(0.0).alias(f'{f}{self.suffix}'))
            else:
                X = X.with_columns(
                    ((pl.col(f) - min_val) / denominator).alias(f'{f}{self.suffix}')
                )
        
        return X
    


class RobustScaler(BaseTransformer):
    """Scale features using median and interquartile range (IQR)."""

    def __init__(self, features: List[str], suffix: str = ''):
        """
        Args:
            features: Columns to scale
            suffix: Suffix for scaled column names (default: '')
        """
        self.features = features
        self.suffix = suffix

    def fit(self, X: pl.LazyFrame):
        """Compute median, Q25, Q75 for each feature."""
        stats = X.select(
            [pl.col(f).median().alias(f'{f}_median') for f in self.features]
            + [pl.col(f).quantile(0.25).alias(f'{f}_q25') for f in self.features]
            + [pl.col(f).quantile(0.75).alias(f'{f}_q75') for f in self.features]
        )
        self.stats: pl.DataFrame = stats.collect()
        return self

    def transform(self, X: pl.LazyFrame):
        """Apply robust scaling: (x - median) / IQR. Handles IQR=0."""
        for f in self.features:
            median_val = self.stats[f'{f}_median'].item()
            q25_val = self.stats[f'{f}_q25'].item()
            q75_val = self.stats[f'{f}_q75'].item()
            iqr = q75_val - q25_val

            if iqr is None or iqr == 0:
                X = X.with_columns(pl.lit(0.0).alias(f'{f}{self.suffix}'))
            else:
                X = X.with_columns(
                    ((pl.col(f) - median_val) / iqr).alias(f'{f}{self.suffix}')
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
# Binning
# ------------------------------------------------------------------------------------------

class QuantileBinning(BaseTransformer):
    """Discretize continuous features into quantile-based bins."""

    def __init__(
        self,
        features: List[str],
        num_bins: int = 10,
        suffix: str = '_qbin',
        labels: Union[List[str], None] = None,
    ):
        """
        Args:
            features: Columns to discretize
            num_bins: Number of quantile bins (default: 10)
            suffix: Suffix for binned column names (default: '_qbin')
            labels: Optional string labels for bins; length must equal num_bins

        Raises:
            ValueError: If labels length does not match num_bins
        """
        self.features = features
        self.num_bins = num_bins
        self.suffix = suffix
        self.labels = labels

        if self.labels is not None and len(self.labels) != self.num_bins:
            raise ValueError(
                f"labels length ({len(self.labels)}) must equal num_bins ({self.num_bins})"
            )

    def fit(self, X: pl.LazyFrame):
        """Compute quantile bin edges for each feature and materialize."""
        self._bin_edges: Dict[str, List[float]] = {}
        quantiles = [i / self.num_bins for i in range(1, self.num_bins)]

        for f in self.features:
            edges = (
                X.select(
                    pl.col(f).quantile(q).alias(f'q_{q}')
                    for q in quantiles
                )
                .collect()
                .row(0)
            )
            unique_edges = sorted(set(e for e in edges if e is not None))
            self._bin_edges[f] = unique_edges

        return self

    def transform(self, X: pl.LazyFrame):
        """Assign bin indices based on learned quantile edges."""
        for f in self.features:
            edges = self._bin_edges[f]
            out_col = f'{f}{self.suffix}'

            if len(edges) == 0:
                X = X.with_columns(
                    pl.when(pl.col(f).is_null())
                    .then(pl.lit(None))
                    .otherwise(pl.lit(0))
                    .cast(pl.Int32)
                    .alias(out_col)
                )
                continue

            # Build when/then chain: null -> null, <= edge[0] -> 0, ...
            expr = pl.when(pl.col(f).is_null()).then(pl.lit(None))
            expr = expr.when(pl.col(f) <= edges[0]).then(pl.lit(0))
            for i in range(1, len(edges)):
                expr = expr.when(pl.col(f) <= edges[i]).then(pl.lit(i))
            expr = expr.otherwise(pl.lit(len(edges)))

            if self.labels is not None:
                # Actual number of bins may differ from num_bins due to deduplication
                actual_bins = len(edges) + 1
                label_map = {
                    i: self.labels[i] if i < len(self.labels) else self.labels[-1]
                    for i in range(actual_bins)
                }
                X = X.with_columns(expr.cast(pl.Int32).alias(out_col))
                X = X.with_columns(
                    pl.col(out_col).replace_strict(label_map, default=None).alias(out_col)
                )
            else:
                X = X.with_columns(expr.cast(pl.Int32).alias(out_col))

        return X


# ------------------------------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------------------------------

class RankFeatures(BaseTransformer):
    """Convert features to percentile rank based on training distribution."""

    def __init__(
        self,
        features: List[str],
        suffix: str = '_rank',
        method: Literal['average', 'min', 'max', 'dense'] = 'average',
    ):
        """
        Args:
            features: Columns to rank
            suffix: Suffix for ranked column names (default: '_rank')
            method: Ranking method (default: 'average')
                - 'average': average of min and max rank positions
                - 'min': lowest rank position
                - 'max': highest rank position
                - 'dense': like 'min' but ranks always increase by 1
        """
        self.features = features
        self.suffix = suffix
        self.method = method

    def fit(self, X: pl.LazyFrame):
        """Materialize sorted training values per feature for CDF-based ranking."""
        self._sorted_values: Dict[str, np.ndarray] = {}

        for f in self.features:
            vals = (
                X.select(pl.col(f).drop_nulls().sort())
                .collect()
                .to_series()
                .to_numpy()
            )
            self._sorted_values[f] = vals

        return self

    def transform(self, X: pl.LazyFrame):
        """Compute percentile rank [0, 1] relative to training distribution."""
        for f in self.features:
            out_col = f'{f}{self.suffix}'
            sorted_vals = self._sorted_values[f]
            n = len(sorted_vals)

            if n == 0:
                X = X.with_columns(pl.lit(None).cast(pl.Float64).alias(out_col))
                continue

            col_series = X.select(f).collect().to_series()
            np_values = col_series.to_numpy()
            null_mask = col_series.is_null().to_numpy()

            if self.method == 'min':
                ranks = np.searchsorted(sorted_vals, np_values, side='left')
            elif self.method == 'max':
                ranks = np.searchsorted(sorted_vals, np_values, side='right') - 1
            elif self.method == 'dense':
                unique_sorted = np.unique(sorted_vals)
                ranks = np.searchsorted(unique_sorted, np_values, side='left')
                n = len(unique_sorted)
            else:  # 'average'
                left = np.searchsorted(sorted_vals, np_values, side='left')
                right = np.searchsorted(sorted_vals, np_values, side='right')
                ranks = (left + right - 1) / 2.0

            percentile_ranks = np.clip(ranks / max(n - 1, 1), 0.0, 1.0)

            result = np.where(null_mask, np.nan, percentile_ranks)

            X = X.with_columns(
                pl.Series(name=out_col, values=result).cast(pl.Float64)
            )

        return X


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
        """Compute imputation values based on strategy and materialize results."""
        if self.strategy == 'median':
            stats = X.select(self.features).median().collect()
        else:
            stats = X.select(self.features).mean().collect()
        
        # Store as dictionary for fast lookup in transform without nested collection
        # Handle cases where column is empty or all nulls
        self.impute_values = {}
        for col in self.features:
            val = stats[col].item()
            self.impute_values[col] = val if val is not None else 0.0
        
        return self

    def transform(self, X: pl.LazyFrame):
        """Fill null and NaN values with computed statistics."""
        return X.with_columns([
            pl.col(col).fill_null(self.impute_values[col]).fill_nan(self.impute_values[col]) 
            for col in self.features
        ])

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

    def _validate_date_col(self, X: pl.LazyFrame):
        """Check if date_col is of type Date or Datetime."""
        dtype = X.select(self.date_col).limit(1).collect().to_series().dtype
        if not (dtype == pl.Date or isinstance(dtype, pl.Datetime)):
            raise TypeError(f"Column '{self.date_col}' must be of type Date or Datetime, not {dtype}")
        return dtype

    def fit(self, X: pl.LazyFrame):
        """Store base data for lag computation and materialize."""
        self._validate_date_col(X)
        self.base_lag = X.select([self.ts_index, self.date_col, self.lag_col]).collect()
        return self
    
    def transform(self, X: pl.LazyFrame):
        """Generate lag features by joining shifted dates."""
        # Ensure date_col is valid
        dtype = self._validate_date_col(X)
        
        # Combine materialized fit data with transform data (still lazy)
        # We materialise 'base' lookup table to prevent the plan from exploding in the loop
        base = pl.concat([
            self.base_lag.lazy(), 
            X.select([self.ts_index, self.date_col, self.lag_col])
        ]).unique().collect()

        # Materialize time unit once (only for Datetime)
        time_unit = getattr(dtype, 'time_unit', None)

        # Create lag feature for each time delta
        for delta in range(self.lag_min, self.lag_max + 1, self.lag_step):
            duration_dct = {self.lag_frequency: delta, 'time_unit': time_unit}
            X = (
                X.join(
                    base.lazy().with_columns(
                        pl.col(self.date_col) + pl.duration(**duration_dct)
                    ).rename({self.lag_col: f'{self.lag_col}_lag{delta}{self.lag_frequency}'}), 
                    how='left', 
                    on=[self.ts_index, self.date_col]
                )
            )

        return X


# ------------------------------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------------------------------

class KMeansCluster(BaseTransformer):
    """Assign cluster labels using KMeans on selected features."""

    def __init__(
        self,
        features: List[str],
        num_clusters: int = 8,
        new_feature: str = 'kmeans_cluster',
        random_state: int = 42,
    ):
        """
        Args:
            features: Numeric columns to use for clustering
            num_clusters: Number of clusters (default: 8)
            new_feature: Name of the output cluster column (default: 'kmeans_cluster')
            random_state: Random seed for reproducibility (default: 42)
        """
        self.features = features
        self.num_clusters = num_clusters
        self.new_feature = new_feature
        self.random_state = random_state

    def fit(self, X: pl.LazyFrame):
        """Collect features to numpy, impute NaN with column means, fit KMeans."""
        from sklearn.cluster import KMeans

        data = X.select(self.features).collect().to_numpy().astype(np.float64)

        # KMeans does not accept NaN; impute with column means
        col_means = np.nanmean(data, axis=0)
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            data[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        self._col_means: np.ndarray = col_means

        self._kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            n_init='auto',
        )
        self._kmeans.fit(data)
        return self

    def transform(self, X: pl.LazyFrame):
        """Predict cluster labels and add as a new column."""
        data = X.select(self.features).collect().to_numpy().astype(np.float64)

        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            data[nan_mask] = np.take(self._col_means, np.where(nan_mask)[1])

        labels = self._kmeans.predict(data)

        return X.with_columns(
            pl.Series(name=self.new_feature, values=labels).cast(pl.Int32)
        )


# ------------------------------------------------------------------------------------------
# Dimensionality Reduction
# ------------------------------------------------------------------------------------------

class PCATransformer(BaseTransformer):
    """Reduce dimensionality using Principal Component Analysis."""

    def __init__(
        self,
        features: List[str],
        n_components: int = 2,
        prefix: str = 'pc_',
    ):
        """
        Args:
            features: Numeric columns to use for PCA
            n_components: Number of principal components to keep (default: 2)
            prefix: Prefix for output column names (default: 'pc_')
        """
        self.features = features
        self.n_components = n_components
        self.prefix = prefix

    def fit(self, X: pl.LazyFrame):
        """Collect features to numpy, impute NaN with column means, fit PCA."""
        from sklearn.decomposition import PCA

        data = X.select(self.features).collect().to_numpy().astype(np.float64)

        col_means = np.nanmean(data, axis=0)
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            data[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        self._col_means: np.ndarray = col_means

        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(data)
        return self

    def transform(self, X: pl.LazyFrame):
        """Transform features and add principal component columns."""
        data = X.select(self.features).collect().to_numpy().astype(np.float64)

        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            data[nan_mask] = np.take(self._col_means, np.where(nan_mask)[1])

        components = self._pca.transform(data)

        pc_columns = [
            pl.Series(name=f'{self.prefix}{i}', values=components[:, i])
            for i in range(self.n_components)
        ]

        return X.with_columns(pc_columns)