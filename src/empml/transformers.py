# base imports 
from typing import Union, Literal, List, Dict, Tuple

# data wranglers 
import polars as pl # type:ignore
import numpy as np # type:ignore

# internal imports 
from empml.base import BaseTransformer


# ------------------------------------------------------------------------------------------
# Algebric operation between features
# ------------------------------------------------------------------------------------------

class AvgFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns(pl.mean_horizontal(self.features).alias(self.new_feature))
    
    
class MaxFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns(pl.max_horizontal(self.features).alias(self.new_feature))
    

class MinFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns(pl.min_horizontal(self.features).alias(self.new_feature))
    

class StdFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().std(1)).alias(self.new_feature))
    

class MedianFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().median(1)).alias(self.new_feature))
    

class KurtFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().kurt(1)).alias(self.new_feature))
    

class SkewFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns(pl.Series(X.select(self.features).collect().to_pandas().skew(1)).alias(self.new_feature))
    

class ModuleFeatures(BaseTransformer):
    def __init__(self, features : Tuple[str, str], new_feature : str):
        self.features = features
        self.new_feature = new_feature

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        f1, f2 = self.features # the two features on which compute the module
        return X.with_columns(((pl.col(f1)**2) + (pl.col(f2)**2)).sqrt().alias(self.new_feature))
    

# ------------------------------------------------------------------------------------------
# Categorical Encoding 
# ------------------------------------------------------------------------------------------


# -------------------- TARGET ENCODING -------------------------------- #
class MeanTargetEncoder(BaseTransformer):
    def __init__(self, features : List[str], encoder_col : str):

        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X : pl.LazyFrame):

        self.target_encoder_dict : Dict[str, pl.LazyFrame] = {
            f : X.group_by(f).agg(pl.col(self.encoder_col).mean().alias(f'mean_{f}_target_encoded')) for f in self.features
        }

        return self
    
    def transform(self, X : pl.LazyFrame):

        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how = 'left', on = f)

        return transf_X


class StdTargetEncoder(BaseTransformer):
    def __init__(self, features : List[str], encoder_col : str):
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X : pl.LazyFrame):
        self.target_encoder_dict : Dict[str, pl.LazyFrame] = {
            f : X.group_by(f).agg(pl.col(self.encoder_col).std().alias(f'std_{f}_target_encoded')) for f in self.features
        }
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how = 'left', on = f)
        return transf_X


class MaxTargetEncoder(BaseTransformer):
    def __init__(self, features : List[str], encoder_col : str):
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X : pl.LazyFrame):
        self.target_encoder_dict : Dict[str, pl.LazyFrame] = {
            f : X.group_by(f).agg(pl.col(self.encoder_col).max().alias(f'max_{f}_target_encoded')) for f in self.features
        }
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how = 'left', on = f)
        return transf_X


class MinTargetEncoder(BaseTransformer):
    def __init__(self, features : List[str], encoder_col : str):
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X : pl.LazyFrame):
        self.target_encoder_dict : Dict[str, pl.LazyFrame] = {
            f : X.group_by(f).agg(pl.col(self.encoder_col).min().alias(f'min_{f}_target_encoded')) for f in self.features
        }
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how = 'left', on = f)
        return transf_X


class MedianTargetEncoder(BaseTransformer):
    def __init__(self, features : List[str], encoder_col : str):
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X : pl.LazyFrame):
        self.target_encoder_dict : Dict[str, pl.LazyFrame] = {
            f : X.group_by(f).agg(pl.col(self.encoder_col).median().alias(f'median_{f}_target_encoded')) for f in self.features
        }
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how = 'left', on = f)
        return transf_X


class KurtTargetEncoder(BaseTransformer):
    def __init__(self, features : List[str], encoder_col : str):
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X : pl.LazyFrame):
        self.target_encoder_dict : Dict[str, pl.LazyFrame] = {
            f : X.group_by(f).agg(pl.col(self.encoder_col).kurtosis().alias(f'kurt_{f}_target_encoded')) for f in self.features
        }
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how = 'left', on = f)
        return transf_X


class SkewTargetEncoder(BaseTransformer):
    def __init__(self, features : List[str], encoder_col : str):
        self.features = features
        self.encoder_col = encoder_col

    def fit(self, X : pl.LazyFrame):
        self.target_encoder_dict : Dict[str, pl.LazyFrame] = {
            f : X.group_by(f).agg(pl.col(self.encoder_col).skew().alias(f'skew_{f}_target_encoded')) for f in self.features
        }
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        for f in self.features:
            transf_X = transf_X.join(self.target_encoder_dict[f], how = 'left', on = f)
        return transf_X


# -------------------- ORDINAL ENCODING -------------------------------- #

class OrdinalEncoder(BaseTransformer):
    def __init__(self, features : List[str]):
        self.features = features

    def fit(self, X : pl.LazyFrame):
        self.encoding_dict : Dict[str, pl.LazyFrame] = {}
        
        for f in self.features:
            # Get unique non-null values for each feature and assign ordinal values
            unique_values = (
                X.select(f)
                .filter(pl.col(f).is_not_null())
                .unique()
                .sort(f)
                .with_row_index(name=f'{f}_ordinal_encoded')
            )
            self.encoding_dict[f] = unique_values
        
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        
        for f in self.features:
            # Join the encoding dictionary to get ordinal values
            transf_X = transf_X.join(
                self.encoding_dict[f], 
                how='left', 
                on=f
            )
            
            # Handle null values and unknown categories
            transf_X = transf_X.with_columns(
                pl.when(pl.col(f).is_null())
                .then(pl.lit(-99))
                .when(pl.col(f'{f}_ordinal_encoded').is_null())
                .then(pl.lit(-9999))
                .otherwise(pl.col(f'{f}_ordinal_encoded'))
                .alias(f'{f}_ordinal_encoded')
            )
        
        return transf_X


# -------------------- DUMMY ENCODING -------------------------------- #

class DummyEncoder(BaseTransformer):
    def __init__(self, features: List[str]):
        self.features = features
    
    def fit(self, X: pl.LazyFrame):
        self.encoding_dict: Dict[str, List[str]] = {}
        
        for f in self.features:
            # Get unique non-null values for each feature
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
        transf_X = X.clone()
        
        for f in self.features:
            known_categories = self.encoding_dict[f]
            
            # Create dummy columns for each known category
            for category in known_categories:
                transf_X = transf_X.with_columns(
                    (pl.col(f) == category).cast(pl.Int8).alias(f'{f}_dummy_{category}')
                )
            
            # Create column for null values
            transf_X = transf_X.with_columns(
                pl.col(f).is_null().cast(pl.Int8).alias(f'{f}_dummy_null')
            )
            
            # Create column for unknown categories
            # Unknown = not null AND not in any known category
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
    def __init__(self, features : List[str]):
        self.features = features

    def fit(self, X : pl.LazyFrame):
        # Calculate mean and std for each feature
        stats = X.select([
            pl.col(f).mean().alias(f'{f}_mean') for f in self.features
        ] + [
            pl.col(f).std().alias(f'{f}_std') for f in self.features
        ])
        
        self.stats = stats
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        
        # Join stats (this will broadcast the single row to all rows)
        transf_X = transf_X.join(self.stats, how='cross')
        
        # Apply standardization: (x - mean) / std
        for f in self.features:
            transf_X = transf_X.with_columns(
                ((pl.col(f) - pl.col(f'{f}_mean')) / pl.col(f'{f}_std'))
                .alias(f'{f}_standard_scaled')
            )
        
        # Drop the temporary mean and std columns
        cols_to_drop = [f'{f}_mean' for f in self.features] + [f'{f}_std' for f in self.features]
        transf_X = transf_X.drop(cols_to_drop)
        
        return transf_X


class MinMaxScaler(BaseTransformer):
    def __init__(self, features : List[str]):
        self.features = features

    def fit(self, X : pl.LazyFrame):
        # Calculate min and max for each feature
        stats = X.select([
            pl.col(f).min().alias(f'{f}_min') for f in self.features
        ] + [
            pl.col(f).max().alias(f'{f}_max') for f in self.features
        ])
        
        self.stats = stats
        return self
    
    def transform(self, X : pl.LazyFrame):
        transf_X = X.clone()
        
        # Join stats (this will broadcast the single row to all rows)
        transf_X = transf_X.join(self.stats, how='cross')
        
        # Apply min-max scaling: (x - min) / (max - min)
        for f in self.features:
            transf_X = transf_X.with_columns(
                ((pl.col(f) - pl.col(f'{f}_min')) / (pl.col(f'{f}_max') - pl.col(f'{f}_min')))
                .alias(f'{f}_minmax_scaled')
            )
        
        # Drop the temporary min and max columns
        cols_to_drop = [f'{f}_min' for f in self.features] + [f'{f}_max' for f in self.features]
        transf_X = transf_X.drop(cols_to_drop)
        
        return transf_X
    


# ------------------------------------------------------------------------------------------
# Transformation on Features
# ------------------------------------------------------------------------------------------   

class Log1pFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_features_suffix : str):
        self.features = features
        self.new_feature_suffix = new_features_suffix

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns((pl.col(f)+1).log().alias(f'{f}_{self.new_feature_suffix}') for f in self.features)
    

class Expm1Features(BaseTransformer):
    def __init__(self, features : List[str], new_features_suffix : str):
        self.features = features
        self.new_feature_suffix = new_features_suffix

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns((pl.col(f)-1).exp().alias(f'{f}_{self.new_feature_suffix}') for f in self.features)
    

class PowerFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_features_suffix : str, power : float = 2):
        self.features = features
        self.new_feature_suffix = new_features_suffix
        self.power = power

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns((pl.col(f)).pow(self.power).alias(f'{f}_{self.new_feature_suffix}') for f in self.features)
    

class InverseFeatures(BaseTransformer):
    def __init__(self, features : List[str], new_features_suffix : str):
        self.features = features
        self.new_feature_suffix = new_features_suffix

    def fit(self, X : pl.LazyFrame):
        return self
    
    def transform(self, X : pl.LazyFrame):
        return X.with_columns((1/pl.col(f)).alias(f'{f}_{self.new_feature_suffix}') for f in self.features)
