# base imports 
from typing import Union, Dict, List, Tuple

# data wranglers 
import polars as pl # type: ignore
import numpy as np # type: ignore

# internal imports 
from empml.base import BaseTransformer, BaseEstimator, Metric # base classes 
from empml.utils import log_execution_time, log_step, time_execution

    
# ------------------------------------------------------------------------------------------
# PIPELINE 
# ------------------------------------------------------------------------------------------

class Pipeline:
    """
    Custom pipeline for chaining transformers and an optional final estimator.
    Works exclusively with Polars LazyFrames.
    
    Supports:
    - Transformer-only pipelines (returns transformed LazyFrame)
    - Transformer + Estimator pipelines (returns predictions)
    - Nested pipelines (a Pipeline can be a step in another Pipeline)
    
    Example (with estimator):
        pipeline = Pipeline([
            ('imputer', SimpleImputerTransformer(features=['col1', 'col2'])),
            ('scaler', StandardScalerTransformer(features=['col1', 'col2'])),
            ('model', lgbm_reg(features=['col1', 'col2'], target='target'))
        ])
        
        pipeline.fit(train_lf)
        predictions = pipeline.predict(test_lf)
    
    Example (transformer-only):
        preprocessing = Pipeline([
            ('imputer', SimpleImputerTransformer(features=['col1', 'col2'])),
            ('scaler', StandardScalerTransformer(features=['col1', 'col2']))
        ])
        
        preprocessing.fit(train_lf)
        transformed_lf = preprocessing.transform(test_lf)
    
    Example (nested pipelines):
        preprocessing = Pipeline([
            ('imputer', SimpleImputerTransformer(features=['col1', 'col2'])),
            ('scaler', StandardScalerTransformer(features=['col1', 'col2']))
        ])
        
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', lgbm_reg(features=['col1', 'col2'], target='target'))
        ])
    """
    def __init__(self, steps: list[tuple[str, Union[BaseTransformer, BaseEstimator, 'Pipeline']]]):
        """
        Parameters:
        -----------
        steps : list of tuples
            List of (name, transformer/estimator/pipeline) tuples in the order they should be applied.
            If the last step is an estimator, the pipeline will support predict().
            If all steps are transformers (or pipelines acting as transformers), the pipeline 
            will support transform().
        """
        self.steps = steps
        self._validate_steps()
        self._is_transformer_only = self._check_if_transformer_only()
    
    def _validate_steps(self):
        """Validate that steps are properly configured."""
        if len(self.steps) == 0:
            raise ValueError("Pipeline must have at least one step")
        
        # Check that all steps except possibly the last are transformers or pipelines
        for name, step in self.steps[:-1]:
            if not (isinstance(step, (BaseTransformer, Pipeline))):
                raise ValueError(
                    f"All steps except the last must be transformers or pipelines. "
                    f"'{name}' is neither."
                )
        
        # The last step can be a transformer, estimator, or pipeline
        last_name, last_step = self.steps[-1]
        if not isinstance(last_step, (BaseTransformer, BaseEstimator, Pipeline)):
            raise ValueError(
                f"Last step '{last_name}' must be a transformer, estimator, or pipeline."
            )
    
    def _check_if_transformer_only(self) -> bool:
        """Check if pipeline contains only transformers (no final estimator)."""
        last_name, last_step = self.steps[-1]
        
        # If last step is a Pipeline, check if it's transformer-only
        if isinstance(last_step, Pipeline):
            return last_step._is_transformer_only
        
        # Otherwise, check if it's a transformer
        return isinstance(last_step, BaseTransformer)
    
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
        # Apply transformers sequentially
        lf_transformed = lf
        for name, step in self.steps[:-1]:
            if isinstance(step, Pipeline):
                lf_transformed = step.fit_transform(lf_transformed)
            else:
                lf_transformed = step.fit_transform(lf_transformed)
        
        # Fit the final step
        final_name, final_step = self.steps[-1]
        if isinstance(final_step, Pipeline):
            final_step.fit(lf_transformed, **fit_params)
        elif isinstance(final_step, BaseTransformer):
            final_step.fit(lf_transformed)
        else:  # BaseEstimator
            final_step.fit(lf_transformed, **fit_params)
        
        return self
    
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
        if not self._is_transformer_only:
            raise ValueError(
                "fit_transform() is only available for transformer-only pipelines. "
                "This pipeline has an estimator as the final step. Use fit_predict() instead."
            )
        
        self.fit(lf)
        return self.transform(lf)
    
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
        if self._is_transformer_only:
            raise ValueError(
                "predict() is only available for pipelines with an estimator. "
                "This pipeline contains only transformers. Use transform() instead."
            )
        
        # Apply transformers sequentially
        lf_transformed = lf
        for name, step in self.steps[:-1]:
            if isinstance(step, Pipeline):
                lf_transformed = step.transform(lf_transformed)
            else:
                lf_transformed = step.transform(lf_transformed)
        
        # Predict with the final estimator
        final_name, final_estimator = self.steps[-1]
        if isinstance(final_estimator, Pipeline):
            return final_estimator.predict(lf_transformed)
        else:
            return final_estimator.predict(lf_transformed)
    
    def fit_predict(self, lf: pl.LazyFrame, **fit_params) -> np.ndarray:
        """
        Fit the pipeline and return predictions on the same data.
        Only available for pipelines with an estimator.
        """
        if self._is_transformer_only:
            raise ValueError(
                "fit_predict() is only available for pipelines with an estimator. "
                "This pipeline contains only transformers. Use fit_transform() instead."
            )
        
        self.fit(lf, **fit_params)
        return self.predict(lf)
    
    def __getitem__(self, index: Union[int, str]):
        """Access a step by index or name."""
        if isinstance(index, int):
            return self.steps[index][1]
        
        for name, step in self.steps:
            if name == index:
                return step
        
        raise KeyError(f"Step '{index}' not found in pipeline")
    
    def __len__(self):
        return len(self.steps)
    
    def __repr__(self):
        steps_str = ",\n    ".join([f"('{name}', {step.__class__.__name__})" for name, step in self.steps])
        pipeline_type = "transformer-only" if self._is_transformer_only else "with estimator"
        return f"Pipeline({pipeline_type})[\n    {steps_str}\n]"
    

# ------------------------------------------------------------------------------------------
# FUNCTIONS FOR PIPELINE EVALUATION 
# ------------------------------------------------------------------------------------------

def relative_performance(minimize : bool, x1 : float, x2 : float) -> float:
    """
    Compute the relative performance of a pipeline with score x2 with respect to another of score x1 (reference).
    The same function can be used to compute overfitting.
    """
    if minimize:
        performance = round(((x1 - x2)/(x1)) * 100 ,2)
    else:
        performance = round(((x2 - x1)/(x1)) * 100 ,2)

    return performance


def train_pipeline(pipeline: Pipeline, train: pl.LazyFrame) -> Pipeline:
    """Train the pipeline on training data."""
    pipeline.fit(train)
    return pipeline


def predict_with_pipeline(pipeline: Pipeline, data: pl.LazyFrame) -> np.array:
    """Generate predictions using the pipeline."""
    return pipeline.predict(data)


def compute_score(
    data: pl.LazyFrame, 
    preds: np.array, 
    metric: Metric, 
    target: str
) -> float:
    """Compute metric score for predictions."""
    data_with_preds = data.with_columns(pl.Series(preds).alias('preds'))
    return metric.compute_metric(lf=data_with_preds, target=target, preds='preds')


@log_execution_time
def eval_pipeline_single_fold(
    pipeline : Pipeline,
    train : pl.LazyFrame, 
    valid : pl.LazyFrame, 
    metric : Metric, 
    target : str,
    minimize : bool, 
    eval_overfitting : bool = True, 
    store_preds : bool = True, 
    verbose : bool = True
) -> Dict[str, Union[float, np.array]]:
    """
    Evalute pipeline performance by training on the train dataset and validate the prediction on valid dataset. 
    """
    
    with log_step('Training', verbose):
        _, duration_train = time_execution(train_pipeline)(pipeline, train)
    
    with log_step('Inference', verbose):
        preds, duration_inf = time_execution(predict_with_pipeline)(pipeline, valid)
    
    score = compute_score(valid, preds, metric, target)
    
    if eval_overfitting:
        with log_step('Computing Overfitting', verbose):
            train_preds = predict_with_pipeline(pipeline, train)
            score_on_train = compute_score(train, train_preds, metric, target)
            overfitting = relative_performance(minimize, score, score_on_train)
    else:
        score_on_train = np.nan
        overfitting = np.nan
    
    return {
        'validation_score': score, 
        'train_score': score_on_train, 
        'overfitting': overfitting, 
        'duration_train': duration_train, 
        'duration_inf': duration_inf, 
        'preds': list(preds) if store_preds else np.nan
    }


def eval_pipeline_cv(
    pipeline : Pipeline,
    lz : pl.LazyFrame, 
    cv_indexes : List[Tuple[np.array]], 
    row_id : str,
    metric : Metric, 
    target : str,
    minimize : bool, 
    eval_overfitting : bool = True, 
    store_preds : bool = True, 
    verbose : bool = True
) -> pl.DataFrame:
    """
    Evalute pipeline performance in a cross-validation fashion, by using cv_indexes. 
    """
    
    fold_results = []
    for fold, (train_idx, valid_idx) in enumerate(cv_indexes):

        with log_step(f'Fold {fold+1}', verbose):

            train = lz.filter(pl.col(row_id).is_in(train_idx))
            valid = lz.filter(pl.col(row_id).is_in(valid_idx))
            results = eval_pipeline_single_fold(
                pipeline=pipeline, 
                train=train, 
                valid=valid,  
                metric=metric, 
                target=target,
                minimize=minimize, 
                eval_overfitting=eval_overfitting, 
                store_preds=store_preds, 
                verbose=verbose
            )

            fold_results.append(results)
    
    return pl.DataFrame(fold_results)