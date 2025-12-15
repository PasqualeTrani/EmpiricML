from typing import Any, Dict, List
import uuid
import os
from datetime import datetime
import pytz # type:ignore
import pickle 
from dataclasses import dataclass 
import math

import polars as pl # type:ignore

from empml.base import (
    DataDownloader, 
    Metric, 
    CVGenerator
)

from empml.pipelines import Pipeline, eval_pipeline_cv, relative_performance, compare_results_stats
from empml.estimators import RegressorWrapper, ClassifierWrapper
from empml.utils import log_execution_time, log_step
from empml.lab_utils import (
    setup_row_id_column, 
    create_results_schema, 
    create_results_details_schema, 
    format_experiment_results, 
    format_experiment_details, 
    prepare_predictions_for_save, 
    log_performance_against
)

# --- Logging Setup ---
import logging 
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass 
class EvalParams:
    kfold_threshold : float = 0.6

# ANSI escape codes for colors in print and logging 
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ------------------------------------------------------------------------------------------
# Lab Class
# ------------------------------------------------------------------------------------------

class Lab:
    # ------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------

    def __init__(
        self,
        train_downloader: DataDownloader,
        metric: Metric,
        cv_generator: CVGenerator,
        target: str,
        eval_params : EvalParams,
        minimize: bool = True,
        row_id: str | None = None,
        test_downloader: DataDownloader | None = None,
        name: str | None = None
    ):
        self.name = name or uuid.uuid1().hex[:8]
        self.metric = metric
        self.cv_generator = cv_generator
        self.target = target
        self.minimize = minimize
        
        self._setup_directories()
        self._load_data(train_downloader, test_downloader)
        self._setup_row_id(row_id)
        self._setup_results_tracking()
        
        self.cv_indexes = self.cv_generator.split(self.train, self.row_id)
        self.n_folds = len(self.cv_indexes)

        self.eval_params = eval_params
        self.n_folds_threshold = math.floor((1-self.eval_params.kfold_threshold) * self.n_folds)
                                               
        self.next_experiment_id = 1 
        self.best_experiment = None

    def _setup_directories(self):
        """Create lab directory structure."""
        base = f'./{self.name}'
        os.makedirs(f'{base}/pipelines', exist_ok=True)
        os.makedirs(f'{base}/predictions', exist_ok=True)

    def _load_data(self, train_downloader, test_downloader):
        """Load training and test data."""
        self.train = train_downloader.get_data()
        self.test = test_downloader.get_data() if test_downloader else None

    def _setup_row_id(self, row_id):
        """Setup row identifier column."""
        self.train, self.row_id = setup_row_id_column(self.train, row_id)

    def _setup_results_tracking(self):
        """Initialize Polars DataFrames for tracking experiments."""
        self.results = create_results_schema()
        self.results_details = create_results_details_schema()

    def _set_best_experiment(self, experiment_id : int):
        self.best_experiment = experiment_id

    # ------------------------------------------------------------------------------------------
    # Experiments Metrics
    # ------------------------------------------------------------------------------------------

    def run_experiment(
        self,
        pipeline: Pipeline,
        eval_overfitting : bool = True, 
        store_preds : bool = True, 
        verbose : bool = True,
        compare_against: int | None = None
    ):
        """Run an experiment and save all the metrics."""
        
        eval = eval_pipeline_cv(
            pipeline=pipeline, 
            lz=self.train, 
            cv_indexes=self.cv_indexes, 
            row_id=self.row_id,
            metric=self.metric, 
            target=self.target, 
            minimize=self.minimize, 
            eval_overfitting=eval_overfitting, 
            store_preds=store_preds,
            verbose=verbose, 
            compare_df=self.results_details.filter(pl.col('experiment_id')==compare_against) if compare_against else pl.DataFrame(), 
            th_lower_performance_n_folds=self.n_folds_threshold
        )

        self._update_results_table(eval=eval, description=pipeline.description, name=pipeline.name)

        # Add new rows to details table regarding the experiment
        self._update_details_table(eval=eval)

        # Save pipeline
        self._save_pipeline(pipeline=pipeline)

        # Save predictions
        self._save_predictions(eval=eval)

        if compare_against and eval.shape[0] == self.n_folds:
            self.compare_experiments(experiment_id_a = compare_against, experiment_id_b = self.next_experiment_id)
        elif eval.shape[0]<self.n_folds:
            logging.info(f"{BOLD}{RED}Experiment arrested since comparison showed no improvement over baseline.{RESET}")
        else:
            pass
        
        # Set new experiment_id for the next one
        self.next_experiment_id += 1

    def _update_results_table(self, eval: pl.DataFrame, description: str = '', name: str = ''):
        """Add row to results table regarding an experiment"""
        tmp = format_experiment_results(eval, self.next_experiment_id, eval.shape[0] == self.n_folds, description, name)
        self.results = pl.concat([
            self.results,
            tmp.select(self.results.columns)
        ], how='vertical_relaxed')

    def _update_details_table(self, eval: pl.DataFrame):
        """Add rows to results details table regarding an experiment"""
        tmp = format_experiment_details(eval, self.next_experiment_id)
        self.results_details = pl.concat([
            self.results_details,
            tmp.select(self.results_details.columns)
        ], how='vertical_relaxed')

    def _save_pipeline(self, pipeline: Pipeline):
        """Save pipeline used in an experiment"""
        pickle.dump(
            pipeline,
            open(f'./{self.name}/pipelines/pipeline_{self.next_experiment_id}.pkl', 'wb')
        )

    @log_execution_time
    def _save_predictions(self, eval: pl.DataFrame):
        """Save predictions created in an experiment to a parquet file"""
        preds = prepare_predictions_for_save(eval)
        preds.write_parquet(
            f'./{self.name}/predictions/predictions_{self.next_experiment_id}.parquet',
            compression='zstd',
            compression_level=22
        )

    def compare_experiments(self, experiment_id_a : int, experiment_id_b : int):
        """Compare results between two experiments"""
        results_a = self.results_details.filter(pl.col('experiment_id')==experiment_id_a)
        results_b = self.results_details.filter(pl.col('experiment_id')==experiment_id_b)
        comparison = compare_results_stats(results_a, results_b, minimize = self.minimize)
        log_performance_against(comparison = comparison, threshold = self.eval_params.kfold_threshold)

    # ------------------------------------------------------------------------------------------
    # MULTI-EXPERIMENTS
    # ------------------------------------------------------------------------------------------
    
    def multi_run_experiment(
        self,
        pipelines: List[Pipeline],
        eval_overfitting : bool = True, 
        store_preds : bool = True, 
        verbose : bool = True,
        compare_against: int | None = None
    ):
        """Run multiple experiments with a single command."""
        logging.info(f'{BOLD}{BLUE}Total Number of Experiments to run: {len(pipelines)}{RESET}')

        for i, pipeline in enumerate(pipelines):
            
            with log_step(f'{BOLD}{BLUE}Experiment {i+1} with name = {pipeline.name}{RESET}', verbose):

                self.run_experiment(
                    pipeline=pipeline, 
                    eval_overfitting=eval_overfitting, 
                    store_preds=store_preds,
                    verbose=verbose,
                    compare_against=compare_against
                )

    def run_base_experiments(
        self, 
        features: str, 
        preprocess_pipe : Pipeline | None = None, 
        eval_overfitting: bool = True, 
        store_preds: bool = True, 
        verbose: bool = True,
        compare_against: int | None = None,
        problem_type: str = 'regression'  # 'regression' or 'classification'
    ):
        """Run multiple experiments by using base estimators"""

        from sklearn.pipeline import Pipeline as SKlearnPipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Determine if classification or regression
        is_classification = problem_type.lower() == 'classification'
        
        if is_classification:
            # Classification imports
            from sklearn.linear_model import LogisticRegression as log_reg
            from sklearn.svm import SVC as svc
            from sklearn.neighbors import KNeighborsClassifier as knn
            from sklearn.ensemble import RandomForestClassifier as rf
            from lightgbm import LGBMClassifier as lgb
            from xgboost import XGBClassifier as xgb
            from catboost import CatBoostClassifier as ctb
            from sklearn.neural_network import MLPClassifier as mlp
            from sklearn.tree import DecisionTreeClassifier as dtree
            from sklearn.ensemble import HistGradientBoostingClassifier as hgb
            
            estimators = {
                'logistic_regression_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('clf', log_reg(max_iter=1000))
                ]),
                'knn_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('clf', knn())
                ]),
                'svm_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('clf', svc())
                ]),
                'random_forest_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('clf', rf(random_state=0, n_jobs=-1))
                ]),
                'decision_tree_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('clf', dtree())
                ]),
                'lightgbm_base': lgb(verbose=-1, random_state=0),
                'xgboost_base': xgb(verbosity=0, random_state=0),
                'catboost_base': ctb(verbose=0, random_state=0), 
                'hgb_base' : hgb(),
                'mlp_base' : SKlearnPipeline([
                    ('imputer', SimpleImputer()), 
                    ('scaler', StandardScaler()),
                    ('clf', mlp(hidden_layer_sizes = (64,32)))
                ]),
            }
            wrapper_class = ClassifierWrapper  
            
        else:
            # Regression imports
            from sklearn.linear_model import LinearRegression as lr
            from sklearn.svm import SVR as svr
            from sklearn.neighbors import KNeighborsRegressor as knn
            from sklearn.ensemble import RandomForestRegressor as rf
            from lightgbm import LGBMRegressor as lgb
            from xgboost import XGBRegressor as xgb
            from catboost import CatBoostRegressor as ctb
            from sklearn.neural_network import MLPRegressor as mlp
            from sklearn.tree import DecisionTreeRegressor as dtree
            from sklearn.ensemble import HistGradientBoostingRegressor as hgb
            
            estimators = {
                'linear_regression_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('reg', lr())
                ]),
                'knn_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('reg', knn())
                ]),
                'svm_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('scaler', StandardScaler()), 
                    ('reg', svr())
                ]),
                'random_forest_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('reg', rf(random_state=0, n_jobs=-1))
                ]),
                'decision_tree_base': SKlearnPipeline([
                    ('impute', SimpleImputer()), 
                    ('reg', dtree())
                ]),
                'lightgbm_base': lgb(verbose=-1),
                'xgboost_base': xgb(verbosity=0),
                'catboost_base': ctb(verbose=0), 
                'hgb_base' : hgb(),
                'mlp_base' : SKlearnPipeline([
                    ('imputer', SimpleImputer()), 
                    ('scaler', StandardScaler()),
                    ('reg', mlp(hidden_layer_sizes = (64,32)))
                ]),
            }
            wrapper_class = RegressorWrapper

        # Create pipelines
        if preprocess_pipe:
            pipes = [
                Pipeline([
                    ('preprocess', preprocess_pipe),
                    ('model', wrapper_class(estimator=estimator, features=features, target=self.target))
                ], 
                name=name, 
                description=f'{name} with features = {features}'
                )
                for name, estimator in estimators.items()
            ]
        else:
            pipes = [
                Pipeline([
                    ('model', wrapper_class(estimator=estimator, features=features, target=self.target))
                ], 
                name=name, 
                description=f'{name} with features = {features}'
                )
                for name, estimator in estimators.items()
            ]


        self.multi_run_experiment(
            pipelines=pipes, 
            eval_overfitting=eval_overfitting, 
            store_preds=store_preds, 
            verbose=verbose, 
            compare_against=compare_against
        )


    # ------------------------------------------------------------------------------------------
    # RETRIEVE PIPELINE PREDICTIONS
    # ------------------------------------------------------------------------------------------

    def retrieve_predictions(self, experiment_ids = List[int], extra_features : List[str] = []) -> pl.LazyFrame:
        """
        Retrieve the predictions made by all the pipelines indicated into experiment_ids
        """
        base_preds = pl.concat([
                pl.LazyFrame(self.cv_indexes[j][1], schema=[self.row_id]).with_columns(pl.lit(j+1).alias('fold_number')) 
                for j in range(len(self.cv_indexes))
            ], 
            how = 'vertical_relaxed'
        )

        base_preds = base_preds.join(self.train.select([self.row_id, self.target] + extra_features), how = 'left', on = self.row_id)

        preds = (
            base_preds
            .with_columns(
                pl.read_parquet(f'./{self.name}/predictions/predictions_{idx}.parquet').to_series().alias(f'preds_{idx}') 
                for idx in experiment_ids
            )
        )

        return preds