"""
Machine learning experimentation framework for systematic model evaluation and comparison.

Provides Lab class for running experiments with cross-validation, tracking results,
hyperparameter optimization, and statistical comparison of models.
"""

from typing import Any, Dict, List, Tuple, Union
import uuid
import os
from datetime import datetime
import time 
import pickle 
from dataclasses import dataclass 

import polars as pl 
import numpy as np

from empml.base import (
    DataDownloader, 
    Metric, 
    CVGenerator
)

from empml.base import BaseTransformer, SKlearnEstimator
from empml.errors import RunExperimentConfigException, RunExperimentOnTestException
from empml.transformers import Identity
from empml.pipeline import (
    Pipeline,
    eval_pipeline_single_fold,
    eval_pipeline_single_fold_multi,
    eval_pipeline_cv,
    eval_pipeline_cv_multi,
    relative_performance,
    compare_results_stats,
    compare_results_stats_multi,
)
from empml.wrappers import SKlearnWrapper
from empml.utils import log_execution_time, log_step
from empml.lab_utils import (
    setup_row_id_column,
    create_results_schema,
    create_results_schema_multi,
    create_results_details_schema,
    create_results_details_schema_multi,
    format_experiment_results,
    format_experiment_results_multi,
    format_experiment_details,
    format_experiment_details_multi,
    prepare_predictions_for_save,
    log_performance_against,
    log_performance_against_multi,
    format_log_performance,
    retrieve_predictions_from_path,
    generate_params_list,
    generate_shuffle_preds,
    compute_anomaly,
)

# --- Logging Setup ---
import logging 
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    force=True 
)

# streaming engine as the default for .collect()
pl.Config.set_engine_affinity(engine='streaming')

@dataclass 
class ComparisonCriteria:
    """
    Statistical criteria for comparing experiment performance.
    
    Choose either percentage threshold OR statistical testing approach.
    """
    n_folds_threshold: int
    pct_threshold: float | None = None
    alpha: float | None = None 
    n_iters: int | None = None
    
    def __post_init__(self):
        has_pct = self.pct_threshold is not None
        has_statistical = (self.alpha is not None) and (self.n_iters is not None)
        
        if not (has_pct or has_statistical):
            raise ValueError(
                "Must provide either 'pct_threshold' OR both 'alpha' and 'n_iters'"
            )
        
        if has_pct and has_statistical:
            raise ValueError(
                "Cannot provide both 'pct_threshold' and ('alpha', 'n_iters'). "
                "Choose one approach only."
            )
        
        self.has_pct = has_pct 
        self.has_statistical = has_statistical


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
    """
    Experimentation framework for ML model development and evaluation.
    
    Manages experiment lifecycle: data loading, CV splitting, pipeline execution,
    results tracking, and statistical comparison. Supports HPO and feature selection.
    """
    
    # ------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------

    def __init__(
        self,
        train_downloader: DataDownloader,
        metric: Metric | List[Metric],
        cv_generator: CVGenerator,
        target: str,
        comparison_criteria: ComparisonCriteria,
        minimize: bool | List[bool] = True,
        row_id: str | None = None,
        test_downloader: DataDownloader | None = None,
        name: str | None = None,
    ):
        """
        Initialize Lab with data, evaluation metric(s), and CV strategy.

        Args:
            train_downloader: Source for training data
            metric: Single metric or list of metrics for evaluation
            cv_generator: Cross-validation splitting strategy
            target: Name of target column
            comparison_criteria: Criteria for experiment comparison
            minimize: Whether to minimize metric(s). Single bool or
                list of bools matching length of metric list.
            row_id: Column name for row identifier
            test_downloader: Optional test data source
            name: Lab identifier (auto-generated if None)
        """
        self.name = name or uuid.uuid1().hex[:8]
        self.cv_generator = cv_generator
        self.target = target

        self.train_downloader = train_downloader
        self.test_downloader = test_downloader

        # Multi-metric setup
        self._multi_metric = isinstance(metric, list)
        if self._multi_metric:
            self.metrics = metric
            self.n_metrics = len(metric)
            self.metric = metric[0]
            if isinstance(minimize, list):
                if len(minimize) != self.n_metrics:
                    raise ValueError(
                        f"len(minimize)={len(minimize)} != "
                        f"len(metric)={self.n_metrics}"
                    )
                self.minimize_list = minimize
            else:
                self.minimize_list = [minimize] * self.n_metrics
            self.minimize = self.minimize_list[0]
        else:
            self.metrics = [metric]
            self.n_metrics = 1
            self.metric = metric
            if isinstance(minimize, bool):
                self.minimize = minimize
                self.minimize_list = [minimize]
            else:
                self.minimize = minimize[0]
                self.minimize_list = minimize

        self._setup_directories()
        self._load_data(train_downloader, test_downloader)
        self._setup_row_id(row_id)
        self._setup_results_tracking()

        self.cv_indexes = self.cv_generator.split(
            self.train, self.row_id
        )
        self.n_folds = len(self.cv_indexes)

        self._set_eval_params(
            comparison_criteria=comparison_criteria
        )

        self.next_experiment_id = 1
        self._set_best_experiment()

    def _setup_directories(self):
        """Create directory structure for lab artifacts."""
        base = f'./{self.name}'
        os.makedirs(f'{base}/pipelines', exist_ok=True)
        os.makedirs(f'{base}/predictions', exist_ok=True)
        os.makedirs(f'{base}/check_points', exist_ok=True)

    def _load_data(self, train_downloader, test_downloader):
        """Load train and optional test datasets."""
        self.train = train_downloader.get_data()
        self.test = test_downloader.get_data() if test_downloader else None

    def _setup_row_id(self, row_id):
        """Initialize or create row identifier column."""
        self.train, self.row_id = setup_row_id_column(self.train, row_id)

    def _setup_results_tracking(self):
        """Create empty DataFrames for experiment tracking."""
        if self._multi_metric:
            self.results = create_results_schema_multi(
                self.n_metrics
            )
            self.results_details = (
                create_results_details_schema_multi(
                    self.n_metrics
                )
            )
        else:
            self.results = create_results_schema()
            self.results_details = (
                create_results_details_schema()
            )

    def _set_eval_params(self, comparison_criteria : ComparisonCriteria):
        """Configure evaluation mode (percentage vs statistical)."""
        self.n_folds_threshold = comparison_criteria.n_folds_threshold
        self.pct_threshold = comparison_criteria.pct_threshold 
        self.alpha = comparison_criteria.alpha
        self.n_iters = comparison_criteria.n_iters

        # Set evaluation mode flags
        if comparison_criteria.has_pct:  
            self.eval_has_pct = True
            self.eval_has_statistical = False
        else:  
            self.eval_has_pct = False
            self.eval_has_statistical = True

    def _set_best_experiment(self, experiment_id : int | None = None):
        """Set or clear best experiment tracker."""
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
        compare_against: int | None = None, 
        auto_mode : bool = False
    ):
        """
        Execute pipeline evaluation with CV and track results.
        
        Args:
            pipeline: Pipeline to evaluate
            eval_overfitting: Whether to check train/valid gap
            store_preds: Whether to save predictions
            verbose: Enable detailed logging
            compare_against: Experiment ID to compare against
            auto_mode: Auto-update best experiment if improvement found
        """
        # Validate configuration
        if auto_mode and not(self.best_experiment): 
            raise RunExperimentConfigException(
                "Select a best experiment before using auto_mode."
            )
        
        if auto_mode:
            logging.info("Auto mode: comparing against current best.")
            compare_against = self.best_experiment
        
        # Build comparison DataFrame
        compare_df = (
            self.results_details.filter(
                pl.col('experiment_id') == compare_against
            )
            if compare_against
            else pl.DataFrame()
        )

        # Run CV evaluation
        if self._multi_metric:
            eval = eval_pipeline_cv_multi(
                pipeline=pipeline,
                lz=self.train,
                cv_indexes=self.cv_indexes,
                row_id=self.row_id,
                metrics=self.metrics,
                target=self.target,
                minimize=self.minimize_list,
                eval_overfitting=eval_overfitting,
                store_preds=store_preds,
                verbose=verbose,
                compare_df=compare_df,
                th_lower_performance_n_folds=(
                    self.n_folds_threshold
                ),
            )
        else:
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
                compare_df=compare_df,
                th_lower_performance_n_folds=(
                    self.n_folds_threshold
                ),
            )

        # Update tracking tables
        self._update_results_table(eval=eval, description=pipeline.description, name=pipeline.name)
        self._update_details_table(eval=eval)
        
        # Save artifacts
        self._save_pipeline(pipeline=pipeline)
        self._save_predictions(eval=eval)

        # Handle comparison and auto-update
        if compare_against and eval.shape[0] == self.n_folds:
            self._log_compare_experiments(experiment_ids=(compare_against, self.next_experiment_id))

            if auto_mode:
                self._update_best_experiment(experiment_ids=(compare_against, self.next_experiment_id))
                logging.info(f"{BLUE}{BOLD}BEST EXPERIMENT UPDATED: {self.best_experiment}{RESET}")

        elif eval.shape[0] < self.n_folds:
            logging.info(f"{BOLD}{RED}Experiment arrested: no improvement over baseline.{RESET}")
        
        self.next_experiment_id += 1

    def _update_results_table(
        self,
        eval: pl.DataFrame,
        description: str = '',
        name: str = '',
    ):
        """Append experiment summary to results table."""
        is_completed = eval.shape[0] == self.n_folds
        if self._multi_metric:
            tmp = format_experiment_results_multi(
                eval,
                self.next_experiment_id,
                is_completed,
                self.n_metrics,
                description,
                name,
            )
        else:
            tmp = format_experiment_results(
                eval,
                self.next_experiment_id,
                is_completed,
                description,
                name,
            )
        self.results = pl.concat(
            [self.results, tmp.select(self.results.columns)],
            how='vertical_relaxed',
        )

    def _update_details_table(self, eval: pl.DataFrame):
        """Append fold-level details to results table."""
        if self._multi_metric:
            tmp = format_experiment_details_multi(
                eval,
                self.next_experiment_id,
                self.n_metrics,
            )
        else:
            tmp = format_experiment_details(
                eval, self.next_experiment_id
            )
        self.results_details = pl.concat(
            [
                self.results_details,
                tmp.select(self.results_details.columns),
            ],
            how='vertical_relaxed',
        )

    def _save_pipeline(self, pipeline: Pipeline):
        """Serialize pipeline to disk."""
        pickle.dump(
            pipeline,
            open(f'./{self.name}/pipelines/pipeline_{self.next_experiment_id}.pkl', 'wb')
        )

    @log_execution_time
    def _save_predictions(self, eval: pl.DataFrame):
        """Save predictions as compressed parquet."""
        preds = prepare_predictions_for_save(eval)
        preds.write_parquet(
            f'./{self.name}/predictions/predictions_{self.next_experiment_id}.parquet',
            compression='zstd',
            compression_level=22
        )

    def _update_best_experiment(
        self, experiment_ids: Tuple[int, int]
    ):
        """Update best experiment if new one outperforms."""
        idx_a, idx_b = experiment_ids
        results_a = self.results_details.filter(
            pl.col('experiment_id') == idx_a
        )
        results_b = self.results_details.filter(
            pl.col('experiment_id') == idx_b
        )

        if self._multi_metric:
            self._update_best_multi(
                idx_a, idx_b, results_a, results_b,
                experiment_ids,
            )
        else:
            self._update_best_single(
                idx_a, idx_b, results_a, results_b,
                experiment_ids,
            )

    def _update_best_single(
        self,
        idx_a: int,
        idx_b: int,
        results_a: pl.DataFrame,
        results_b: pl.DataFrame,
        experiment_ids: Tuple[int, int],
    ):
        """Single-metric best experiment update."""
        comparison = compare_results_stats(
            results_a, results_b, minimize=self.minimize
        )
        if self.eval_has_pct:
            c1 = comparison['mean_cv_performance'] > (
                self.pct_threshold
            )
            c2 = comparison['n_folds_lower_performance'] <= (
                self.n_folds_threshold
            )
            if c1 and c2:
                self.best_experiment = idx_b
        else:
            pvalue = self.compute_pvalue(
                experiment_ids=experiment_ids,
                n_iters=self.n_iters,
            )
            c1 = comparison['mean_cv_performance'] > 0
            c2 = comparison['n_folds_lower_performance'] <= (
                self.n_folds_threshold
            )
            c3 = pvalue < self.alpha
            if c1 and c2 and c3:
                self.best_experiment = idx_b

    def _update_best_multi(
        self,
        idx_a: int,
        idx_b: int,
        results_a: pl.DataFrame,
        results_b: pl.DataFrame,
        experiment_ids: Tuple[int, int],
    ):
        """Multi-metric: B is better only if ALL metrics pass."""
        comparisons = compare_results_stats_multi(
            results_a, results_b,
            self.minimize_list, self.n_metrics,
        )
        if self.eval_has_pct:
            all_pass = all(
                c['mean_cv_performance'] > self.pct_threshold
                and c['n_folds_lower_performance']
                <= self.n_folds_threshold
                for c in comparisons
            )
            if all_pass:
                self.best_experiment = idx_b
        else:
            pvalues = self.compute_pvalue(
                experiment_ids=experiment_ids,
                n_iters=self.n_iters,
            )
            all_pass = all(
                c['mean_cv_performance'] > 0
                and c['n_folds_lower_performance']
                <= self.n_folds_threshold
                and pv < self.alpha
                for c, pv in zip(comparisons, pvalues)
            )
            if all_pass:
                self.best_experiment = idx_b

    def _log_compare_experiments(
        self, experiment_ids: Tuple[int, int]
    ):
        """Log comparison statistics between two experiments."""
        idx_a, idx_b = experiment_ids
        results_a = self.results_details.filter(
            pl.col('experiment_id') == idx_a
        )
        results_b = self.results_details.filter(
            pl.col('experiment_id') == idx_b
        )
        if self._multi_metric:
            comparisons = compare_results_stats_multi(
                results_a, results_b,
                self.minimize_list, self.n_metrics,
            )
            log_performance_against_multi(
                comparisons=comparisons,
                n_folds_threshold=self.n_folds_threshold,
                n_metrics=self.n_metrics,
            )
        else:
            comparison = compare_results_stats(
                results_a, results_b,
                minimize=self.minimize,
            )
            log_performance_against(
                comparison=comparison,
                n_folds_threshold=self.n_folds_threshold,
            )

    # ------------------------------------------------------------------------------------------
    # MULTI-EXPERIMENTS
    # ------------------------------------------------------------------------------------------
    
    def multi_run_experiment(
        self,
        pipelines: List[Pipeline],
        eval_overfitting : bool = True, 
        store_preds : bool = True, 
        verbose : bool = True,
        compare_against: int | None = None, 
        auto_mode : bool = False
    ):
        """Execute multiple experiments sequentially."""
        logging.info(f'{BOLD}{BLUE}Total experiments: {len(pipelines)}{RESET}')

        for i, pipeline in enumerate(pipelines):
            with log_step(f'{BOLD}{BLUE}Experiment {i+1}: {pipeline.name}{RESET}', verbose):
                self.run_experiment(
                    pipeline=pipeline, 
                    eval_overfitting=eval_overfitting, 
                    store_preds=store_preds,
                    verbose=verbose,
                    compare_against=compare_against, 
                    auto_mode=auto_mode
                )

    def run_base_experiments(
        self, 
        features: str, 
        preprocess_pipe : Pipeline | None = None, 
        eval_overfitting: bool = True, 
        store_preds: bool = True, 
        verbose: bool = True,
        compare_against: int | None = None,
        problem_type: str = 'regression'
    ):
        """
        Run suite of baseline models for quick benchmarking.
        
        Args:
            features: Feature columns to use
            preprocess_pipe: Optional preprocessing pipeline
            problem_type: 'regression' or 'classification'
        """
        from sklearn.pipeline import Pipeline as SKlearnPipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        is_classification = problem_type.lower() == 'classification'
        
        # Import models based on problem type
        if is_classification:
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
                'hgb_base': hgb(),
                'mlp_base': SKlearnPipeline([
                    ('imputer', SimpleImputer()), 
                    ('scaler', StandardScaler()),
                    ('clf', mlp(hidden_layer_sizes=(64,32)))
                ]),
            }  
        else:
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
                'hgb_base': hgb(),
                'mlp_base': SKlearnPipeline([
                    ('imputer', SimpleImputer()), 
                    ('scaler', StandardScaler()),
                    ('reg', mlp(hidden_layer_sizes=(64,32)))
                ]),
            }

        # Create experiment pipelines
        if preprocess_pipe:
            pipes = [
                Pipeline([
                    ('preprocess', preprocess_pipe),
                    ('model', SKlearnWrapper(estimator=estimator, features=features, target=self.target))
                ], 
                name=name, 
                description=f'{name} with features = {features}'
                )
                for name, estimator in estimators.items()
            ]
        else:
            pipes = [
                Pipeline([
                    ('model', SKlearnWrapper(estimator=estimator, features=features, target=self.target))
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
    # HPO
    # ------------------------------------------------------------------------------------------

    def hpo(
        self,
        features: List[str],
        params_list: Dict[str, List[float | int | str]],
        estimator: SKlearnEstimator,
        preprocessor: Pipeline | BaseTransformer = Identity(),
        eval_overfitting: bool = True,
        store_preds: bool = True,
        verbose: bool = True,
        compare_against: int | None = None,
        search_type: str = 'grid',
        num_samples: int = 64,
        random_state: int = 0,
        primary_metric_idx: int | str = 'all',
    ):
        """
        Hyperparameter optimization via grid or random search.

        Args:
            features: Features to use in model
            params_list: Parameter grid/ranges
            estimator: sklearn-compatible estimator class
            preprocessor: Optional preprocessing step
            search_type: 'grid' or 'random'
            num_samples: Number of random samples (random search)
            primary_metric_idx: For multi-metric, which metric
                to sort by (0-indexed int) or 'all' to require
                improvement on every metric. Ignored for single
                metric.
        """
        # Validate primary_metric_idx for multi-metric
        if self._multi_metric and primary_metric_idx != 'all':
            if not isinstance(primary_metric_idx, int):
                raise ValueError(
                    "primary_metric_idx must be int or 'all'"
                )
            if (
                primary_metric_idx < 0
                or primary_metric_idx >= self.n_metrics
            ):
                raise ValueError(
                    f"primary_metric_idx={primary_metric_idx}"
                    f" out of range "
                    f"[0, {self.n_metrics - 1}]"
                )

        if (
            self._multi_metric
            and primary_metric_idx == 'all'
        ):
            logging.warning(
                "primary_metric_idx='all': HPO will only "
                "select experiments that improve on ALL "
                "metrics. This may yield no results."
            )

        # Generate parameter combinations
        pipelines = [
            Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('estimator', SKlearnWrapper(
                        estimator=estimator(**p),
                        features=features,
                        target=self.target,
                    )),
                ],
                name=f'{repr(estimator(**p))}',
                description=(
                    f'{repr(estimator(**p))} with '
                    f'features={features} and '
                    f'preprocessor={repr(preprocessor)}'
                ),
            )
            for p in generate_params_list(
                params_list=params_list,
                search_type=search_type,
                num_samples=num_samples,
                random_state=random_state,
            )
        ]

        n_experiments = len(pipelines)

        self.multi_run_experiment(
            pipelines=pipelines,
            eval_overfitting=eval_overfitting,
            store_preds=store_preds,
            verbose=verbose,
            compare_against=compare_against,
        )

        # Select best HPO result
        return self._select_hpo_best(
            n_experiments=n_experiments,
            primary_metric_idx=primary_metric_idx,
            compare_against=compare_against,
        )

    def _select_hpo_best(
        self,
        n_experiments: int,
        primary_metric_idx: int | str,
        compare_against: int | None,
    ) -> Dict | None:
        """Select best experiment from HPO results."""
        candidates = (
            self.results
            .tail(n_experiments)
            .filter(pl.col('is_completed') == True)
        )

        if not self._multi_metric:
            sort_col = 'cv_mean_score'
            sort_desc = self.minimize
        elif primary_metric_idx == 'all':
            # Filter to experiments better on ALL metrics
            candidates = self._filter_all_metrics_better(
                candidates, compare_against
            )
            if candidates.height == 0:
                logging.warning(
                    "No HPO experiment improves on all "
                    "metrics. Returning None."
                )
                return None
            sort_col = 'cv_mean_score_1'
            sort_desc = self.minimize_list[0]
        else:
            idx = primary_metric_idx + 1
            sort_col = f'cv_mean_score_{idx}'
            sort_desc = self.minimize_list[primary_metric_idx]

        hpo_results = (
            candidates
            .sort(sort_col, descending=sort_desc)
            .tail(1)
            .row(0, named=True)
        )
        hpo_results['best_pipeline'] = (
            self.retrieve_pipeline(
                experiment_id=hpo_results['experiment_id']
            )
        )
        return hpo_results

    def _filter_all_metrics_better(
        self,
        candidates: pl.DataFrame,
        compare_against: int | None,
    ) -> pl.DataFrame:
        """Filter candidates that improve on ALL metrics."""
        if not compare_against:
            return candidates

        baseline = self.results.filter(
            pl.col('experiment_id') == compare_against
        )
        if baseline.height == 0:
            return candidates

        # Keep only experiments better on every metric
        for i, mini in enumerate(self.minimize_list, 1):
            col = f'cv_mean_score_{i}'
            baseline_val = baseline[col].item()
            if mini:
                candidates = candidates.filter(
                    pl.col(col) < baseline_val
                )
            else:
                candidates = candidates.filter(
                    pl.col(col) > baseline_val
                )
        return candidates

    # ------------------------------------------------------------------------------------------
    # RETRIEVE PIPELINE PREDICTIONS
    # ------------------------------------------------------------------------------------------

    def retrieve_predictions(self, experiment_ids = List[int], extra_features : List[str] = []) -> pl.LazyFrame:
        """
        Load predictions from specified experiments.
        
        Returns LazyFrame with row_id, fold, target, and predictions from each experiment.
        """
        # Create base frame with fold assignments
        base_preds = pl.concat([
            pl.LazyFrame(self.cv_indexes[j][1], schema=[self.row_id])
              .with_columns(pl.lit(j+1).alias('fold_number')) 
            for j in range(len(self.cv_indexes))
        ], how='vertical_relaxed')

        base_preds = base_preds.join(
            self.train.select([self.row_id, self.target] + extra_features), 
            how='left', 
            on=self.row_id
        )

        # Add predictions from each experiment
        preds = base_preds.with_columns(
            retrieve_predictions_from_path(lab_name=self.name, experiment_id=idx) 
            for idx in experiment_ids
        )

        return preds

    def compute_pvalue(
        self,
        experiment_ids: Tuple[int, int],
        n_iters: int = 200,
        extra_features: List[str] = [],
    ) -> Union[float, List[float]]:
        """
        Compute permutation test p-value(s) comparing two experiments.

        Returns single float for single-metric, list of floats
        for multi-metric (one p-value per metric).
        """
        if (
            not isinstance(experiment_ids, tuple)
            or len(experiment_ids) != 2
        ):
            raise ValueError(
                'experiment_ids must be tuple of two IDs.'
            )

        preds = self.retrieve_predictions(
            experiment_ids=list(experiment_ids),
            extra_features=extra_features,
        )
        idx_1, idx_2 = experiment_ids

        if not self._multi_metric:
            return self._compute_pvalue_single(
                preds, idx_1, idx_2, n_iters
            )

        # Multi-metric: one p-value per metric
        pvalues = []
        for metric in self.metrics:
            pv = self._compute_pvalue_for_metric(
                metric, preds, idx_1, idx_2, n_iters
            )
            pvalues.append(pv)
        return pvalues

    def _compute_pvalue_single(
        self,
        preds: pl.LazyFrame,
        idx_1: int,
        idx_2: int,
        n_iters: int,
    ) -> float:
        """Compute p-value for single-metric case."""
        return self._compute_pvalue_for_metric(
            self.metric, preds, idx_1, idx_2, n_iters
        )

    def _compute_pvalue_for_metric(
        self,
        metric: Metric,
        preds: pl.LazyFrame,
        idx_1: int,
        idx_2: int,
        n_iters: int,
    ) -> float:
        """Compute permutation p-value for one metric."""
        obs_anomaly = compute_anomaly(
            metric=metric,
            lf=preds,
            preds_1=f'preds_{idx_1}',
            preds_2=f'preds_{idx_2}',
            target=self.target,
        )
        sim_anomaly = [
            compute_anomaly(
                metric=metric,
                lf=generate_shuffle_preds(
                    lf=preds,
                    preds_1=f'preds_{idx_1}',
                    preds_2=f'preds_{idx_2}',
                    random_state=i,
                ),
                preds_1='shuffle_a',
                preds_2='shuffle_b',
                target=self.target,
            )
            for i in range(n_iters)
        ]
        r = (np.array(sim_anomaly) > obs_anomaly).sum()
        return (r + 1) / (n_iters + 1)
        
    def permutation_feature_importance(
        self, 
        pipeline : Pipeline, 
        features : List[str],
        n_iters : int = 5, 
        verbose : bool = True      
    ) -> pl.DataFrame:
        """
        Compute permutation feature importance for each feature.

        Measures performance drop when feature is randomly shuffled.
        Returns DataFrame with importance scores per fold.

        Note:
            For multi-metric Labs, uses the first metric only.
        """
        pfi_dfs = []
        for fold, (train_idx, valid_idx) in enumerate(self.cv_indexes):
            with log_step(f'Fold {fold+1}', verbose):
                train = self.train.filter(pl.col(self.row_id).is_in(train_idx))
                valid = self.train.filter(pl.col(self.row_id).is_in(valid_idx))

                pipeline.fit(train)
                valid = valid.with_columns(pl.Series(pipeline.predict(valid)).alias('base_preds'))

                pfi = {}
                for f in features:
                    # Generate shuffled predictions
                    shadow = valid.with_columns(
                        pl.Series(pipeline.predict(
                            valid.with_columns(pl.col(f).sample(fraction=1, seed=j, shuffle=True))
                        )).alias(f'shadow_{j}') 
                        for j in range(n_iters)
                    )

                    # Compare base vs shuffled performance
                    base_metric = self.metric.compute_metric(shadow, target=self.target, preds='base_preds')
                    shadow_metric = np.array([
                        self.metric.compute_metric(shadow, target=self.target, preds=f'shadow_{j}') 
                        for j in range(n_iters)
                    ]).mean()

                    pfi[f] = relative_performance(minimize=not(self.minimize), x1=base_metric, x2=shadow_metric)

                pfi_dfs.append(pl.DataFrame(pfi).with_columns(pl.lit(fold+1).alias('fold_number')))

        return pl.concat(pfi_dfs)

    def recursive_permutation_feature_selection(
        self, 
        estimator : SKlearnEstimator, 
        features : List[str], 
        preprocessor : Pipeline | BaseTransformer = Identity(), 
        n_iters : int = 5, 
        verbose : bool = True
    ) -> List[str]:
        """
        Recursively eliminate features with negative importance.
        
        Returns list of selected features after iterative removal.
        """
        pipeline = Pipeline([
            ('preprocessor', preprocessor), 
            ('estimator', SKlearnWrapper(estimator=estimator, features=features, target=self.target))
        ])

        pfi = self.permutation_feature_importance(pipeline=pipeline, features=features, n_iters=n_iters, verbose=verbose)
        pfi_final = pfi.drop('fold_number').mean().transpose(include_header=True, header_name='features', column_names=['pfi'])
        features_to_drop = pfi_final.filter(pl.col('pfi') <= 0)['features'].to_list()

        if len(features_to_drop) > 0:
            new_features = [f for f in features if f not in features_to_drop]
            logging.info(f"Features eliminated: {features_to_drop}")
            return self.recursive_permutation_feature_selection(
                preprocessor=preprocessor, 
                estimator=estimator, 
                features=new_features, 
                n_iters=n_iters, 
                verbose=verbose
            )
        else:
            logging.info("No features eliminated.")
            return features
        
    
    def run_experiment_on_test(
        self,
        experiment_id: int,
        eval_overfitting: bool = True,
        store_preds: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Union[float, List[float]]]:
        """Evaluate experiment pipeline on the test set.

        For multi-metric Labs, evaluates and prints stats
        for each metric independently.
        """
        if not self.test_downloader:
            raise RunExperimentOnTestException(
                'No test set detected. Please provide one '
                'through a downloader data class before '
                'final pipeline evaluation.'
            )

        path = (
            f"./{self.name}/pipelines/"
            f"pipeline_{experiment_id}.pkl"
        )
        pipe = pickle.load(open(path, 'rb'))

        if self._multi_metric:
            test_results = eval_pipeline_single_fold_multi(
                pipeline=pipe,
                train=self.train,
                valid=self.test,
                metrics=self.metrics,
                target=self.target,
                minimize=self.minimize_list,
                eval_overfitting=eval_overfitting,
                store_preds=store_preds,
                verbose=verbose,
            )
            cv_results = self.results.filter(
                pl.col('experiment_id') == experiment_id
            )
            for i in range(self.n_metrics):
                idx = i + 1
                cv_mean = cv_results[
                    f'cv_mean_score_{idx}'
                ].item()
                cv_std = cv_results[
                    f'cv_std_score_{idx}'
                ].item()
                test_score = test_results[
                    f'validation_score_{idx}'
                ]
                rel = relative_performance(
                    self.minimize_list[i],
                    cv_mean,
                    test_score,
                )
                print(
                    f'Metric {idx} - CV vs Test: '
                    f'{format_log_performance(rel, 0)}'
                )
                max_ = cv_mean + 2 * cv_std
                min_ = cv_mean - 2 * cv_std
                in_interval = (
                    min_ <= test_score <= max_
                )
                if in_interval:
                    print(
                        f'{BOLD}{GREEN} Metric {idx}: '
                        f'test score within '
                        f'μ ± 2σ{RESET}'
                    )
                else:
                    print(
                        f'{BOLD}{RED} Metric {idx}: '
                        f'test score NOT within '
                        f'μ ± 2σ{RESET}'
                    )
        else:
            test_results = eval_pipeline_single_fold(
                pipeline=pipe,
                train=self.train,
                valid=self.test,
                metric=self.metric,
                target=self.target,
                minimize=self.minimize,
                eval_overfitting=eval_overfitting,
                store_preds=store_preds,
                verbose=verbose,
            )
            cv_results = self.results.filter(
                pl.col('experiment_id') == experiment_id
            )
            relative_perf = relative_performance(
                minimize=self.minimize,
                x1=cv_results['cv_mean_score'].item(),
                x2=test_results['validation_score'],
            )
            print(
                f'Difference in performance CV vs Test: '
                f'{format_log_performance(relative_perf, 0)}'
            )
            max_ = (
                cv_results['cv_mean_score'].item()
                + 2 * cv_results['cv_std_score'].item()
            )
            min_ = (
                cv_results['cv_mean_score'].item()
                - 2 * cv_results['cv_std_score'].item()
            )
            isin_interval = (
                test_results['validation_score'] >= min_
                and test_results['validation_score'] < max_
            )
            if isin_interval:
                print(
                    f'{BOLD}{GREEN} The test score is '
                    f'between μ ± 2σ{RESET}'
                )
            else:
                print(
                    f'{BOLD}{RED} The test score is NOT '
                    f'between μ ± 2σ{RESET}'
                )

        return test_results


    def retrieve_pipeline(self, experiment_id : int) -> Pipeline:
        """Retrieve a pipeline related to an experiment"""
        path = f'./{self.name}/pipelines/pipeline_{experiment_id}.pkl'
        return pickle.load(open(path, 'rb'))
    
    def show_best_score(
        self, metric_idx: int | None = None
    ) -> pl.DataFrame:
        """Show stats for the experiment with the best score.

        Args:
            metric_idx: For multi-metric Labs, which metric
                to sort by (0-indexed). Defaults to first.
        """
        if not self._multi_metric:
            return (
                self.results
                .filter(pl.col('is_completed') == True)
                .sort(
                    'cv_mean_score',
                    descending=self.minimize,
                )
                .tail(1)
            )

        idx = (metric_idx or 0) + 1
        col = f'cv_mean_score_{idx}'
        desc = self.minimize_list[metric_idx or 0]
        return (
            self.results
            .filter(pl.col('is_completed') == True)
            .sort(col, descending=desc)
            .tail(1)
        )

    def save_check_point(self, check_point_name : str | None = None) -> None:
        """Serialize current lab state to disk."""
        check_point_name_ref = check_point_name if check_point_name else str(int(time.time()))
        pickle.dump(self, open(f'./{self.name}/check_points/{check_point_name_ref}.pkl', 'wb'))


def restore_check_point(lab_name : str, check_point_name : str) -> Lab:
    """Load saved lab state from checkpoint."""
    return pickle.load(open(f'./{lab_name}/check_points/{check_point_name}.pkl', 'rb'))